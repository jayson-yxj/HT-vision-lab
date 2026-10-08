from __future__ import annotations

import copy
import json
import math
import os
import resource
import shutil
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .face_tracks import Detection, _detect, _draw_box, box_iou, cosine_similarity
from .live_semantics import LiveSemanticWorker, SemanticTask
from .models import file_sha256, model_path, require_models
from .scene_semantics import GroqSceneAnalyzer, normalize_live_analysis


@dataclass
class OnlineTrack:
    track_id: str
    last_frame: int
    box: List[int]
    embedding: np.ndarray
    observations: int = 1
    face_id: Optional[str] = None
    reported_unassigned: bool = False


class OnlineFaceTracker:
    """Bounded session-local face tracking for live visual events."""

    def __init__(
        self,
        fps: float,
        *,
        max_gap_seconds: float = 0.8,
        reid_threshold: float = 0.75,
        cluster_threshold: float = 0.45,
        min_observations: int = 3,
        max_people: int = 4,
    ) -> None:
        self.max_gap_frames = max(1, round(max_gap_seconds * fps))
        self.reid_threshold = reid_threshold
        self.cluster_threshold = cluster_threshold
        self.min_observations = min_observations
        self.max_people = max_people
        self.tracks: Dict[str, OnlineTrack] = {}
        self.face_profiles: Dict[str, Tuple[np.ndarray, int]] = {}
        self.next_track = 1
        self.dropped_identity_candidates = 0

    @staticmethod
    def _normalized(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    def _new_track(self, detection: Detection, frame_index: int) -> OnlineTrack:
        track = OnlineTrack(
            track_id=f"live-track-{self.next_track:05d}",
            last_frame=frame_index,
            box=detection.box,
            embedding=self._normalized(detection.embedding),
        )
        self.next_track += 1
        self.tracks[track.track_id] = track
        return track

    def _update_profile(self, face_id: str, embedding: np.ndarray) -> None:
        previous, count = self.face_profiles[face_id]
        self.face_profiles[face_id] = (
            self._normalized(previous * count + embedding),
            count + 1,
        )

    def update(
        self,
        detections: Sequence[Detection],
        frame_index: int,
    ) -> List[Tuple[Detection, OnlineTrack]]:
        self.tracks = {
            identifier: track
            for identifier, track in self.tracks.items()
            if frame_index - track.last_frame <= self.max_gap_frames
        }
        tracks = list(self.tracks.values())
        candidates = []
        for track_index, track in enumerate(tracks):
            gap_ratio = (frame_index - track.last_frame) / self.max_gap_frames
            for detection_index, detection in enumerate(detections):
                similarity = cosine_similarity(track.embedding, detection.embedding)
                overlap = box_iou(track.box, detection.box)
                if similarity < 0.25 or (overlap < 0.20 and similarity < self.reid_threshold):
                    continue
                cost = 0.65 * (1 - similarity) + 0.30 * (1 - overlap) + 0.05 * gap_ratio
                candidates.append((cost, track_index, detection_index))
        matched_tracks, matched_detections, assignments = set(), set(), {}
        for _, track_index, detection_index in sorted(candidates):
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)
            assignments[detection_index] = tracks[track_index]

        current = []
        for detection_index, detection in enumerate(detections):
            track = assignments.get(detection_index)
            if track is None:
                track = self._new_track(detection, frame_index)
            else:
                track.embedding = self._normalized(
                    track.embedding * track.observations + detection.embedding
                )
                track.observations += 1
                track.last_frame = frame_index
                track.box = detection.box
            current.append((detection, track))

        visible_faces = {track.face_id for _, track in current if track.face_id}
        for detection, track in current:
            if track.face_id:
                self._update_profile(track.face_id, detection.embedding)
                continue
            if track.observations < self.min_observations:
                continue
            best_face, best_similarity = None, self.cluster_threshold
            for face_id, (embedding, _) in self.face_profiles.items():
                if face_id in visible_faces:
                    continue
                similarity = cosine_similarity(track.embedding, embedding)
                if similarity >= best_similarity:
                    best_face, best_similarity = face_id, similarity
            if best_face is None and len(self.face_profiles) < self.max_people:
                best_face = f"Face-{len(self.face_profiles) + 1:02d}"
                self.face_profiles[best_face] = (track.embedding.copy(), 1)
            if best_face is None:
                if not track.reported_unassigned:
                    self.dropped_identity_candidates += 1
                    track.reported_unassigned = True
                continue
            track.face_id = best_face
            visible_faces.add(best_face)
            self._update_profile(best_face, detection.embedding)
        return current


def _frame_histogram(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    small = small[:80]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    return cv2.normalize(histogram, histogram)


def _rss_mb() -> float:
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1024 / 1024
    except (OSError, ValueError, IndexError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return float(ordered[index])


class LiveVideoSession:
    def __init__(
        self,
        video_path: Path,
        output_dir: Path,
        *,
        sample_fps: float = 5.0,
        speed: float = 1.0,
        detection_threshold: float = 0.70,
        min_face_size: int = 24,
        max_gap_seconds: float = 0.8,
        reid_threshold: float = 0.75,
        cluster_threshold: float = 0.45,
        min_track_observations: int = 3,
        max_people: int = 4,
        scene_threshold: float = 0.45,
        min_scene_seconds: float = 1.0,
        keyframe_interval_seconds: float = 30.0,
        event_window: int = 100,
        semantic_enabled: bool = False,
        semantic_model: str = "qwen/qwen3.8-27b",
        semantic_timeout: float = 60.0,
        semantic_proxy: Optional[str] = None,
        semantic_minimum_interval: float = 0.2,
        semantic_queue_size: int = 8,
        semantic_analyzer: Optional[GroqSceneAnalyzer] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        if sample_fps <= 0 or not 0.1 <= speed <= 100:
            raise ValueError("sample_fps must be positive and speed must be between 0.1 and 100")
        if min_face_size < 1 or min_track_observations < 1 or max_people < 1:
            raise ValueError("face size, track observations and max people must be positive")
        if max_gap_seconds <= 0 or min_scene_seconds <= 0 or keyframe_interval_seconds <= 0:
            raise ValueError("gap, scene duration and keyframe interval must be positive")
        if not 0 <= scene_threshold <= 1:
            raise ValueError("scene threshold must be between zero and one")
        if not 1 <= event_window <= 1000:
            raise ValueError("event window must be between 1 and 1000")
        if semantic_queue_size < 1:
            raise ValueError("semantic queue size must be positive")
        for name, value in (
            ("detection_threshold", detection_threshold),
            ("reid_threshold", reid_threshold),
            ("cluster_threshold", cluster_threshold),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")

        require_models(["yunet", "sface"])
        self.video_path = video_path.expanduser().resolve()
        if not self.video_path.is_file():
            raise FileNotFoundError(self.video_path)
        self.output_dir = output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keyframe_dir = self.output_dir / "live_keyframes"
        if self.keyframe_dir.exists():
            shutil.rmtree(self.keyframe_dir)
        self.keyframe_dir.mkdir()
        self.event_path = self.output_dir / "vision_events.jsonl"
        self.event_path.write_text("", encoding="utf-8")
        self.semantic_path = self.output_dir / "live_scene_semantics.jsonl"
        self.semantic_path.write_text("", encoding="utf-8")
        self.video_sha256 = file_sha256(self.video_path)

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {self.video_path}")
        self.fps = float(capture.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if self.fps <= 0 or self.frame_count <= 0 or self.width <= 0 or self.height <= 0:
            raise RuntimeError("video metadata is incomplete")

        self.sample_step = max(1, round(self.fps / sample_fps))
        self.sample_fps = self.fps / self.sample_step
        self.sample_period_ms = max(1, round(1000 / self.sample_fps))
        self.duration_ms = round(self.frame_count * 1000 / self.fps)
        self.speed = float(speed)
        self.detection_threshold = detection_threshold
        self.min_face_size = min_face_size
        self.scene_threshold = scene_threshold
        self.min_scene_ms = round(min_scene_seconds * 1000)
        self.keyframe_interval_ms = round(keyframe_interval_seconds * 1000)
        self.event_window = event_window
        self.progress = progress or (lambda _message: None)
        self.tracker = OnlineFaceTracker(
            self.fps,
            max_gap_seconds=max_gap_seconds,
            reid_threshold=reid_threshold,
            cluster_threshold=cluster_threshold,
            min_observations=min_track_observations,
            max_people=max_people,
        )

        self._condition = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._video_finished = False
        self._stop = False
        self._paused = False
        self._status = "ready"
        self._error: Optional[str] = None
        self._current_ms = 0
        self._current_scene_id: Optional[str] = None
        self._current_scene_start = 0
        self._last_keyframe_ms = -self.keyframe_interval_ms
        self._previous_histogram: Optional[np.ndarray] = None
        self._events = deque(maxlen=event_window)
        self._event_count = 0
        self._nodes: Dict[str, dict] = {}
        self._edges: Dict[str, dict] = {}
        self._presence_edges: Dict[Tuple[str, str], str] = {}
        self._keyframes: List[dict] = []
        self._keyframe_paths: Dict[str, Path] = {}
        self._semantic_objects: Dict[Tuple[str, str], str] = {}
        self._semantic_interactions: Dict[Tuple[str, str, str, str], str] = {}
        self._revision = 0
        self._sampled_frames = 0
        self._detections = 0
        self._latencies = deque(maxlen=1000)
        self._latency_total = 0.0
        self._latency_max = 0.0
        self._lag_max = 0.0
        self._rss_start = _rss_mb()
        self._rss_warm: Optional[float] = None
        self._rss_peak = self._rss_start
        self._memory_samples = deque(maxlen=120)
        self._last_memory_sample_ms = -30_000
        self._started_at: Optional[str] = None
        self._finished_at: Optional[str] = None
        self._wall_started = 0.0
        self._paused_total = 0.0
        self._pause_started = 0.0
        self._semantic_status = "disabled"
        self._semantic_analyzed = 0
        self._semantic_failed = 0
        self._semantic_dropped = 0
        self._semantic_cache_hits = 0
        self._semantic_last_error: Optional[str] = None
        self._semantic_model = semantic_model
        self._semantic_worker: Optional[LiveSemanticWorker] = None
        if semantic_enabled or semantic_analyzer is not None:
            analyzer = semantic_analyzer or GroqSceneAnalyzer(
                model=semantic_model,
                timeout=semantic_timeout,
                proxy=semantic_proxy,
                minimum_interval=semantic_minimum_interval,
            )
            self._semantic_model = analyzer.model
            self._semantic_worker = LiveSemanticWorker(
                analyzer,
                self.output_dir / "semantic_cache",
                on_result=self._apply_semantic_result,
                on_error=self._record_semantic_error,
                on_drop=self._record_semantic_drop,
                on_finished=self._finish_semantics,
                queue_size=semantic_queue_size,
            )
            self._semantic_status = "ready"
        self._parameters = {
            "sample_fps": round(self.sample_fps, 6),
            "speed": self.speed,
            "detection_threshold": detection_threshold,
            "min_face_size": min_face_size,
            "max_gap_seconds": max_gap_seconds,
            "reid_threshold": reid_threshold,
            "cluster_threshold": cluster_threshold,
            "min_track_observations": min_track_observations,
            "max_people": max_people,
            "scene_threshold": scene_threshold,
            "min_scene_seconds": min_scene_seconds,
            "keyframe_interval_seconds": keyframe_interval_seconds,
            "event_window": event_window,
            "semantic_enabled": self._semantic_worker is not None,
            "semantic_model": self._semantic_model if self._semantic_worker else None,
            "semantic_queue_size": semantic_queue_size,
        }

    @property
    def finished(self) -> bool:
        with self._condition:
            return self._status in {"complete", "error"}

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._status = "playing"
            self._started_at = datetime.now(timezone.utc).isoformat()
            self._wall_started = time.monotonic()
            if self._semantic_worker:
                self._semantic_status = "running"
                self._semantic_worker.start()
            self._thread = threading.Thread(target=self._run, name="live-video", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._condition:
            self._stop = True
            if self._status in {"playing", "analyzing"}:
                self._status = "stopped"
            if self._semantic_status in {"ready", "running"}:
                self._semantic_status = "stopped"
            self._condition.notify_all()
            thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=5)
        if self._semantic_worker:
            self._semantic_worker.stop()

    def _wait_for_timestamp(self, timestamp_ms: int) -> bool:
        with self._condition:
            while not self._stop:
                if self._paused:
                    self._condition.wait()
                    continue
                target = self._wall_started + self._paused_total + timestamp_ms / 1000 / self.speed
                remaining = target - time.monotonic()
                if remaining <= 0:
                    return True
                self._condition.wait(timeout=min(remaining, 0.25))
            return False

    def _emit(
        self,
        event_type: str,
        timestamp_ms: int,
        subject_ref: str,
        object_ref: Optional[str],
        evidence_refs: Sequence[str],
        reveal_nodes: Sequence[str],
        reveal_edges: Sequence[str],
        payload: Optional[dict],
        event_file,
    ) -> None:
        self._event_count += 1
        event = {
            "schema_version": 1,
            "event_id": f"vision-event-{self._event_count:06d}",
            "sequence": self._event_count,
            "type": event_type,
            "timestamp_ms": int(timestamp_ms),
            "subject_ref": subject_ref,
            "object_ref": object_ref,
            "evidence_refs": list(dict.fromkeys(evidence_refs)),
            "payload": {
                **(payload or {}),
                "reveal_node_ids": list(dict.fromkeys(reveal_nodes)),
                "reveal_edge_ids": list(dict.fromkeys(reveal_edges)),
            },
        }
        self._events.append(event)
        event_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        event_file.flush()

    def _append_edge(
        self,
        source: str,
        target: str,
        predicate: str,
        timestamp_ms: int,
        confidence: float,
        evidence_refs: Sequence[str],
        epistemic_status: str = "observed",
    ) -> dict:
        edge_id = f"edge-{len(self._edges) + 1:06d}"
        edge = {
            "id": edge_id,
            "source": source,
            "target": target,
            "predicate": predicate,
            "epistemic_status": epistemic_status,
            "confidence": round(float(confidence), 6),
            "start_ms": int(timestamp_ms),
            "end_ms": min(self.duration_ms, int(timestamp_ms + self.sample_period_ms)),
            "evidence_refs": list(dict.fromkeys(evidence_refs)),
            "attributes": {},
        }
        self._edges[edge_id] = edge
        return edge

    def _save_keyframe(
        self,
        scene_id: str,
        timestamp_ms: int,
        frame: np.ndarray,
        tracked: Sequence[Tuple[Detection, OnlineTrack]],
    ) -> Tuple[str, Path]:
        image = frame.copy()
        for detection, track in tracked:
            label = track.face_id or track.track_id
            color_index = int((track.face_id or "0").split("-")[-1]) if track.face_id else 0
            color = (76, 185, 255) if not color_index else (98, 214, 115)
            _draw_box(image, detection.box, label, color)
        keyframe_id = f"live-keyframe-{len(self._keyframes) + 1:05d}"
        path = self.keyframe_dir / f"{keyframe_id}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        self._keyframes.append(
            {
                "id": keyframe_id,
                "shot_id": scene_id,
                "timestamp_ms": timestamp_ms,
                "url": "/api/keyframe?id=" + keyframe_id,
            }
        )
        self._keyframe_paths[keyframe_id] = path
        return keyframe_id, path

    def _start_scene(
        self,
        timestamp_ms: int,
        frame: np.ndarray,
        tracked: Sequence[Tuple[Detection, OnlineTrack]],
        event_file,
    ) -> None:
        previous_scene = self._current_scene_id
        if previous_scene:
            previous = self._nodes[previous_scene]
            previous["attributes"]["end_ms"] = timestamp_ms
            previous["attributes"]["duration_ms"] = (
                timestamp_ms - previous["attributes"]["start_ms"]
            )
        scene_index = sum(node["type"] == "scene" for node in self._nodes.values()) + 1
        scene_id = f"live-scene:{scene_index:04d}"
        keyframe_id, _ = self._save_keyframe(scene_id, timestamp_ms, frame, tracked)
        self._nodes[scene_id] = {
            "id": scene_id,
            "type": "scene",
            "label": f"实时场景 {scene_index}",
            "epistemic_status": "observed",
            "confidence": 1.0,
            "evidence_refs": [keyframe_id],
            "attributes": {
                "environment_category": "unknown",
                "descriptions": [],
                "start_ms": timestamp_ms,
                "end_ms": min(self.duration_ms, timestamp_ms + self.sample_period_ms),
                "duration_ms": min(self.sample_period_ms, self.duration_ms - timestamp_ms),
                "shot_ids": [scene_id],
                "keyframe_ids": [keyframe_id],
            },
        }
        if previous_scene:
            edge = self._append_edge(
                previous_scene,
                scene_id,
                "followed_by",
                timestamp_ms,
                1.0,
                [self._nodes[previous_scene]["evidence_refs"][0], keyframe_id],
            )
            edge["end_ms"] = timestamp_ms
            self._emit(
                "relation_observed",
                timestamp_ms,
                previous_scene,
                scene_id,
                edge["evidence_refs"],
                [previous_scene, scene_id],
                [edge["id"]],
                {
                    "predicate": "followed_by",
                    "confidence": 1.0,
                    "epistemic_status": "observed",
                    "end_ms": timestamp_ms,
                },
                event_file,
            )
        self._current_scene_id = scene_id
        self._current_scene_start = timestamp_ms
        self._last_keyframe_ms = timestamp_ms
        self._emit(
            "scene_changed",
            timestamp_ms,
            scene_id,
            None,
            [keyframe_id],
            [scene_id],
            [],
            {"label": f"实时场景 {scene_index}", "confidence": 1.0},
            event_file,
        )
        self._enqueue_semantics(keyframe_id, scene_id, timestamp_ms, tracked)

    def _sample_scene(
        self,
        timestamp_ms: int,
        frame: np.ndarray,
        tracked: Sequence[Tuple[Detection, OnlineTrack]],
        event_file,
    ) -> None:
        assert self._current_scene_id is not None
        keyframe_id, _ = self._save_keyframe(
            self._current_scene_id,
            timestamp_ms,
            frame,
            tracked,
        )
        scene = self._nodes[self._current_scene_id]
        scene["evidence_refs"].append(keyframe_id)
        scene["attributes"]["keyframe_ids"].append(keyframe_id)
        self._last_keyframe_ms = timestamp_ms
        self._emit(
            "scene_sampled",
            timestamp_ms,
            self._current_scene_id,
            None,
            [keyframe_id],
            [self._current_scene_id],
            [],
            {"label": scene["label"], "confidence": 1.0},
            event_file,
        )
        self._enqueue_semantics(
            keyframe_id,
            self._current_scene_id,
            timestamp_ms,
            tracked,
        )

    def _enqueue_semantics(
        self,
        keyframe_id: str,
        scene_id: str,
        timestamp_ms: int,
        tracked: Sequence[Tuple[Detection, OnlineTrack]],
    ) -> None:
        if self._semantic_worker is None or self._semantic_status != "running":
            return
        visible_people = []
        for detection, track in tracked:
            if track.face_id is None:
                continue
            x, y, width, height = detection.box
            center_x = (x + width / 2) / self.width
            center_y = (y + height / 2) / self.height
            visible_people.append(
                {
                    "face_id": track.face_id,
                    "speaker_label": None,
                    "identity_status": "visual_only",
                    "horizontal_region": (
                        "left" if center_x < 1 / 3 else "right" if center_x > 2 / 3 else "center"
                    ),
                    "vertical_region": (
                        "top" if center_y < 1 / 3 else "bottom" if center_y > 2 / 3 else "middle"
                    ),
                }
            )
        self._semantic_worker.submit(
            SemanticTask(
                keyframe_id=keyframe_id,
                scene_id=scene_id,
                timestamp_ms=timestamp_ms,
                metadata={
                    "keyframe_id": keyframe_id,
                    "shot_id": scene_id,
                    "timestamp_ms": timestamp_ms,
                    "shot_start_ms": self._nodes[scene_id]["attributes"]["start_ms"],
                    "shot_end_ms": min(self.duration_ms, timestamp_ms + self.sample_period_ms),
                    "visible_people": visible_people,
                    "spatial_relations": [],
                    "transcript": [],
                },
                image_path=self._keyframe_paths[keyframe_id],
            )
        )

    def _upsert_semantic_object(
        self,
        scene_id: str,
        value: dict,
        timestamp_ms: int,
        evidence_ref: str,
    ) -> Tuple[str, str]:
        label = value["label"]
        key = (scene_id, label)
        node_id = self._semantic_objects.get(key)
        if node_id is None:
            node_id = f"{scene_id}:object:{len(self._semantic_objects) + 1:03d}"
            self._semantic_objects[key] = node_id
            self._nodes[node_id] = {
                "id": node_id,
                "type": "object",
                "label": label,
                "epistemic_status": "inferred",
                "confidence": round(float(value["confidence"]), 6),
                "evidence_refs": [evidence_ref],
                "attributes": {
                    "scene_id": scene_id,
                    "count": int(value.get("count", 1)),
                    "regions": [value.get("region", "unknown")],
                },
            }
            edge = self._append_edge(
                scene_id,
                node_id,
                "contains",
                timestamp_ms,
                value["confidence"],
                [evidence_ref],
                "inferred",
            )
            return node_id, edge["id"]
        node = self._nodes[node_id]
        node["confidence"] = max(node["confidence"], round(float(value["confidence"]), 6))
        node["attributes"]["count"] = max(
            node["attributes"]["count"], int(value.get("count", 1))
        )
        region = value.get("region", "unknown")
        if region not in node["attributes"]["regions"]:
            node["attributes"]["regions"].append(region)
        if evidence_ref not in node["evidence_refs"]:
            node["evidence_refs"].append(evidence_ref)
        edge = next(
            edge
            for edge in self._edges.values()
            if edge["source"] == scene_id
            and edge["target"] == node_id
            and edge["predicate"] == "contains"
        )
        edge["end_ms"] = min(
            self.duration_ms,
            max(edge["end_ms"], timestamp_ms + self.sample_period_ms),
        )
        edge["confidence"] = max(edge["confidence"], node["confidence"])
        if evidence_ref not in edge["evidence_refs"]:
            edge["evidence_refs"].append(evidence_ref)
        return node_id, edge["id"]

    def _apply_semantic_result(
        self,
        task: SemanticTask,
        raw: dict,
        request_hash: str,
        cache_hit: bool,
    ) -> None:
        visible_faces = [item["face_id"] for item in task.metadata["visible_people"]]
        analysis = normalize_live_analysis(raw, visible_faces)
        with self._condition, self.event_path.open("a", encoding="utf-8") as event_file:
            if self._stop or task.scene_id not in self._nodes:
                return
            self._semantic_analyzed += 1
            self._semantic_cache_hits += int(cache_hit)
            analysis_id = f"live-semantic-{self._semantic_analyzed:06d}"
            evidence_ref = analysis_id
            scene = self._nodes[task.scene_id]
            environment = analysis["environment"]
            description = environment["description"].strip()
            scene["attributes"]["environment_category"] = environment["category"]
            if description and description not in scene["attributes"]["descriptions"]:
                scene["attributes"]["descriptions"].append(description)
            scene["attributes"]["semantic_status"] = "analyzed"
            scene["attributes"].setdefault("semantic_keyframe_ids", []).append(
                task.keyframe_id
            )
            if description:
                scene["label"] = description
            if evidence_ref not in scene["evidence_refs"]:
                scene["evidence_refs"].append(evidence_ref)
            self._emit(
                "node_available",
                task.timestamp_ms,
                task.scene_id,
                None,
                [task.keyframe_id, evidence_ref],
                [task.scene_id],
                [],
                {
                    "label": scene["label"],
                    "environment_category": environment["category"],
                    "confidence": environment["confidence"],
                    "epistemic_status": "inferred",
                },
                event_file,
            )
            for value in analysis["objects"]:
                node_id, edge_id = self._upsert_semantic_object(
                    task.scene_id,
                    value,
                    task.timestamp_ms,
                    evidence_ref,
                )
                self._emit(
                    "object_detected",
                    task.timestamp_ms,
                    task.scene_id,
                    node_id,
                    [task.keyframe_id, evidence_ref],
                    [task.scene_id, node_id],
                    [edge_id],
                    {
                        "predicate": "contains",
                        "confidence": value["confidence"],
                        "epistemic_status": "inferred",
                    },
                    event_file,
                )
            for value in analysis["interactions"]:
                source = "visual_identity:" + value["subject_ref"]
                if source not in self._nodes:
                    continue
                object_ref = value.get("object_ref")
                if object_ref and object_ref.startswith("Face-"):
                    target = "visual_identity:" + object_ref
                elif object_ref and object_ref.startswith("object:"):
                    target, _ = self._upsert_semantic_object(
                        task.scene_id,
                        {
                            "label": object_ref.split(":", 1)[1],
                            "count": 1,
                            "region": "unknown",
                            "confidence": value["confidence"],
                        },
                        task.timestamp_ms,
                        evidence_ref,
                    )
                else:
                    target = task.scene_id
                if target not in self._nodes:
                    continue
                key = (task.scene_id, source, value["predicate"], target)
                edge_id = self._semantic_interactions.get(key)
                if edge_id is None:
                    edge = self._append_edge(
                        source,
                        target,
                        value["predicate"],
                        task.timestamp_ms,
                        value["confidence"],
                        [evidence_ref],
                        "inferred",
                    )
                    edge["attributes"] = {
                        "object_scope": object_ref,
                        "descriptions": [value["description"]],
                        "evidence_basis": sorted(set(value["evidence_basis"])),
                    }
                    edge_id = edge["id"]
                    self._semantic_interactions[key] = edge_id
                else:
                    edge = self._edges[edge_id]
                    edge["end_ms"] = min(
                        self.duration_ms,
                        max(edge["end_ms"], task.timestamp_ms + self.sample_period_ms),
                    )
                    edge["confidence"] = max(
                        edge["confidence"], round(float(value["confidence"]), 6)
                    )
                    if evidence_ref not in edge["evidence_refs"]:
                        edge["evidence_refs"].append(evidence_ref)
                    if value["description"] not in edge["attributes"]["descriptions"]:
                        edge["attributes"]["descriptions"].append(value["description"])
                    edge["attributes"]["evidence_basis"] = sorted(
                        set(edge["attributes"]["evidence_basis"] + value["evidence_basis"])
                    )
                self._emit(
                    "interaction_detected",
                    task.timestamp_ms,
                    source,
                    target,
                    [task.keyframe_id, evidence_ref],
                    [source, target],
                    [edge_id],
                    {
                        "predicate": value["predicate"],
                        "confidence": value["confidence"],
                        "epistemic_status": "inferred",
                    },
                    event_file,
                )
            record = {
                "schema_version": 1,
                "analysis_id": analysis_id,
                "scene_id": task.scene_id,
                "timestamp_ms": task.timestamp_ms,
                "request_sha256": request_hash,
                "cache_hit": cache_hit,
                **analysis,
            }
            with self.semantic_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._revision += 1
            self.progress(
                f"Qwen completed {task.keyframe_id}: "
                f"objects={len(analysis['objects'])}, "
                f"interactions={len(analysis['interactions'])}"
            )

    def _record_semantic_error(self, task: SemanticTask, error: Exception) -> None:
        with self._condition:
            self._semantic_failed += 1
            self._semantic_status = "degraded"
            self._semantic_last_error = str(error)
            self._revision += 1
        self.progress(f"Qwen failed {task.keyframe_id}: {error}")

    def _record_semantic_drop(self, task: SemanticTask) -> None:
        with self._condition:
            self._semantic_dropped += 1
            self._revision += 1
        self.progress(f"Qwen queue skipped {task.keyframe_id}")

    def _finish_semantics(self) -> None:
        with self._condition:
            if self._semantic_status not in {"degraded", "stopped"}:
                self._semantic_status = "complete"
            if self._video_finished and self._status not in {"error", "stopped"}:
                self._status = "complete"
                self._finished_at = datetime.now(timezone.utc).isoformat()
            self._revision += 1
        self._write_outputs()

    def _update_faces(
        self,
        timestamp_ms: int,
        tracked: Sequence[Tuple[Detection, OnlineTrack]],
        event_file,
    ) -> None:
        assert self._current_scene_id is not None
        for detection, track in tracked:
            face_id = track.face_id
            if face_id is None:
                continue
            node_id = "visual_identity:" + face_id
            x, y, width, height = detection.box
            normalized_box = [
                round(x / self.width, 6),
                round(y / self.height, 6),
                round(width / self.width, 6),
                round(height / self.height, 6),
            ]
            if node_id not in self._nodes:
                self._nodes[node_id] = {
                    "id": node_id,
                    "type": "visual_identity",
                    "label": face_id,
                    "epistemic_status": "observed",
                    "confidence": round(detection.score, 6),
                    "evidence_refs": [track.track_id],
                    "attributes": {
                        "face_id": face_id,
                        "speaker_label": None,
                        "participant_id": None,
                        "display_name": None,
                        "identity_status": "visual_only",
                        "latest_bbox_normalized": normalized_box,
                        "last_seen_ms": timestamp_ms,
                    },
                }
            else:
                node = self._nodes[node_id]
                node["confidence"] = max(node["confidence"], round(detection.score, 6))
                node["attributes"]["latest_bbox_normalized"] = normalized_box
                node["attributes"]["last_seen_ms"] = timestamp_ms
                if track.track_id not in node["evidence_refs"]:
                    node["evidence_refs"].append(track.track_id)
            key = (node_id, self._current_scene_id)
            edge_id = self._presence_edges.get(key)
            if edge_id:
                edge = self._edges[edge_id]
                edge["end_ms"] = min(self.duration_ms, timestamp_ms + self.sample_period_ms)
                edge["confidence"] = max(edge["confidence"], round(detection.score, 6))
                continue
            edge = self._append_edge(
                node_id,
                self._current_scene_id,
                "present_in",
                timestamp_ms,
                detection.score,
                [track.track_id],
            )
            self._presence_edges[key] = edge["id"]
            self._emit(
                "person_seen",
                timestamp_ms,
                node_id,
                self._current_scene_id,
                [track.track_id],
                [node_id, self._current_scene_id],
                [edge["id"]],
                {
                    "predicate": "present_in",
                    "confidence": edge["confidence"],
                    "epistemic_status": "observed",
                    "end_ms": edge["end_ms"],
                },
                event_file,
            )

    def _update_metrics(self, latency_ms: float, lag_ms: float) -> None:
        self._sampled_frames += 1
        self._latencies.append(latency_ms)
        self._latency_total += latency_ms
        self._latency_max = max(self._latency_max, latency_ms)
        self._lag_max = max(self._lag_max, lag_ms)
        current_rss = _rss_mb()
        if self._rss_warm is None and self._current_ms >= 30_000:
            self._rss_warm = current_rss
        self._rss_peak = max(self._rss_peak, current_rss)
        if self._current_ms - self._last_memory_sample_ms >= 30_000:
            self._memory_samples.append(
                {"timestamp_ms": self._current_ms, "rss_mb": round(current_rss, 3)}
            )
            self._last_memory_sample_ms = self._current_ms

    def _close_scene(self, end_ms: int) -> None:
        if self._current_scene_id:
            scene = self._nodes[self._current_scene_id]
            scene["attributes"]["end_ms"] = end_ms
            scene["attributes"]["duration_ms"] = end_ms - scene["attributes"]["start_ms"]

    def _run(self) -> None:
        capture = cv2.VideoCapture(str(self.video_path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"cannot open video: {self.video_path}")
            detector = cv2.FaceDetectorYN.create(
                str(model_path("yunet")),
                "",
                (self.width, self.height),
                self.detection_threshold,
                0.3,
                5000,
            )
            recognizer = cv2.FaceRecognizerSF.create(str(model_path("sface")), "")
            current_frame = 0
            with self.event_path.open("a", encoding="utf-8") as event_file:
                for frame_index in range(0, self.frame_count, self.sample_step):
                    timestamp_ms = round(frame_index * 1000 / self.fps)
                    if not self._wait_for_timestamp(timestamp_ms):
                        break
                    while current_frame < frame_index:
                        if not capture.grab():
                            raise RuntimeError(
                                f"video decoding stopped before frame {frame_index}"
                            )
                        current_frame += 1
                    ok, frame = capture.read()
                    current_frame += 1
                    if not ok:
                        raise RuntimeError(f"video decoding failed at frame {frame_index}")
                    started = time.monotonic()
                    detections = _detect(detector, recognizer, frame, self.min_face_size)
                    tracked = self.tracker.update(detections, frame_index)
                    histogram = _frame_histogram(frame)
                    with self._condition:
                        if self._stop:
                            break
                        scene_change = self._current_scene_id is None
                        if self._previous_histogram is not None:
                            distance = cv2.compareHist(
                                self._previous_histogram,
                                histogram,
                                cv2.HISTCMP_BHATTACHARYYA,
                            )
                            scene_change = scene_change or (
                                timestamp_ms - self._current_scene_start >= self.min_scene_ms
                                and distance >= self.scene_threshold
                            )
                        if scene_change:
                            self._start_scene(timestamp_ms, frame, tracked, event_file)
                        elif timestamp_ms - self._last_keyframe_ms >= self.keyframe_interval_ms:
                            self._sample_scene(timestamp_ms, frame, tracked, event_file)
                        self._previous_histogram = histogram
                        self._current_ms = timestamp_ms
                        self._detections += len(detections)
                        self._update_faces(timestamp_ms, tracked, event_file)
                        if self._current_scene_id:
                            scene = self._nodes[self._current_scene_id]
                            scene["attributes"]["end_ms"] = min(
                                self.duration_ms, timestamp_ms + self.sample_period_ms
                            )
                            scene["attributes"]["duration_ms"] = (
                                scene["attributes"]["end_ms"] - scene["attributes"]["start_ms"]
                            )
                        latency_ms = (time.monotonic() - started) * 1000
                        target = (
                            self._wall_started
                            + self._paused_total
                            + timestamp_ms / 1000 / self.speed
                        )
                        lag_ms = max(0.0, (time.monotonic() - target) * 1000)
                        self._update_metrics(latency_ms, lag_ms)
                        self._revision += 1
                        progress_interval = max(1, round(self.sample_fps * 5))
                        if self._sampled_frames % progress_interval == 0:
                            scene_count = sum(
                                node["type"] == "scene" for node in self._nodes.values()
                            )
                            self.progress(
                                f"{timestamp_ms / 1000:.1f}/{self.duration_ms / 1000:.1f}s, "
                                f"faces={len(self.tracker.face_profiles)}, "
                                f"scenes={scene_count}, "
                                f"latency={latency_ms:.1f}ms"
                            )
                with self._condition:
                    if not self._stop:
                        self._current_ms = self.duration_ms
                        self._close_scene(self.duration_ms)
                        self._video_finished = True
                        if self._semantic_worker and not self._semantic_worker.finished:
                            self._status = "analyzing"
                        else:
                            self._status = "complete"
                            self._finished_at = datetime.now(timezone.utc).isoformat()
                    self._revision += 1
                if self._semantic_worker and not self._stop:
                    self._semantic_worker.close()
        except Exception as error:
            with self._condition:
                self._status = "error"
                self._error = str(error)
                self._finished_at = datetime.now(timezone.utc).isoformat()
                self._revision += 1
        finally:
            capture.release()
            self._write_outputs()

    def _statistics(self) -> dict:
        edges = list(self._edges.values())
        nodes = list(self._nodes.values())
        return {
            "nodes": len(nodes),
            "edges": len(edges),
            "persons": sum(node["type"] == "person" for node in nodes),
            "visual_identities": sum(node["type"] == "visual_identity" for node in nodes),
            "scenes": sum(node["type"] == "scene" for node in nodes),
            "objects": sum(node["type"] == "object" for node in nodes),
            "interactions": sum(
                edge["predicate"]
                not in {"present_in", "contains", "followed_by", "identity_candidate"}
                for edge in edges
            ),
            "observed_edges": sum(edge["epistemic_status"] == "observed" for edge in edges),
            "inferred_edges": sum(edge["epistemic_status"] == "inferred" for edge in edges),
        }

    def _graph(self) -> dict:
        return {
            "schema_version": 1,
            "context_type": "live_visual_graph",
            "session_id": self.output_dir.name,
            "created_at": self._started_at or datetime.now(timezone.utc).isoformat(),
            "source": {
                "video_path": str(self.video_path),
                "video_sha256": self.video_sha256,
                "duration_ms": self.duration_ms,
                "fps": round(self.fps, 6),
                "frame_count": self.frame_count,
                "width": self.width,
                "height": self.height,
            },
            "processing": {
                "mode": "live_frame_stream_v2",
                "inference_device": "cpu",
                "parameters": self._parameters,
                "scene_semantics": self._semantic_metrics(),
            },
            "nodes": list(self._nodes.values()),
            "edges": list(self._edges.values()),
            "statistics": self._statistics(),
            "warnings": ([self._error] if self._error else [])
            + (
                ["Live scene semantics degraded: " + self._semantic_last_error]
                if self._semantic_last_error
                else []
            )
            + (
                [
                    f"Dropped {self.tracker.dropped_identity_candidates} unassigned "
                    "identity candidates after reaching max_people."
                ]
                if self.tracker.dropped_identity_candidates
                else []
            ),
        }

    def _semantic_metrics(self) -> dict:
        pending = self._semantic_worker.pending if self._semantic_worker else 0
        return {
            "enabled": self._semantic_worker is not None,
            "provider": "groq" if self._semantic_worker else None,
            "model": self._semantic_model if self._semantic_worker else None,
            "status": self._semantic_status,
            "pending": pending,
            "analyzed": self._semantic_analyzed,
            "failed": self._semantic_failed,
            "dropped": self._semantic_dropped,
            "cache_hits": self._semantic_cache_hits,
            "last_error": self._semantic_last_error,
        }

    def _metrics(self) -> dict:
        latencies = list(self._latencies)
        count = self._sampled_frames
        current_rss = _rss_mb()
        peak_rss = max(self._rss_peak, current_rss)
        return {
            "schema_version": 1,
            "session_id": self.output_dir.name,
            "status": self._status,
            "started_at": self._started_at,
            "finished_at": self._finished_at,
            "current_ms": self._current_ms,
            "duration_ms": self.duration_ms,
            "sampled_frames": count,
            "detections": self._detections,
            "face_ids": len(self.tracker.face_profiles),
            "active_tracks": len(self.tracker.tracks),
            "scenes": self._statistics()["scenes"],
            "events": self._event_count,
            "event_buffer_size": len(self._events),
            "event_buffer_limit": self.event_window,
            "latency_ms": {
                "mean": round(self._latency_total / count, 3) if count else 0.0,
                "p50_recent": round(_percentile(latencies, 0.50), 3),
                "p95_recent": round(_percentile(latencies, 0.95), 3),
                "max": round(self._latency_max, 3),
            },
            "schedule_lag_max_ms": round(self._lag_max, 3),
            "memory_mb": {
                "rss_start": round(self._rss_start, 3),
                "rss_after_30s": None if self._rss_warm is None else round(self._rss_warm, 3),
                "rss_current": round(current_rss, 3),
                "rss_peak": round(peak_rss, 3),
                "growth_after_30s": None
                if self._rss_warm is None
                else round(current_rss - self._rss_warm, 3),
                "process_gpu": 0.0,
                "samples": list(self._memory_samples),
            },
            "inference_device": "cpu",
            "scene_semantics": self._semantic_metrics(),
            "error": self._error,
        }

    def graph_payload(self) -> Tuple[dict, dict, Dict[str, Path], int]:
        with self._condition:
            return (
                copy.deepcopy(self._graph()),
                {"keyframes": copy.deepcopy(self._keyframes)},
                dict(self._keyframe_paths),
                self._revision,
            )

    def snapshot(self) -> dict:
        with self._condition:
            status = "paused" if self._paused and self._status == "playing" else self._status
            return {
                "schema_version": 1,
                "mode": "live",
                "status": status,
                "current_ms": self._current_ms,
                "duration_ms": self.duration_ms,
                "speed": self.speed,
                "events_emitted": self._event_count,
                "events_total": self._event_count,
                "event_window": self.event_window,
                "current_scene_id": self._current_scene_id,
                "visible_node_ids": list(self._nodes),
                "visible_edge_ids": list(self._edges),
                "events": list(self._events),
                "metrics": self._metrics(),
                "scene_semantics": self._semantic_metrics(),
                "error": self._error,
            }

    def control(self, action: str, timestamp_ms: Optional[int] = None) -> dict:
        del timestamp_ms
        with self._condition:
            if action == "pause" and self._status == "playing" and not self._paused:
                self._paused = True
                self._pause_started = time.monotonic()
            elif action == "resume" and self._status == "playing" and self._paused:
                self._paused = False
                self._paused_total += time.monotonic() - self._pause_started
                self._condition.notify_all()
            elif action not in {"pause", "resume"}:
                raise ValueError("live video supports only pause and resume")
        return self.snapshot()

    def _write_outputs(self) -> None:
        with self._condition:
            graph = copy.deepcopy(self._graph())
            metrics = copy.deepcopy(self._metrics())
        for path, payload in (
            (self.output_dir / "live_visual_graph.json", graph),
            (self.output_dir / "live_metrics.json", metrics),
        ):
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)


def serve_live_video(
    video_path: Path,
    output_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    exit_on_complete: bool = False,
    **session_options,
) -> int:
    from .scene_graph_server import create_server

    session = LiveVideoSession(video_path, output_dir, **session_options)
    server, source, url = create_server(
        session.video_path,
        host,
        port,
        replay=session,
        live=session,
    )
    session.start()
    print("Streaming " + str(source), flush=True)
    print(url, flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        if exit_on_complete:
            server.timeout = 0.25
            while not session.finished:
                server.handle_request()
        else:
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
        server.server_close()
    metrics = session.snapshot()["metrics"]
    print(f"[done] {session.output_dir / 'live_visual_graph.json'}", flush=True)
    print(f"[done] {session.output_dir / 'live_metrics.json'}", flush=True)
    print(
        f"[live] status={metrics['status']}, frames={metrics['sampled_frames']}, "
        f"faces={metrics['face_ids']}, scenes={metrics['scenes']}, "
        f"p95={metrics['latency_ms']['p95_recent']:.1f}ms, "
        f"rss-peak={metrics['memory_mb']['rss_peak']:.1f}MB",
        flush=True,
    )
    return 0 if metrics["status"] == "complete" else 1
