from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .models import file_sha256, manifest, model_path, require_models


Color = Tuple[int, int, int]
COLORS: Sequence[Color] = (
    (76, 185, 255),
    (98, 214, 115),
    (255, 142, 86),
    (214, 112, 218),
    (79, 220, 220),
    (235, 191, 84),
)
MIN_SPATIAL_MATCH_SIMILARITY = 0.25


def _normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(_normalized(left), _normalized(right)))


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def _clamp_box(box: Sequence[float], width: int, height: int) -> List[int]:
    x, y, w, h = box
    x1 = max(0, min(width - 1, int(round(x))))
    y1 = max(0, min(height - 1, int(round(y))))
    x2 = max(x1 + 1, min(width, int(round(x + w))))
    y2 = max(y1 + 1, min(height, int(round(y + h))))
    return [x1, y1, x2 - x1, y2 - y1]


@dataclass
class Detection:
    box: List[int]
    landmarks: List[List[float]]
    score: float
    quality: float
    embedding: np.ndarray


@dataclass
class Tracklet:
    tracklet_id: str
    start_ms: int
    end_ms: int
    last_frame: int
    last_box: List[int]
    embedding_sum: np.ndarray
    embedding_weight: float
    observation_ids: List[str] = field(default_factory=list)
    detection_scores: List[float] = field(default_factory=list)
    representative_observation_id: str = ""
    representative_quality: float = -1.0
    face_id: Optional[str] = None

    @property
    def embedding(self) -> np.ndarray:
        return _normalized(self.embedding_sum)

    def add(self, detection: Detection, observation_id: str, frame_index: int, timestamp_ms: int) -> None:
        weight = max(0.05, detection.quality)
        self.embedding_sum += detection.embedding * weight
        self.embedding_weight += weight
        self.observation_ids.append(observation_id)
        self.detection_scores.append(detection.score)
        self.last_box = detection.box
        self.last_frame = frame_index
        self.end_ms = timestamp_ms
        if detection.quality > self.representative_quality:
            self.representative_quality = detection.quality
            self.representative_observation_id = observation_id


class TrackManager:
    def __init__(self, fps: float, max_gap_seconds: float, reid_threshold: float):
        self.fps = fps
        self.max_gap_frames = max(1, int(round(max_gap_seconds * fps)))
        self.reid_threshold = reid_threshold
        self.tracklets: List[Tracklet] = []
        self.observations: List[dict] = []

    def _active_tracklets(self, frame_index: int) -> List[Tracklet]:
        return [track for track in self.tracklets if frame_index - track.last_frame <= self.max_gap_frames]

    def _new_tracklet(self, detection: Detection, frame_index: int, timestamp_ms: int) -> Tracklet:
        tracklet = Tracklet(
            tracklet_id=f"tracklet-{len(self.tracklets) + 1:04d}",
            start_ms=timestamp_ms,
            end_ms=timestamp_ms,
            last_frame=frame_index,
            last_box=detection.box,
            embedding_sum=np.zeros_like(detection.embedding),
            embedding_weight=0.0,
        )
        self.tracklets.append(tracklet)
        return tracklet

    def update(
        self,
        detections: Sequence[Detection],
        frame_index: int,
        timestamp_ms: int,
        width: int,
        height: int,
    ) -> None:
        active = self._active_tracklets(frame_index)
        matches: Dict[int, Tuple[int, float]] = {}
        if active and detections:
            costs = np.full((len(active), len(detections)), 1e6, dtype=np.float32)
            confidences = np.zeros_like(costs)
            for row, track in enumerate(active):
                gap_ratio = (frame_index - track.last_frame) / self.max_gap_frames
                for column, detection in enumerate(detections):
                    similarity = cosine_similarity(track.embedding, detection.embedding)
                    overlap = box_iou(track.last_box, detection.box)
                    # A box can stay in the same place across a camera cut while the
                    # person changes. Require some facial agreement even when IoU is
                    # high, then use the stricter re-identification threshold when
                    # the boxes do not overlap.
                    allowed = similarity >= MIN_SPATIAL_MATCH_SIMILARITY and (
                        overlap >= 0.20 or similarity >= self.reid_threshold
                    )
                    if not allowed:
                        continue
                    costs[row, column] = 0.65 * (1.0 - similarity) + 0.30 * (1.0 - overlap) + 0.05 * gap_ratio
                    confidences[row, column] = max(0.0, min(1.0, 0.7 * similarity + 0.3 * overlap))
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows.tolist(), columns.tolist()):
                if costs[row, column] < 1e5:
                    matches[column] = (self.tracklets.index(active[row]), float(confidences[row, column]))

        for index, detection in enumerate(detections):
            observation_id = f"observation-{len(self.observations) + 1:06d}"
            if index in matches:
                tracklet = self.tracklets[matches[index][0]]
                match_confidence: Optional[float] = matches[index][1]
            else:
                tracklet = self._new_tracklet(detection, frame_index, timestamp_ms)
                match_confidence = None
            tracklet.add(detection, observation_id, frame_index, timestamp_ms)
            x, y, w, h = detection.box
            self.observations.append(
                {
                    "observation_id": observation_id,
                    "tracklet_id": tracklet.tracklet_id,
                    "face_id": None,
                    "frame_index": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "bbox_px": detection.box,
                    "bbox_normalized": [
                        round(x / width, 6),
                        round(y / height, 6),
                        round(w / width, 6),
                        round(h / height, 6),
                    ],
                    "landmarks_px": [[round(point[0], 2), round(point[1], 2)] for point in detection.landmarks],
                    "detection_confidence": round(detection.score, 6),
                    "track_match_confidence": None if match_confidence is None else round(match_confidence, 6),
                    "quality": round(detection.quality, 6),
                }
            )


def _temporal_overlap(left: Tracklet, right: Tracklet, tolerance_ms: int) -> bool:
    overlap = min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms)
    return overlap > tolerance_ms


def cluster_tracklets(
    tracklets: Sequence[Tracklet],
    observations: Sequence[dict],
    min_observations: int,
    threshold: float,
    sample_period_ms: int,
) -> List[List[Tracklet]]:
    retained = [track for track in tracklets if len(track.observation_ids) >= min_observations]
    retained.sort(key=lambda item: (item.start_ms, item.tracklet_id))
    clusters: List[List[Tracklet]] = [[tracklet] for tracklet in retained]

    # Merge the strongest complete-link pair first. This lets two appearances of
    # the same person join before a weaker lookalike edge is considered. If any
    # members overlap in time, the clusters can never represent one identity.
    while True:
        best_pair: Optional[Tuple[int, int]] = None
        best_similarity = threshold
        for left_index, left_members in enumerate(clusters):
            for right_index in range(left_index + 1, len(clusters)):
                right_members = clusters[right_index]
                if any(
                    _temporal_overlap(left, right, sample_period_ms)
                    for left in left_members
                    for right in right_members
                ):
                    continue
                similarity = min(
                    cosine_similarity(left.embedding, right.embedding)
                    for left in left_members
                    for right in right_members
                )
                if similarity >= best_similarity:
                    candidate = (left_index, right_index)
                    if similarity > best_similarity or best_pair is None or candidate < best_pair:
                        best_pair = candidate
                        best_similarity = similarity
        if best_pair is None:
            break
        left_index, right_index = best_pair
        clusters[left_index].extend(clusters.pop(right_index))

    clusters.sort(key=lambda members: min(track.start_ms for track in members))
    observation_by_id = {item["observation_id"]: item for item in observations}
    for index, members in enumerate(clusters, start=1):
        face_id = f"Face-{index:02d}"
        for tracklet in members:
            tracklet.face_id = face_id
            for observation_id in tracklet.observation_ids:
                observation_by_id[observation_id]["face_id"] = face_id
    return clusters


def _union_duration(intervals: Sequence[Tuple[int, int]], sample_period_ms: int) -> int:
    if not intervals:
        return 0
    ranges = sorted((start, end + sample_period_ms) for start, end in intervals)
    total = 0
    current_start, current_end = ranges[0]
    for start, end in ranges[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _face_records(clusters: Sequence[Sequence[Tracklet]], observations: Sequence[dict], sample_period_ms: int) -> List[dict]:
    observation_by_id = {item["observation_id"]: item for item in observations}
    faces = []
    for index, members in enumerate(clusters, start=1):
        face_id = f"Face-{index:02d}"
        observation_ids = [item for track in members for item in track.observation_ids]
        representative = max(observation_ids, key=lambda item: observation_by_id[item]["quality"])
        faces.append(
            {
                "face_id": face_id,
                "first_seen_ms": min(track.start_ms for track in members),
                "last_seen_ms": max(track.end_ms for track in members),
                "visible_duration_ms": _union_duration(
                    [(track.start_ms, track.end_ms) for track in members], sample_period_ms
                ),
                "observation_count": len(observation_ids),
                "tracklet_ids": [track.tracklet_id for track in sorted(members, key=lambda item: item.start_ms)],
                "representative_observation_id": representative,
                "evidence_image": f"evidence/{face_id}.jpg",
            }
        )
    return faces


def _tracklet_records(tracklets: Sequence[Tracklet]) -> List[dict]:
    return [
        {
            "tracklet_id": track.tracklet_id,
            "face_id": track.face_id,
            "start_ms": track.start_ms,
            "end_ms": track.end_ms,
            "observation_count": len(track.observation_ids),
            "observation_ids": track.observation_ids,
            "mean_detection_confidence": round(float(np.mean(track.detection_scores)), 6),
            "representative_observation_id": track.representative_observation_id,
        }
        for track in tracklets
    ]


def _model_record(name: str) -> dict:
    data = manifest()
    spec = data["models"][name]
    return {
        "name": name,
        "filename": spec["filename"],
        "sha256": spec["sha256"],
        "repository_revision": data["revision"],
        "license": spec["license"],
    }


def _detect(detector, recognizer, frame: np.ndarray, min_face_size: int) -> List[Detection]:
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, rows = detector.detect(frame)
    if rows is None:
        return []
    detections = []
    for row in rows:
        box = _clamp_box(row[:4], width, height)
        if min(box[2], box[3]) < min_face_size:
            continue
        try:
            aligned = recognizer.alignCrop(frame, row)
            embedding = _normalized(recognizer.feature(aligned))
        except cv2.error:
            continue
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        blur = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 180.0)
        scale = min(1.0, math.sqrt(box[2] * box[3]) / 112.0)
        score = float(row[-1])
        quality = max(0.0, min(1.0, score * (0.45 + 0.30 * scale + 0.25 * blur)))
        landmarks = [[float(row[i]), float(row[i + 1])] for i in range(4, 14, 2)]
        detections.append(Detection(box, landmarks, score, quality, embedding))
    return detections


def _draw_box(frame: np.ndarray, box: Sequence[int], label: str, color: Color) -> None:
    x, y, width, height = [int(value) for value in box]
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.62, 2
    (text_width, text_height), _ = cv2.getTextSize(label, font, scale, thickness)
    top = max(0, y - text_height - 10)
    cv2.rectangle(frame, (x, top), (x + text_width + 10, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 5), font, scale, (15, 15, 15), thickness, cv2.LINE_AA)


def _write_evidence(video_path: Path, output_dir: Path, faces: Sequence[dict], observations: Sequence[dict]) -> None:
    evidence_dir = output_dir / "evidence"
    if evidence_dir.exists():
        shutil.rmtree(str(evidence_dir))
    evidence_dir.mkdir(parents=True)
    observation_by_id = {item["observation_id"]: item for item in observations}
    capture = cv2.VideoCapture(str(video_path))
    try:
        for face_index, face in enumerate(faces):
            observation = observation_by_id[face["representative_observation_id"]]
            capture.set(cv2.CAP_PROP_POS_FRAMES, observation["frame_index"])
            ok, frame = capture.read()
            if not ok:
                face["evidence_image"] = ""
                continue
            _draw_box(
                frame,
                observation["bbox_px"],
                f"{face['face_id']}  {observation['timestamp_ms'] / 1000:.1f}s",
                COLORS[face_index % len(COLORS)],
            )
            cv2.imwrite(str(output_dir / face["evidence_image"]), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    finally:
        capture.release()


def _interpolated_boxes(tracklets: Sequence[Tracklet], observations: Sequence[dict], max_gap_frames: int) -> Dict[int, List[tuple]]:
    observation_by_id = {item["observation_id"]: item for item in observations}
    boxes: Dict[int, List[tuple]] = {}
    retained = [track for track in tracklets if track.face_id]
    for track in retained:
        items = sorted((observation_by_id[item] for item in track.observation_ids), key=lambda item: item["frame_index"])
        color_index = max(0, int(track.face_id.split("-")[-1]) - 1)
        for item in items:
            boxes.setdefault(item["frame_index"], []).append((track.face_id, item["bbox_px"], color_index))
        for left, right in zip(items, items[1:]):
            gap = right["frame_index"] - left["frame_index"]
            if gap <= 1 or gap > max_gap_frames:
                continue
            for offset in range(1, gap):
                ratio = offset / gap
                box = [int(round(a + ratio * (b - a))) for a, b in zip(left["bbox_px"], right["bbox_px"])]
                boxes.setdefault(left["frame_index"] + offset, []).append((track.face_id, box, color_index))
    return boxes


def render_annotated_video(
    video_path: Path,
    output_dir: Path,
    fps: float,
    width: int,
    height: int,
    frame_count: int,
    tracklets: Sequence[Tracklet],
    observations: Sequence[dict],
    max_gap_frames: int,
) -> Path:
    boxes = _interpolated_boxes(tracklets, observations, max_gap_frames)
    silent_path = output_dir / "annotated.silent.mp4"
    final_path = output_dir / "annotated.mp4"
    silent_path.unlink(missing_ok=True)
    final_path.unlink(missing_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not open the annotated video writer")
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for face_id, box, color_index in boxes.get(frame_index, []):
                _draw_box(frame, box, face_id, COLORS[color_index % len(COLORS)])
            writer.write(frame)
            frame_index += 1
            if frame_index % 1000 == 0:
                print(f"[render] {frame_index}/{frame_count} frames")
    finally:
        capture.release()
        writer.release()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(silent_path),
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(final_path),
        ]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode == 0:
            silent_path.unlink()
            return final_path
    silent_path.replace(final_path)
    return final_path


def analyze_video(
    video_path: Path,
    output_dir: Path,
    sample_fps: float = 5.0,
    detection_threshold: float = 0.70,
    min_face_size: int = 24,
    max_gap_seconds: float = 0.8,
    reid_threshold: float = 0.75,
    cluster_threshold: float = 0.45,
    min_track_observations: int = 3,
    max_people: int = 4,
    render: bool = True,
) -> dict:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero")
    if min_face_size < 1 or min_track_observations < 1 or max_people < 1:
        raise ValueError("min_face_size, min_track_observations and max_people must be positive")
    if max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be greater than zero")
    for name, value in (
        ("detection_threshold", detection_threshold),
        ("reid_threshold", reid_threshold),
        ("cluster_threshold", cluster_threshold),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between zero and one")
    require_models(["yunet", "sface"])
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("video metadata is incomplete")
    sample_step = max(1, int(round(fps / sample_fps)))
    actual_sample_fps = fps / sample_step
    sample_period_ms = max(1, int(round(1000 / actual_sample_fps)))

    detector = cv2.FaceDetectorYN.create(
        str(model_path("yunet")), "", (width, height), detection_threshold, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(str(model_path("sface")), "")
    manager = TrackManager(fps, max_gap_seconds, reid_threshold)
    started = time.monotonic()
    sampled_frames = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_step == 0:
                timestamp_ms = int(round(frame_index * 1000 / fps))
                detections = _detect(detector, recognizer, frame, min_face_size)
                manager.update(detections, frame_index, timestamp_ms, width, height)
                sampled_frames += 1
                if sampled_frames % 100 == 0:
                    print(
                        f"[tracks] {frame_index + 1}/{frame_count} frames, "
                        f"{len(manager.observations)} detections, {len(manager.tracklets)} tracklets"
                    )
            frame_index += 1
    finally:
        capture.release()

    clusters = cluster_tracklets(
        manager.tracklets,
        manager.observations,
        min_track_observations,
        cluster_threshold,
        sample_period_ms,
    )
    faces = _face_records(clusters, manager.observations, sample_period_ms)
    _write_evidence(video_path, output_dir, faces, manager.observations)
    elapsed = time.monotonic() - started
    warnings = []
    if not faces:
        warnings.append("No retained face tracks were found.")
    if len(faces) > max_people:
        warnings.append(
            f"Detected {len(faces)} visible face identities, above the configured conversation-speaker limit of "
            f"{max_people}; non-speaking and background faces remain visual candidates until active-speaker filtering."
        )
    discarded = sum(track.face_id is None for track in manager.tracklets)
    data = {
        "schema_version": 1,
        "session_id": output_dir.name,
        "source": {
            "path": str(video_path),
            "sha256": file_sha256(video_path),
            "duration_ms": int(round(frame_count * 1000 / fps)),
            "fps": round(fps, 6),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "time_base": "frame_index/fps",
        },
        "processing": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sample_fps": round(actual_sample_fps, 6),
            "sample_step_frames": sample_step,
            "detector": _model_record("yunet"),
            "recognizer": _model_record("sface"),
            "parameters": {
                "detection_threshold": detection_threshold,
                "min_face_size": min_face_size,
                "max_gap_seconds": max_gap_seconds,
                "reid_threshold": reid_threshold,
                "cluster_threshold": cluster_threshold,
                "min_track_observations": min_track_observations,
                "max_people": max_people,
            },
            "elapsed_seconds": round(elapsed, 3),
        },
        "faces": faces,
        "tracklets": _tracklet_records(manager.tracklets),
        "observations": manager.observations,
        "active_speaker_scores": [],
        "active_speaker_segments": [],
        "speaker_face_associations": [],
        "statistics": {
            "sampled_frames": sampled_frames,
            "detections": len(manager.observations),
            "retained_faces": len(faces),
            "retained_tracklets": len(manager.tracklets) - discarded,
            "discarded_tracklets": discarded,
        },
        "warnings": warnings,
    }
    output_json = output_dir / "visual_tracks.json"
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if render:
        render_annotated_video(
            video_path,
            output_dir,
            fps,
            width,
            height,
            frame_count,
            manager.tracklets,
            manager.observations,
            manager.max_gap_frames,
        )
    return data


def validate_output(data: dict) -> List[str]:
    errors: List[str] = []
    required = {
        "schema_version",
        "session_id",
        "source",
        "processing",
        "faces",
        "tracklets",
        "observations",
        "active_speaker_segments",
        "speaker_face_associations",
        "statistics",
        "warnings",
    }
    missing = required - set(data)
    if missing:
        errors.append("missing top-level fields: " + ", ".join(sorted(missing)))
        return errors
    face_ids = [item["face_id"] for item in data["faces"]]
    tracklet_ids = [item["tracklet_id"] for item in data["tracklets"]]
    observation_ids = [item["observation_id"] for item in data["observations"]]
    for label, values in (("face", face_ids), ("tracklet", tracklet_ids), ("observation", observation_ids)):
        if len(values) != len(set(values)):
            errors.append(f"duplicate {label} ids")
    face_set, tracklet_set, observation_set = set(face_ids), set(tracklet_ids), set(observation_ids)
    for tracklet in data["tracklets"]:
        if tracklet["face_id"] is not None and tracklet["face_id"] not in face_set:
            errors.append(f"{tracklet['tracklet_id']} references unknown face")
        if not set(tracklet["observation_ids"]).issubset(observation_set):
            errors.append(f"{tracklet['tracklet_id']} references unknown observations")
        if tracklet["start_ms"] > tracklet["end_ms"]:
            errors.append(f"{tracklet['tracklet_id']} has reversed time")
    for observation in data["observations"]:
        if observation["tracklet_id"] not in tracklet_set:
            errors.append(f"{observation['observation_id']} references unknown tracklet")
        if observation["face_id"] is not None and observation["face_id"] not in face_set:
            errors.append(f"{observation['observation_id']} references unknown face")
        normalized_box = observation["bbox_normalized"]
        if len(normalized_box) != 4 or any(value < 0 or value > 1 for value in normalized_box):
            errors.append(f"{observation['observation_id']} has invalid normalized box")
    active_scores = data.get("active_speaker_scores", [])
    score_ids = [item["score_id"] for item in active_scores]
    if len(score_ids) != len(set(score_ids)):
        errors.append("duplicate active-speaker score ids")
    for score in active_scores:
        if score["face_id"] not in face_set:
            errors.append(f"{score['score_id']} references unknown face")
        if score["tracklet_id"] not in tracklet_set:
            errors.append(f"{score['score_id']} references unknown tracklet")
        if not 0 <= score["score"] <= 1:
            errors.append(f"{score['score_id']} has invalid probability")
    active_parameters = data.get("processing", {}).get("active_speaker_parameters", {})
    if active_parameters.get("exclusive_speaker"):
        speaking_buckets: Dict[int, str] = {}
        for score in active_scores:
            if not score["is_speaking"]:
                continue
            bucket = int(round(score["timestamp_ms"] / 40))
            previous = speaking_buckets.setdefault(bucket, score["face_id"])
            if previous != score["face_id"]:
                errors.append(
                    f"active-speaker bucket {bucket} contains both {previous} and {score['face_id']}"
                )
    for segment in data["active_speaker_segments"]:
        if segment["face_id"] is not None and segment["face_id"] not in face_set:
            errors.append(f"{segment.get('segment_id', 'active-speaker segment')} references unknown face")
        if segment["start_ms"] >= segment["end_ms"]:
            errors.append(f"{segment.get('segment_id', 'active-speaker segment')} has invalid time range")
        if not set(segment.get("tracklet_ids", [])).issubset(tracklet_set):
            errors.append(f"{segment.get('segment_id', 'active-speaker segment')} references unknown tracklets")
    return errors
