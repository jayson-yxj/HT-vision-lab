from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector

from .models import file_sha256


def detect_shots(
    video_path: Path,
    adaptive_threshold: float = 3.0,
    min_content_val: float = 15.0,
    min_shot_seconds: float = 0.5,
) -> List[Tuple[int, int]]:
    if adaptive_threshold <= 0 or min_content_val < 0 or min_shot_seconds <= 0:
        raise ValueError("shot detector parameters must be positive")
    video = open_video(str(video_path))
    minimum_frames = max(1, int(round(video.frame_rate * min_shot_seconds)))
    manager = SceneManager()
    manager.add_detector(
        AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
            min_content_val=min_content_val,
            min_scene_len=minimum_frames,
        )
    )
    manager.detect_scenes(video, show_progress=False)
    scenes = manager.get_scene_list(start_in_scene=True)
    return [
        (int(round(start.get_seconds() * 1000)), int(round(end.get_seconds() * 1000)))
        for start, end in scenes
        if end > start
    ]


def _confirmed_speakers(data: dict) -> Dict[str, str]:
    return {
        item["face_id"]: item["speaker_label"]
        for item in data.get("speaker_face_associations", [])
        if item.get("status") == "confirmed" and item.get("face_id")
    }


def _candidate_frames(
    start_ms: int,
    end_ms: int,
    fps: float,
    observation_frames: Sequence[int],
) -> List[int]:
    start_frame = max(0, int(math.ceil(start_ms * fps / 1000)))
    end_frame = max(start_frame + 1, int(math.ceil(end_ms * fps / 1000)))
    last_frame = end_frame - 1
    candidates = []
    for ratio in (0.2, 0.5, 0.8):
        target = min(last_frame, max(start_frame, int(round(start_frame + ratio * (last_frame - start_frame)))))
        candidates.append(target)
        if observation_frames:
            candidates.append(min(observation_frames, key=lambda value: abs(value - target)))
    return list(dict.fromkeys(candidates))


def _read_frame(capture: cv2.VideoCapture, frame_index: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    return frame if ok else None


def _sharpness(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _normalized(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high <= low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _select_keyframe(
    capture: cv2.VideoCapture,
    candidates: Sequence[int],
    observations_by_frame: Dict[int, List[dict]],
) -> Tuple[int, object, float, float]:
    rows = []
    for frame_index in candidates:
        frame = _read_frame(capture, frame_index)
        if frame is None:
            continue
        observations = observations_by_frame.get(frame_index, [])
        rows.append(
            {
                "frame_index": frame_index,
                "frame": frame,
                "sharpness": _sharpness(frame),
                "face_count": len(observations),
                "face_area": sum(
                    float(item["bbox_normalized"][2]) * float(item["bbox_normalized"][3])
                    for item in observations
                ),
            }
        )
    if not rows:
        raise RuntimeError("could not decode any keyframe candidates")
    sharpness = _normalized([item["sharpness"] for item in rows])
    face_counts = _normalized([float(item["face_count"]) for item in rows])
    face_areas = _normalized([item["face_area"] for item in rows])
    for item, sharp, count, area in zip(rows, sharpness, face_counts, face_areas):
        item["score"] = 0.45 * sharp + 0.35 * count + 0.20 * area
    selected = max(
        rows,
        key=lambda item: (item["face_count"], item["score"], -item["frame_index"]),
    )
    return (
        selected["frame_index"],
        selected["frame"],
        selected["sharpness"],
        selected["score"],
    )


def _region(value: float, labels: Sequence[str]) -> str:
    return labels[min(len(labels) - 1, int(value * len(labels)))]


def _person_state(
    observation: dict,
    keyframe_id: str,
    speaker_by_face: Dict[str, str],
) -> dict:
    x, y, width, height = [float(value) for value in observation["bbox_normalized"]]
    center = [round(x + width / 2, 6), round(y + height / 2, 6)]
    return {
        "keyframe_id": keyframe_id,
        "face_id": observation["face_id"],
        "speaker_label": speaker_by_face.get(observation["face_id"]),
        "identity_status": "confirmed" if observation["face_id"] in speaker_by_face else "visual_only",
        "source_observation_id": observation["observation_id"],
        "bbox_normalized": [round(value, 6) for value in (x, y, width, height)],
        "center_normalized": center,
        "horizontal_region": _region(center[0], ("left", "center", "right")),
        "vertical_region": _region(center[1], ("upper", "middle", "lower")),
        "area_ratio": round(width * height, 6),
        "detection_confidence": observation["detection_confidence"],
        "head_orientation": None,
    }


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection_w = max(0.0, min(lx + lw, rx + rw) - max(lx, rx))
    intersection_h = max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    intersection = intersection_w * intersection_h
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def build_spatial_relations(
    states: Sequence[dict],
    horizontal_threshold: float = 0.08,
    vertical_threshold: float = 0.08,
    near_threshold: float = 0.35,
    overlap_iou_threshold: float = 0.10,
    start_index: int = 1,
) -> List[dict]:
    if not 0 <= horizontal_threshold <= 1 or not 0 <= vertical_threshold <= 1:
        raise ValueError("position thresholds must be between zero and one")
    if not 0 < near_threshold <= math.sqrt(2) or not 0 < overlap_iou_threshold <= 1:
        raise ValueError("distance and overlap thresholds are invalid")
    relations = []

    def append(subject: dict, predicate: str, object_: dict, confidence: float) -> None:
        relations.append(
            {
                "relation_id": f"spatial-relation-{start_index + len(relations):06d}",
                "keyframe_id": subject["keyframe_id"],
                "subject_state_id": subject["person_state_id"],
                "predicate": predicate,
                "object_state_id": object_["person_state_id"],
                "coordinate_space": "image_normalized",
                "confidence": round(max(0.0, min(1.0, confidence)), 6),
            }
        )

    for left_index, left in enumerate(states):
        for right in states[left_index + 1 :]:
            base = min(float(left["detection_confidence"]), float(right["detection_confidence"]))
            append(left, "co_visible_with", right, base)
            dx = right["center_normalized"][0] - left["center_normalized"][0]
            dy = right["center_normalized"][1] - left["center_normalized"][1]
            if abs(dx) >= horizontal_threshold:
                subject, object_ = (left, right) if dx > 0 else (right, left)
                confidence = base * (0.5 + 0.5 * min(1.0, abs(dx) / 0.5))
                append(subject, "left_of", object_, confidence)
            if abs(dy) >= vertical_threshold:
                subject, object_ = (left, right) if dy > 0 else (right, left)
                confidence = base * (0.5 + 0.5 * min(1.0, abs(dy) / 0.5))
                append(subject, "above", object_, confidence)
            distance = math.hypot(dx, dy)
            if distance <= near_threshold:
                confidence = base * (0.5 + 0.5 * (1.0 - distance / near_threshold))
                append(left, "visually_close_to", right, confidence)
            overlap = _box_iou(left["bbox_normalized"], right["bbox_normalized"])
            if overlap >= overlap_iou_threshold:
                append(left, "overlaps_in_image", right, base * overlap)
    return relations


def _draw_annotations(frame, states: Sequence[dict]):
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for state in states:
        x, y, box_width, box_height = state["bbox_normalized"]
        left, top = int(round(x * width)), int(round(y * height))
        right = int(round((x + box_width) * width))
        bottom = int(round((y + box_height) * height))
        label = state["face_id"]
        if state["speaker_label"]:
            label = f"{state['speaker_label']} / {label}"
        cv2.rectangle(annotated, (left, top), (right, bottom), (0, 220, 255), 2)
        cv2.putText(
            annotated,
            label,
            (left, max(18, top - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def analyze_scenes(
    visual_path: Path,
    output_path: Optional[Path] = None,
    adaptive_threshold: float = 3.0,
    min_content_val: float = 15.0,
    min_shot_seconds: float = 0.5,
    horizontal_threshold: float = 0.08,
    vertical_threshold: float = 0.08,
    near_threshold: float = 0.35,
    overlap_iou_threshold: float = 0.10,
) -> dict:
    visual_path = visual_path.expanduser().resolve()
    with visual_path.open(encoding="utf-8") as handle:
        visual = json.load(handle)
    video_path = Path(visual["source"]["path"]).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"source video is unavailable: {video_path}")
    if file_sha256(video_path) != visual["source"]["sha256"]:
        raise ValueError("source video has changed since visual tracks were created")
    output_path = (
        output_path.expanduser().resolve()
        if output_path
        else visual_path.parent / "scene_context.json"
    )
    if output_path in {visual_path, video_path}:
        raise ValueError("scene context output must not overwrite either source file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keyframe_dir = output_path.parent / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    shots = detect_shots(video_path, adaptive_threshold, min_content_val, min_shot_seconds)
    duration_ms = int(visual["source"]["duration_ms"])
    if not shots and duration_ms > 0:
        shots = [(0, duration_ms)]
    elif shots:
        shots[-1] = (shots[-1][0], duration_ms)

    observations_by_frame: Dict[int, List[dict]] = {}
    for observation in visual["observations"]:
        if observation.get("face_id"):
            observations_by_frame.setdefault(int(observation["frame_index"]), []).append(observation)
    observation_frames = sorted(observations_by_frame)
    speaker_by_face = _confirmed_speakers(visual)
    visual_entities = [
        {
            "face_id": face["face_id"],
            "speaker_label": speaker_by_face.get(face["face_id"]),
            "identity_status": "confirmed" if face["face_id"] in speaker_by_face else "visual_only",
        }
        for face in visual["faces"]
    ]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(visual["source"]["fps"])
    shot_records, keyframes, person_states, relations = [], [], [], []
    try:
        for shot_index, (start_ms, end_ms) in enumerate(shots, start=1):
            shot_id = f"shot-{shot_index:05d}"
            frames_in_shot = [
                frame_index
                for frame_index in observation_frames
                if start_ms <= int(round(frame_index * 1000 / fps)) < end_ms
            ]
            candidates = _candidate_frames(start_ms, end_ms, fps, frames_in_shot)
            frame_index, frame, sharpness, score = _select_keyframe(
                capture, candidates, observations_by_frame
            )
            timestamp_ms = min(duration_ms, int(round(frame_index * 1000 / fps)))
            keyframe_id = f"keyframe-{shot_index:05d}"
            states = [
                _person_state(item, keyframe_id, speaker_by_face)
                for item in observations_by_frame.get(frame_index, [])
            ]
            states.sort(key=lambda item: (item["center_normalized"][0], item["face_id"]))
            for offset, state in enumerate(states, start=1):
                state["person_state_id"] = f"person-state-{len(person_states) + offset:06d}"
            frame_relations = build_spatial_relations(
                states,
                horizontal_threshold,
                vertical_threshold,
                near_threshold,
                overlap_iou_threshold,
                start_index=len(relations) + 1,
            )
            raw_path = keyframe_dir / f"{keyframe_id}.jpg"
            annotated_path = keyframe_dir / f"{keyframe_id}-annotated.jpg"
            if not cv2.imwrite(str(raw_path), frame):
                raise RuntimeError(f"could not write keyframe: {raw_path}")
            if not cv2.imwrite(str(annotated_path), _draw_annotations(frame, states)):
                raise RuntimeError(f"could not write annotated keyframe: {annotated_path}")
            keyframes.append(
                {
                    "keyframe_id": keyframe_id,
                    "shot_id": shot_id,
                    "timestamp_ms": timestamp_ms,
                    "frame_index": frame_index,
                    "image_path": str(raw_path.relative_to(output_path.parent)),
                    "annotated_image_path": str(annotated_path.relative_to(output_path.parent)),
                    "sharpness": round(sharpness, 6),
                    "selection_score": round(score, 6),
                    "person_state_ids": [item["person_state_id"] for item in states],
                }
            )
            shot_records.append(
                {
                    "shot_id": shot_id,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "keyframe_ids": [keyframe_id],
                    "semantic_scene_id": None,
                }
            )
            person_states.extend(states)
            relations.extend(frame_relations)
    finally:
        capture.release()

    data = {
        "schema_version": 1,
        "context_type": "visual_scene_context",
        "session_id": visual["session_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "visual_tracks_path": str(visual_path),
            "visual_tracks_sha256": file_sha256(visual_path),
            "video_path": str(video_path),
            "video_sha256": visual["source"]["sha256"],
            "duration_ms": duration_ms,
            "fps": fps,
            "width": int(visual["source"]["width"]),
            "height": int(visual["source"]["height"]),
        },
        "processing": {
            "shot_detector": {
                "name": "PySceneDetect AdaptiveDetector",
                "version": version("scenedetect"),
                "adaptive_threshold": adaptive_threshold,
                "min_content_val": min_content_val,
                "min_shot_seconds": min_shot_seconds,
            },
            "keyframe_selector": "best_of_20_50_80_by_sharpness_and_visible_faces_v2",
            "spatial_relations": {
                "algorithm": "normalized_face_box_geometry_v1",
                "horizontal_threshold": horizontal_threshold,
                "vertical_threshold": vertical_threshold,
                "near_threshold": near_threshold,
                "overlap_iou_threshold": overlap_iou_threshold,
            },
        },
        "visual_entities": visual_entities,
        "shots": shot_records,
        "keyframes": keyframes,
        "person_states": person_states,
        "spatial_relations": relations,
        "statistics": {
            "shots": len(shot_records),
            "keyframes": len(keyframes),
            "person_states": len(person_states),
            "spatial_relations": len(relations),
        },
        "warnings": [],
    }
    errors = validate_scene_context(data)
    if errors:
        raise RuntimeError("scene context validation failed: " + "; ".join(errors))
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output_path)
    return data


def validate_scene_context(data: dict) -> List[str]:
    errors = []
    required = {
        "schema_version",
        "context_type",
        "session_id",
        "created_at",
        "source",
        "processing",
        "visual_entities",
        "shots",
        "keyframes",
        "person_states",
        "spatial_relations",
        "statistics",
        "warnings",
    }
    missing = required - set(data)
    if missing:
        return ["missing top-level fields: " + ", ".join(sorted(missing))]
    duration = int(data["source"]["duration_ms"])
    face_id_list = [item["face_id"] for item in data["visual_entities"]]
    face_ids = set(face_id_list)
    if len(face_id_list) != len(face_ids):
        errors.append("duplicate visual entity IDs")
    shot_ids = [item["shot_id"] for item in data["shots"]]
    keyframe_ids = [item["keyframe_id"] for item in data["keyframes"]]
    state_ids = [item["person_state_id"] for item in data["person_states"]]
    relation_ids = [item["relation_id"] for item in data["spatial_relations"]]
    for label, values in (
        ("shot", shot_ids),
        ("keyframe", keyframe_ids),
        ("person state", state_ids),
        ("spatial relation", relation_ids),
    ):
        if len(values) != len(set(values)):
            errors.append(f"duplicate {label} IDs")
    shot_set, keyframe_set, state_set = set(shot_ids), set(keyframe_ids), set(state_ids)
    previous_end = 0
    for shot in data["shots"]:
        if shot["start_ms"] != previous_end:
            errors.append(f"{shot['shot_id']} does not begin at the previous shot boundary")
        if shot["start_ms"] >= shot["end_ms"] or shot["end_ms"] > duration:
            errors.append(f"{shot['shot_id']} has an invalid time range")
        if not set(shot["keyframe_ids"]).issubset(keyframe_set):
            errors.append(f"{shot['shot_id']} references unknown keyframes")
        previous_end = shot["end_ms"]
    if data["shots"] and previous_end != duration:
        errors.append("shots do not cover the full source duration")
    for keyframe in data["keyframes"]:
        if keyframe["shot_id"] not in shot_set:
            errors.append(f"{keyframe['keyframe_id']} references an unknown shot")
        if not set(keyframe["person_state_ids"]).issubset(state_set):
            errors.append(f"{keyframe['keyframe_id']} references unknown person states")
        if not 0 <= keyframe["timestamp_ms"] <= duration:
            errors.append(f"{keyframe['keyframe_id']} is outside the source duration")
    for state in data["person_states"]:
        if state["keyframe_id"] not in keyframe_set:
            errors.append(f"{state['person_state_id']} references an unknown keyframe")
        if state["face_id"] not in face_ids:
            errors.append(f"{state['person_state_id']} references an unknown face")
        box = state["bbox_normalized"]
        if len(box) != 4 or any(value < 0 or value > 1 for value in box):
            errors.append(f"{state['person_state_id']} has an invalid box")
    valid_predicates = {
        "co_visible_with",
        "left_of",
        "above",
        "visually_close_to",
        "overlaps_in_image",
    }
    for relation in data["spatial_relations"]:
        if relation["keyframe_id"] not in keyframe_set:
            errors.append(f"{relation['relation_id']} references an unknown keyframe")
        if relation["subject_state_id"] not in state_set or relation["object_state_id"] not in state_set:
            errors.append(f"{relation['relation_id']} references an unknown person state")
        if relation["predicate"] not in valid_predicates:
            errors.append(f"{relation['relation_id']} has an unknown predicate")
        if not 0 <= relation["confidence"] <= 1:
            errors.append(f"{relation['relation_id']} has invalid confidence")
    expected_stats = {
        "shots": len(data["shots"]),
        "keyframes": len(data["keyframes"]),
        "person_states": len(data["person_states"]),
        "spatial_relations": len(data["spatial_relations"]),
    }
    if data["statistics"] != expected_stats:
        errors.append("statistics do not match scene context contents")
    return errors
