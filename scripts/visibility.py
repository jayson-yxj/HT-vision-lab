from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .models import file_sha256
from .speaker_face_binding import SpeechInterval, _load_speech_timeline


def detect_shot_boundaries(video_path: Path, threshold: float = 0.08, sample_fps: float = 10.0) -> List[int]:
    if not 0 < threshold <= 1 or sample_fps <= 0:
        raise ValueError("shot threshold and sample_fps must be positive")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    step = max(1, int(round(fps / sample_fps)))
    previous = None
    candidates: List[Tuple[int, float]] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % step == 0:
                gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
                if previous is not None:
                    difference = float(np.mean(cv2.absdiff(previous, gray))) / 255
                    if difference >= threshold:
                        candidates.append((int(round(frame_index * 1000 / fps)), difference))
                previous = gray
            frame_index += 1
    finally:
        capture.release()

    boundaries: List[Tuple[int, float]] = []
    for timestamp_ms, difference in candidates:
        if boundaries and timestamp_ms - boundaries[-1][0] <= 300:
            if difference > boundaries[-1][1]:
                boundaries[-1] = (timestamp_ms, difference)
        else:
            boundaries.append((timestamp_ms, difference))
    return [item[0] for item in boundaries]


def _merge_ranges(ranges: Sequence[Tuple[int, int]], gap_ms: int = 0) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + gap_ms:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _face_ranges(data: dict, face_id: str) -> List[Tuple[int, int]]:
    duration_ms = int(data["source"]["duration_ms"])
    sample_fps = float(data["processing"]["sample_fps"])
    sample_period_ms = max(1, int(round(1000 / sample_fps)))
    ranges = [
        (int(item["start_ms"]), min(duration_ms, int(item["end_ms"]) + sample_period_ms))
        for item in data["tracklets"]
        if item.get("face_id") == face_id
    ]
    return _merge_ranges(ranges)


def _overlapping_tracklets(data: dict, face_id: str, start_ms: int, end_ms: int) -> List[str]:
    return sorted(
        item["tracklet_id"]
        for item in data["tracklets"]
        if item.get("face_id") == face_id
        and int(item["end_ms"]) >= start_ms
        and int(item["start_ms"]) < end_ms
    )


def _cuts_between(cuts: Sequence[int], start_ms: int, end_ms: int) -> bool:
    return any(start_ms <= item <= end_ms for item in cuts)


def _confirmed_states(
    data: dict,
    face_id: str,
    interval: SpeechInterval,
    cuts: Sequence[int],
    occlusion_max_ms: int,
    boundary_tolerance_ms: int,
) -> List[dict]:
    visible_ranges = _face_ranges(data, face_id)
    clipped = [
        (max(interval.start_ms, start), min(interval.end_ms, end))
        for start, end in visible_ranges
        if end > interval.start_ms and start < interval.end_ms
    ]
    clipped = _merge_ranges([(start, end) for start, end in clipped if end > start])
    boundaries = {interval.start_ms, interval.end_ms}
    for start, end in clipped:
        boundaries.update((start, end))
    points = sorted(boundaries)
    segments = []
    for start, end in zip(points, points[1:]):
        if end <= start:
            continue
        midpoint = (start + end) // 2
        visible = any(left <= midpoint < right for left, right in clipped)
        if visible:
            state, reason, confidence = "visible", "face_track_present", None
        else:
            duration = end - start
            touches_boundary = start == interval.start_ms or end == interval.end_ms
            previous = max((right for left, right in clipped if right <= start), default=None)
            following = min((left for left, right in clipped if left >= end), default=None)
            if touches_boundary and duration <= boundary_tolerance_ms:
                state, reason, confidence = "visible", "timestamp_boundary_tolerance", None
            elif (
                previous is not None
                and following is not None
                and following - previous <= occlusion_max_ms
                and not _cuts_between(cuts, previous, following)
            ):
                state, reason, confidence = "occluded", "short_gap_within_same_shot", 0.65
            else:
                state, reason, confidence = "offscreen", "bound_face_absent", 0.85
        segments.append(
            {
                "speaker_label": "",
                "face_id": face_id,
                "start_ms": start,
                "end_ms": end,
                "state": state,
                "confidence": confidence,
                "reason": reason,
                "source_speech_span_ids": list(interval.source_ids),
                "tracklet_ids": _overlapping_tracklets(data, face_id, start, end) if state == "visible" else [],
            }
        )
    return segments


def _merge_state_segments(segments: Sequence[dict]) -> List[dict]:
    merged: List[dict] = []
    for item in sorted(segments, key=lambda value: (value["start_ms"], value["end_ms"])):
        if (
            merged
            and merged[-1]["speaker_label"] == item["speaker_label"]
            and merged[-1]["face_id"] == item["face_id"]
            and merged[-1]["state"] == item["state"]
            and merged[-1]["reason"] == item["reason"]
            and item["start_ms"] <= merged[-1]["end_ms"]
        ):
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], item["end_ms"])
            merged[-1]["source_speech_span_ids"] = list(
                dict.fromkeys(merged[-1]["source_speech_span_ids"] + item["source_speech_span_ids"])
            )
            merged[-1]["tracklet_ids"] = sorted(
                set(merged[-1]["tracklet_ids"] + item["tracklet_ids"])
            )
        else:
            merged.append(dict(item))
    return merged


def build_visibility_segments(
    data: dict,
    speech_by_speaker: Dict[str, List[SpeechInterval]],
    cuts: Sequence[int],
    occlusion_max_ms: int = 1200,
    boundary_tolerance_ms: int = 240,
) -> List[dict]:
    associations = {item["speaker_label"]: item for item in data["speaker_face_associations"]}
    pending = []
    for speaker, intervals in speech_by_speaker.items():
        association = associations.get(speaker)
        for interval in intervals:
            if association is None or association["status"] in {"candidate", "ambiguous"}:
                items = [
                    {
                        "speaker_label": speaker,
                        "face_id": association.get("face_id") if association else None,
                        "start_ms": interval.start_ms,
                        "end_ms": interval.end_ms,
                        "state": "unknown",
                        "confidence": 0.9,
                        "reason": "identity_binding_unresolved",
                        "source_speech_span_ids": list(interval.source_ids),
                        "tracklet_ids": [],
                    }
                ]
            elif association["status"] == "offscreen" or association.get("face_id") is None:
                items = [
                    {
                        "speaker_label": speaker,
                        "face_id": None,
                        "start_ms": interval.start_ms,
                        "end_ms": interval.end_ms,
                        "state": "offscreen",
                        "confidence": 0.75,
                        "reason": "no_bound_visible_face",
                        "source_speech_span_ids": list(interval.source_ids),
                        "tracklet_ids": [],
                    }
                ]
            else:
                items = _confirmed_states(
                    data,
                    association["face_id"],
                    interval,
                    cuts,
                    occlusion_max_ms,
                    boundary_tolerance_ms,
                )
                for item in items:
                    item["speaker_label"] = speaker
                    if item["confidence"] is None:
                        item["confidence"] = association["confidence"]
                    else:
                        item["confidence"] = min(item["confidence"], association["confidence"])
            pending.extend(items)
    segments = _merge_state_segments(pending)
    for index, segment in enumerate(segments, start=1):
        segment["visibility_id"] = f"speaker-visibility-{index:05d}"
        segment["confidence"] = round(float(segment["confidence"]), 6)
    return segments


def classify_visibility(
    visual_path: Path,
    output_path: Optional[Path] = None,
    shot_threshold: float = 0.08,
    shot_sample_fps: float = 10.0,
    occlusion_max_ms: int = 1200,
    boundary_tolerance_ms: int = 240,
) -> dict:
    if occlusion_max_ms < 0 or boundary_tolerance_ms < 0:
        raise ValueError("visibility durations cannot be negative")
    visual_path = visual_path.expanduser().resolve()
    with visual_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not data.get("speaker_face_associations"):
        raise RuntimeError("visual tracks contain no speaker-face bindings; run ./lab bind-speakers first")
    binding = data.get("processing", {}).get("speaker_face_binding", {})
    speech_path = Path(binding.get("speech_timeline_path", "")).expanduser().resolve()
    expected_hash = binding.get("speech_timeline_sha256")
    if not speech_path.is_file() or file_sha256(speech_path) != expected_hash:
        raise RuntimeError("speaker-face binding speech timeline is unavailable or has changed")
    duration_ms = int(data["source"]["duration_ms"])
    speech = _load_speech_timeline(
        speech_path, duration_ms, int(binding.get("timeline_offset_ms", 0))
    )
    video_path = Path(data["source"]["path"])
    cuts = detect_shot_boundaries(video_path, shot_threshold, shot_sample_fps)
    segments = build_visibility_segments(
        data, speech, cuts, occlusion_max_ms, boundary_tolerance_ms
    )
    data["speaker_visibility_segments"] = segments
    data["processing"]["visibility_classification"] = {
        "algorithm": "speech_conditioned_face_visibility_v1",
        "shot_threshold": shot_threshold,
        "shot_sample_fps": shot_sample_fps,
        "shot_boundaries_ms": cuts,
        "occlusion_max_ms": occlusion_max_ms,
        "boundary_tolerance_ms": boundary_tolerance_ms,
    }
    data["statistics"]["speaker_visibility_segment_count"] = len(segments)
    data["statistics"]["speaker_visibility_state_counts"] = {
        state: sum(item["state"] == state for item in segments)
        for state in ("visible", "offscreen", "occluded", "unknown")
    }
    data["warnings"] = [item for item in data["warnings"] if not item.startswith("Visibility state")]
    if any(item["state"] == "unknown" for item in segments):
        data["warnings"].append("Visibility state remains unknown where speaker-face identity is unresolved.")

    output_path = output_path.expanduser().resolve() if output_path else visual_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output_path)
    return data
