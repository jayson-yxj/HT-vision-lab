from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from scripts.scene_context import (
    _select_keyframe,
    analyze_scenes,
    build_spatial_relations,
    validate_scene_context,
)


def _state(identifier: str, center, box) -> dict:
    return {
        "person_state_id": identifier,
        "keyframe_id": "keyframe-00001",
        "bbox_normalized": box,
        "center_normalized": center,
        "detection_confidence": 0.9,
    }


def test_spatial_relations_are_evidence_grounded() -> None:
    left = _state("left", [0.2, 0.4], [0.1, 0.2, 0.2, 0.4])
    right = _state("right", [0.7, 0.45], [0.6, 0.25, 0.2, 0.4])
    relations = build_spatial_relations([left, right])
    assert [item["predicate"] for item in relations] == ["co_visible_with", "left_of"]
    assert relations[1]["subject_state_id"] == "left"
    assert relations[1]["object_state_id"] == "right"


def test_keyframe_selection_prioritizes_visible_people() -> None:
    class Capture:
        def __init__(self):
            self.index = 0

        def set(self, _property, value):
            self.index = int(value)

        def read(self):
            if self.index == 1:
                return True, np.full((90, 160, 3), 80, dtype=np.uint8)
            noisy = np.indices((90, 160)).sum(axis=0) % 2 * 255
            return True, np.repeat(noisy[:, :, None], 3, axis=2).astype(np.uint8)

    observations = {
        1: [
            {"bbox_normalized": [0.1, 0.2, 0.2, 0.3]},
            {"bbox_normalized": [0.6, 0.2, 0.2, 0.3]},
        ],
        2: [{"bbox_normalized": [0.1, 0.2, 0.2, 0.3]}],
    }
    selected, _frame, _sharpness_value, _score = _select_keyframe(
        Capture(), [1, 2], observations
    )
    assert selected == 1


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 90))
    if not writer.isOpened():
        raise RuntimeError("test video writer is unavailable")
    try:
        for index in range(40):
            color = (20, 20, 220) if index < 20 else (220, 220, 20)
            frame = np.full((90, 160, 3), color, dtype=np.uint8)
            cv2.putText(frame, str(index), (50, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            writer.write(frame)
    finally:
        writer.release()


def _observation(index: int, frame_index: int, face_id: str, x: float) -> dict:
    return {
        "observation_id": f"observation-{index:06d}",
        "tracklet_id": f"tracklet-{face_id}",
        "face_id": face_id,
        "frame_index": frame_index,
        "timestamp_ms": frame_index * 100,
        "bbox_px": [int(x * 160), 20, 32, 45],
        "bbox_normalized": [x, 0.222222, 0.2, 0.5],
        "landmarks_px": [[0, 0]] * 5,
        "detection_confidence": 0.95,
        "track_match_confidence": 0.9,
        "quality": 0.9,
    }


def test_real_pipeline_writes_shots_keyframes_and_positions() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        video_path = root / "two-shots.avi"
        _write_video(video_path)
        observations = []
        for frame_index in (5, 10, 15, 25, 30, 35):
            observations.append(_observation(len(observations) + 1, frame_index, "Face-01", 0.1))
            observations.append(_observation(len(observations) + 1, frame_index, "Face-02", 0.65))
        visual = {
            "session_id": "synthetic-session",
            "source": {
                "path": str(video_path),
                "sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
                "duration_ms": 4000,
                "fps": 10.0,
                "width": 160,
                "height": 90,
            },
            "processing": {"sample_fps": 2.0},
            "faces": [{"face_id": "Face-01"}, {"face_id": "Face-02"}],
            "observations": observations,
            "speaker_face_associations": [
                {"face_id": "Face-01", "speaker_label": "A", "status": "confirmed"},
                {"face_id": "Face-02", "speaker_label": "B", "status": "ambiguous"},
            ],
        }
        visual_path = root / "visual_tracks.json"
        visual_path.write_text(json.dumps(visual), encoding="utf-8")
        result = analyze_scenes(visual_path)
        assert not validate_scene_context(result)
        assert len(result["shots"]) == 2
        assert len(result["keyframes"]) == 2
        assert len(result["person_states"]) == 4
        assert {item["speaker_label"] for item in result["person_states"]} == {"A", None}
        assert all(item["head_orientation"] is None for item in result["person_states"])
        assert (root / "keyframes" / "keyframe-00001.jpg").is_file()
        assert (root / "keyframes" / "keyframe-00002-annotated.jpg").is_file()


if __name__ == "__main__":
    test_spatial_relations_are_evidence_grounded()
    test_keyframe_selection_prioritizes_visible_people()
    test_real_pipeline_writes_shots_keyframes_and_positions()
    print("PASS: scene context shots, keyframes and spatial relations")
