from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from scripts.scene_semantics import analyze_scene_semantics, validate_scene_semantics


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(root: Path) -> Path:
    speech_path = root / "speech.json"
    speech_path.write_text(
        json.dumps(
            [
                {
                    "id": "span-a",
                    "speaker": "A",
                    "start_s": 0,
                    "end_s": 1,
                    "text": "请大家看屏幕",
                    "status": "final",
                    "attribution": "single_speaker",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    visual_path = root / "visual_tracks.json"
    visual_path.write_text(
        json.dumps(
            {
                "processing": {
                    "speaker_face_binding": {
                        "speech_timeline_path": str(speech_path),
                        "speech_timeline_sha256": _sha(speech_path),
                        "timeline_offset_ms": 0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    keyframe_dir = root / "keyframes"
    keyframe_dir.mkdir()
    for index in (1, 2):
        frame = np.full((80, 120, 3), 40 * index, dtype=np.uint8)
        cv2.putText(frame, f"Face-0{index}", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imwrite(str(keyframe_dir / f"keyframe-0000{index}-annotated.jpg"), frame)
    scene = {
        "schema_version": 1,
        "context_type": "visual_scene_context",
        "session_id": "semantic-test",
        "created_at": "2026-01-01T00:00:00+00:00",
        "source": {
            "visual_tracks_path": str(visual_path),
            "visual_tracks_sha256": _sha(visual_path),
            "video_path": str(root / "video.mp4"),
            "video_sha256": "0" * 64,
            "duration_ms": 2000,
            "fps": 10.0,
            "width": 120,
            "height": 80,
        },
        "processing": {
            "shot_detector": {
                "name": "PySceneDetect AdaptiveDetector",
                "version": "test",
                "adaptive_threshold": 3.0,
                "min_content_val": 15.0,
                "min_shot_seconds": 0.5,
            },
            "keyframe_selector": "best_of_20_50_80_by_sharpness_and_visible_faces_v1",
            "spatial_relations": {
                "algorithm": "normalized_face_box_geometry_v1",
                "horizontal_threshold": 0.08,
                "vertical_threshold": 0.08,
                "near_threshold": 0.35,
                "overlap_iou_threshold": 0.1,
            },
        },
        "visual_entities": [
            {"face_id": "Face-01", "speaker_label": "A", "identity_status": "confirmed"},
            {"face_id": "Face-02", "speaker_label": None, "identity_status": "visual_only"},
        ],
        "shots": [
            {"shot_id": "shot-00001", "start_ms": 0, "end_ms": 1000, "keyframe_ids": ["keyframe-00001"], "semantic_scene_id": None},
            {"shot_id": "shot-00002", "start_ms": 1000, "end_ms": 2000, "keyframe_ids": ["keyframe-00002"], "semantic_scene_id": None},
        ],
        "keyframes": [
            {
                "keyframe_id": f"keyframe-0000{index}",
                "shot_id": f"shot-0000{index}",
                "timestamp_ms": 500 if index == 1 else 1500,
                "frame_index": 5 if index == 1 else 15,
                "image_path": f"keyframes/keyframe-0000{index}.jpg",
                "annotated_image_path": f"keyframes/keyframe-0000{index}-annotated.jpg",
                "sharpness": 10.0,
                "selection_score": 1.0,
                "person_state_ids": [f"person-state-00000{index}"],
            }
            for index in (1, 2)
        ],
        "person_states": [
            {
                "person_state_id": f"person-state-00000{index}",
                "keyframe_id": f"keyframe-0000{index}",
                "face_id": f"Face-0{index}",
                "speaker_label": "A" if index == 1 else None,
                "identity_status": "confirmed" if index == 1 else "visual_only",
                "source_observation_id": f"observation-{index}",
                "bbox_normalized": [0.1, 0.2, 0.2, 0.4],
                "center_normalized": [0.2, 0.4],
                "horizontal_region": "left",
                "vertical_region": "middle",
                "area_ratio": 0.08,
                "detection_confidence": 0.95,
                "head_orientation": None,
            }
            for index in (1, 2)
        ],
        "spatial_relations": [],
        "statistics": {"shots": 2, "keyframes": 2, "person_states": 2, "spatial_relations": 0},
        "warnings": [],
    }
    scene_path = root / "scene_context.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    return scene_path


def test_qwen_batches_images_transcript_and_cache() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scene_path = _write_fixture(root)
        calls = []

        def fake_transport(payload):
            calls.append(payload)
            content = payload["messages"][1]["content"]
            assert sum(item["type"] == "image_url" for item in content) == 2
            assert "请大家看屏幕" in content[0]["text"]
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "keyframes": [
                                        {
                                            "keyframe_id": "keyframe-00001",
                                            "environment": {"category": "indoor_meeting", "description": "会议空间", "confidence": 0.9},
                                            "objects": [{"label": "screen", "count": 1, "region": "background", "confidence": 0.8}],
                                            "interactions": [
                                                {
                                                    "subject_ref": "Face-01",
                                                    "predicate": "presenting",
                                                    "object_ref": "object:screen",
                                                    "description": "A 正在展示屏幕内容",
                                                    "evidence_basis": ["image", "transcript"],
                                                    "epistemic_status": "inferred",
                                                    "confidence": 0.78,
                                                }
                                            ],
                                        },
                                        {
                                            "keyframe_id": "keyframe-00002",
                                            "environment": {"category": "indoor_meeting", "description": "会议空间", "confidence": 0.8},
                                            "objects": [],
                                            "interactions": [
                                                {
                                                    "subject_ref": "Face-02",
                                                    "predicate": "speaking",
                                                    "object_ref": "object:microphone",
                                                    "description": "Face-02 对麦克风说话",
                                                    "evidence_basis": ["image"],
                                                    "epistemic_status": "observed",
                                                    "confidence": 0.7,
                                                }
                                            ],
                                        },
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        result = analyze_scene_semantics(scene_path, transport=fake_transport)
        assert not validate_scene_semantics(result)
        assert result["status"] == "complete"
        assert result["statistics"]["semantic_objects"] == 1
        assert result["statistics"]["interactions"] == 1
        assert result["analyses"][0]["interactions"][0]["epistemic_status"] == "inferred"
        assert len(calls) == 1
        cached = analyze_scene_semantics(scene_path, transport=fake_transport)
        assert len(calls) == 1
        assert cached["requests"][0]["cache_hit"] is True


def test_missing_batched_keyframe_falls_back_to_single_images() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scene_path = _write_fixture(root)
        calls = []

        def analysis(keyframe_id):
            return {
                "keyframe_id": keyframe_id,
                "environment": {"category": "indoor_meeting", "description": "会议空间", "confidence": 0.8},
                "objects": [],
                "interactions": [],
            }

        def fake_transport(payload):
            calls.append(payload)
            text = payload["messages"][1]["content"][0]["text"]
            requested = [
                value
                for value in ("keyframe-00001", "keyframe-00002")
                if value in text
            ]
            returned = requested[:1] if len(requested) > 1 else requested
            return {
                "choices": [
                    {"message": {"content": json.dumps({"keyframes": [analysis(value) for value in returned]}, ensure_ascii=False)}}
                ]
            }

        result = analyze_scene_semantics(scene_path, transport=fake_transport)
        assert result["status"] == "complete"
        assert len(result["analyses"]) == 2
        assert len(calls) == 3
        assert "one image at a time" in result["warnings"][0]
        cached = analyze_scene_semantics(scene_path, transport=fake_transport)
        assert cached["status"] == "complete"
        assert len(calls) == 3
        assert all(item["cache_hit"] for item in cached["requests"])


if __name__ == "__main__":
    test_qwen_batches_images_transcript_and_cache()
    test_missing_batched_keyframe_falls_back_to_single_images()
    print("PASS: Qwen scene semantics image batching, transcript evidence and cache")
