from __future__ import annotations

from scripts.vision_evaluation import (
    score_vision_predictions,
    validate_evaluation_annotations,
)


def test_manual_annotations_score_each_visual_dimension() -> None:
    annotations = {
        "schema_version": 1,
        "context_type": "vision_evaluation_annotations",
        "sample_id": "fixture",
        "source": {"video_sha256": "0" * 64, "duration_ms": 2000},
        "provenance": {
            "method": "manual_evidence_review",
            "reviewed_at": "2026-10-07",
            "notes": "fixture",
        },
        "identity_bindings": [
            {"speaker_label": "A", "face_id": "Face-01"},
            {"speaker_label": "B", "face_id": "Face-02"},
        ],
        "object_vocabulary": ["chair", "table"],
        "interaction_predicates": ["speaking"],
        "keyframes": [
            {
                "keyframe_id": "keyframe-00001",
                "timestamp_ms": 500,
                "environment_category": "indoor_meeting",
                "object_labels": ["chair", "table"],
                "interactions": [
                    {"subject_ref": "Face-01", "predicate": "speaking", "object_ref": "group"}
                ],
                "spatial_relations": [
                    {"subject_ref": "Face-02", "predicate": "co_visible_with", "object_ref": "Face-01"},
                    {"subject_ref": "Face-01", "predicate": "left_of", "object_ref": "Face-02"},
                ],
            },
            {
                "keyframe_id": "keyframe-00002",
                "timestamp_ms": 1500,
                "environment_category": "outdoor_public",
                "object_labels": [],
                "interactions": [],
                "spatial_relations": [],
            },
        ],
    }
    visual = {
        "speaker_face_associations": [
            {"speaker_label": "A", "face_id": "Face-01", "status": "confirmed"},
            {"speaker_label": "B", "face_id": "Face-03", "status": "confirmed"},
        ]
    }
    scene = {
        "keyframes": [
            {"keyframe_id": "keyframe-00001", "timestamp_ms": 500},
            {"keyframe_id": "keyframe-00002", "timestamp_ms": 1500},
        ],
        "person_states": [
            {"person_state_id": "state-1", "face_id": "Face-01"},
            {"person_state_id": "state-2", "face_id": "Face-02"},
        ],
        "spatial_relations": [
            {
                "keyframe_id": "keyframe-00001",
                "subject_state_id": "state-1",
                "predicate": "co_visible_with",
                "object_state_id": "state-2",
            }
        ],
    }
    semantics = {
        "analyses": [
            {
                "keyframe_id": "keyframe-00001",
                "environment": {"category": "indoor_meeting"},
                "objects": [{"label": "chair"}, {"label": "plant"}],
                "interactions": [
                    {"subject_ref": "Face-01", "predicate": "speaking", "object_ref": "group"}
                ],
            },
            {
                "keyframe_id": "keyframe-00002",
                "environment": {"category": "indoor_office"},
                "objects": [],
                "interactions": [
                    {"subject_ref": "Face-02", "predicate": "speaking", "object_ref": "group"}
                ],
            },
        ]
    }

    assert validate_evaluation_annotations(annotations) == []
    result = score_vision_predictions(annotations, visual, scene, semantics)
    metrics = result["metrics"]
    assert metrics["identity_binding"]["accuracy"] == 0.5
    assert metrics["environment"]["accuracy"] == 0.5
    assert metrics["objects"]["f1"] == 0.666667
    assert metrics["interactions"]["f1"] == 0.666667
    assert metrics["spatial_relations"]["f1"] == 0.666667
    assert result["summary"]["macro_score"] == 0.6


def test_omitted_dimensions_are_not_scored() -> None:
    annotations = {
        "schema_version": 1,
        "context_type": "vision_evaluation_annotations",
        "sample_id": "fixture",
        "source": {"video_sha256": "0" * 64, "duration_ms": 1000},
        "provenance": {
            "method": "manual_evidence_review",
            "reviewed_at": "2026-10-07",
            "notes": "fixture",
        },
        "identity_bindings": [],
        "keyframes": [
            {
                "keyframe_id": "keyframe-00001",
                "timestamp_ms": 500,
                "environment_category": "unknown",
            }
        ],
    }
    result = score_vision_predictions(
        annotations,
        {"speaker_face_associations": []},
        {
            "keyframes": [{"keyframe_id": "keyframe-00001", "timestamp_ms": 500}],
            "person_states": [],
            "spatial_relations": [],
        },
        {
            "analyses": [
                {
                    "keyframe_id": "keyframe-00001",
                    "environment": {"category": "unknown"},
                    "objects": [{"label": "ignored"}],
                    "interactions": [],
                }
            ]
        },
    )
    assert result["metrics"]["objects"]["evaluated_keyframes"] == 0
    assert result["metrics"]["objects"]["f1"] == 1.0


if __name__ == "__main__":
    test_manual_annotations_score_each_visual_dimension()
    test_omitted_dimensions_are_not_scored()
    print("PASS: manual vision annotations and dimension metrics")
