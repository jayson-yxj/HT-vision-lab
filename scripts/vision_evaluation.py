from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from .face_tracks import validate_output
from .models import file_sha256
from .scene_context import validate_scene_context
from .scene_semantics import (
    ENVIRONMENT_CATEGORIES,
    INTERACTION_PREDICATES,
    validate_scene_semantics,
)


SPATIAL_PREDICATES = (
    "co_visible_with",
    "left_of",
    "above",
    "visually_close_to",
    "overlaps_in_image",
)
SYMMETRIC_PREDICATES = {
    "co_visible_with",
    "visually_close_to",
    "overlaps_in_image",
    "sitting_with",
    "standing_with",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _label(value: str) -> str:
    return " ".join(value.strip().casefold().replace("_", " ").split())


def _relation(subject: str, predicate: str, object_ref: Optional[str]) -> Tuple[str, str, str]:
    target = object_ref or ""
    if predicate in SYMMETRIC_PREDICATES and target < subject:
        subject, target = target, subject
    return subject, predicate, target


def validate_evaluation_annotations(data: dict) -> list[str]:
    errors = []
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if data.get("context_type") != "vision_evaluation_annotations":
        errors.append("context_type must be vision_evaluation_annotations")
    if not data.get("sample_id"):
        errors.append("sample_id is required")
    source = data.get("source")
    if not isinstance(source, dict):
        errors.append("source is required")
    else:
        digest = source.get("video_sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            errors.append("source.video_sha256 must be a SHA-256 digest")
        if not isinstance(source.get("duration_ms"), int) or source.get("duration_ms", -1) < 0:
            errors.append("source.duration_ms must be a non-negative integer")
    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance is required")
    elif provenance.get("method") != "manual_evidence_review":
        errors.append("provenance.method must be manual_evidence_review")
    else:
        reviewed_at = provenance.get("reviewed_at")
        try:
            datetime.strptime(reviewed_at, "%Y-%m-%d")
        except (TypeError, ValueError):
            errors.append("provenance.reviewed_at must use YYYY-MM-DD")
        if not isinstance(provenance.get("notes"), str):
            errors.append("provenance.notes must be a string")

    bindings = data.get("identity_bindings")
    if not isinstance(bindings, list):
        errors.append("identity_bindings must be an array")
        bindings = []
    speakers, faces = set(), set()
    for index, binding in enumerate(bindings):
        if not isinstance(binding, dict):
            errors.append(f"identity_bindings[{index}] must be an object")
            continue
        speaker = binding.get("speaker_label")
        face = binding.get("face_id")
        if speaker not in {"A", "B", "C", "D"}:
            errors.append(f"identity_bindings[{index}] has an invalid speaker label")
        if not isinstance(face, str) or re.fullmatch(r"Face-[0-9]{2}", face) is None:
            errors.append(f"identity_bindings[{index}] has an invalid face ID")
        if speaker in speakers:
            errors.append(f"speaker {speaker} is annotated more than once")
        if face in faces:
            errors.append(f"face {face} is annotated more than once")
        speakers.add(speaker)
        faces.add(face)

    object_vocabulary = data.get("object_vocabulary")
    if object_vocabulary is not None:
        if not isinstance(object_vocabulary, list) or any(
            not isinstance(item, str) or not item.strip() for item in object_vocabulary
        ):
            errors.append("object_vocabulary must contain non-empty strings")
            object_vocabulary = []
        elif len({_label(item) for item in object_vocabulary}) != len(object_vocabulary):
            errors.append("object_vocabulary contains duplicate labels")
    interaction_predicates = data.get("interaction_predicates")
    if interaction_predicates is not None:
        if not isinstance(interaction_predicates, list) or any(
            item not in INTERACTION_PREDICATES for item in interaction_predicates
        ):
            errors.append("interaction_predicates contains an invalid predicate")
            interaction_predicates = []
        elif len(set(interaction_predicates)) != len(interaction_predicates):
            errors.append("interaction_predicates contains duplicates")

    keyframes = data.get("keyframes")
    if not isinstance(keyframes, list) or not keyframes:
        errors.append("keyframes must be a non-empty array")
        keyframes = []
    identifiers = set()
    for index, frame in enumerate(keyframes):
        if not isinstance(frame, dict):
            errors.append(f"keyframes[{index}] must be an object")
            continue
        keyframe_id = frame.get("keyframe_id")
        if not isinstance(keyframe_id, str) or re.fullmatch(r"keyframe-[0-9]{5}", keyframe_id) is None:
            errors.append(f"keyframes[{index}] has an invalid keyframe ID")
        if keyframe_id in identifiers:
            errors.append(f"{keyframe_id} is annotated more than once")
        identifiers.add(keyframe_id)
        if not isinstance(frame.get("timestamp_ms"), int) or frame.get("timestamp_ms", -1) < 0:
            errors.append(f"{keyframe_id} has an invalid timestamp")
        evaluated = {
            "environment_category",
            "object_labels",
            "interactions",
            "spatial_relations",
        } & frame.keys()
        if not evaluated:
            errors.append(f"{keyframe_id} has no annotated evaluation dimension")
        category = frame.get("environment_category")
        if "environment_category" in frame and category not in ENVIRONMENT_CATEGORIES:
            errors.append(f"{keyframe_id} has an invalid environment category")
        labels = frame.get("object_labels")
        if labels is not None:
            if not isinstance(labels, list) or any(not isinstance(item, str) or not item.strip() for item in labels):
                errors.append(f"{keyframe_id} has invalid object labels")
            elif len({_label(item) for item in labels}) != len(labels):
                errors.append(f"{keyframe_id} has duplicate object labels")
            elif object_vocabulary is not None and not {
                _label(item) for item in labels
            }.issubset({_label(item) for item in object_vocabulary}):
                errors.append(f"{keyframe_id} contains an object outside the vocabulary")
        for field, predicates in (
            ("interactions", INTERACTION_PREDICATES),
            ("spatial_relations", SPATIAL_PREDICATES),
        ):
            values = frame.get(field)
            if values is None:
                continue
            if not isinstance(values, list):
                errors.append(f"{keyframe_id}.{field} must be an array")
                continue
            seen = set()
            for relation_index, item in enumerate(values):
                if not isinstance(item, dict):
                    errors.append(f"{keyframe_id}.{field}[{relation_index}] must be an object")
                    continue
                subject = item.get("subject_ref")
                predicate = item.get("predicate")
                object_ref = item.get("object_ref")
                if not isinstance(subject, str) or not subject:
                    errors.append(f"{keyframe_id}.{field}[{relation_index}] has no subject")
                if predicate not in predicates:
                    errors.append(f"{keyframe_id}.{field}[{relation_index}] has an invalid predicate")
                if (
                    field == "interactions"
                    and interaction_predicates is not None
                    and predicate not in interaction_predicates
                ):
                    errors.append(
                        f"{keyframe_id}.{field}[{relation_index}] is outside the evaluation predicates"
                    )
                if object_ref is not None and not isinstance(object_ref, str):
                    errors.append(f"{keyframe_id}.{field}[{relation_index}] has an invalid object")
                normalized = _relation(str(subject), str(predicate), object_ref)
                if normalized in seen:
                    errors.append(f"{keyframe_id}.{field} contains a duplicate relation")
                seen.add(normalized)
    return errors


def _set_metric(
    expected: dict[str, set], predicted: dict[str, set], serializer=lambda item: item
) -> dict:
    true_positive = false_positive = false_negative = 0
    details = []
    for keyframe_id, expected_items in expected.items():
        predicted_items = predicted.get(keyframe_id, set())
        matched = expected_items & predicted_items
        missing = expected_items - predicted_items
        unexpected = predicted_items - expected_items
        true_positive += len(matched)
        false_positive += len(unexpected)
        false_negative += len(missing)
        if missing or unexpected:
            details.append(
                {
                    "keyframe_id": keyframe_id,
                    "missing": [serializer(item) for item in sorted(missing)],
                    "unexpected": [serializer(item) for item in sorted(unexpected)],
                }
            )
    predicted_count = true_positive + false_positive
    expected_count = true_positive + false_negative
    precision = true_positive / predicted_count if predicted_count else float(expected_count == 0)
    recall = true_positive / expected_count if expected_count else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "evaluated_keyframes": len(expected),
        "expected_items": expected_count,
        "predicted_items": predicted_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "details": details,
    }


def _serialize_relation(item: Tuple[str, str, str]) -> dict:
    return {"subject_ref": item[0], "predicate": item[1], "object_ref": item[2] or None}


def score_vision_predictions(
    annotations: dict, visual: dict, scene: dict, semantics: dict
) -> dict:
    annotation_errors = validate_evaluation_annotations(annotations)
    if annotation_errors:
        raise ValueError("invalid evaluation annotations: " + "; ".join(annotation_errors))

    annotated_frames = {item["keyframe_id"]: item for item in annotations["keyframes"]}
    scene_frames = {item["keyframe_id"]: item for item in scene["keyframes"]}
    semantic_frames = {item["keyframe_id"]: item for item in semantics["analyses"]}
    for keyframe_id, annotated in annotated_frames.items():
        actual = scene_frames.get(keyframe_id)
        if actual is None:
            raise ValueError(f"annotated keyframe is missing from scene context: {keyframe_id}")
        if actual["timestamp_ms"] != annotated["timestamp_ms"]:
            raise ValueError(f"annotated timestamp does not match scene context: {keyframe_id}")

    associations = {
        item["speaker_label"]: item for item in visual.get("speaker_face_associations", [])
    }
    identity_details = []
    identity_correct = 0
    for binding in annotations["identity_bindings"]:
        association = associations.get(binding["speaker_label"])
        predicted_face = (
            association.get("face_id")
            if association is not None and association.get("status") == "confirmed"
            else None
        )
        correct = predicted_face == binding["face_id"]
        identity_correct += int(correct)
        identity_details.append(
            {
                "speaker_label": binding["speaker_label"],
                "expected_face_id": binding["face_id"],
                "predicted_face_id": predicted_face,
                "predicted_status": association.get("status") if association else "missing",
                "correct": correct,
            }
        )
    identity_total = len(identity_details)
    identity_accuracy = identity_correct / identity_total if identity_total else 1.0

    environment_details = []
    environment_correct = 0
    environment_total = 0
    for keyframe_id, annotated in annotated_frames.items():
        if "environment_category" not in annotated:
            continue
        predicted = semantic_frames.get(keyframe_id, {}).get("environment", {}).get("category")
        correct = predicted == annotated["environment_category"]
        environment_correct += int(correct)
        environment_total += 1
        if not correct:
            environment_details.append(
                {
                    "keyframe_id": keyframe_id,
                    "expected": annotated["environment_category"],
                    "predicted": predicted,
                }
            )
    environment_accuracy = environment_correct / environment_total if environment_total else 1.0

    expected_objects, predicted_objects = {}, {}
    expected_interactions, predicted_interactions = {}, {}
    expected_spatial, predicted_spatial = {}, {}
    object_vocabulary = (
        {_label(item) for item in annotations["object_vocabulary"]}
        if "object_vocabulary" in annotations
        else None
    )
    interaction_predicates = (
        set(annotations["interaction_predicates"])
        if "interaction_predicates" in annotations
        else None
    )
    for keyframe_id, annotated in annotated_frames.items():
        semantic = semantic_frames.get(keyframe_id, {})
        if "object_labels" in annotated:
            expected_objects[keyframe_id] = {_label(item) for item in annotated["object_labels"]}
            predicted_objects[keyframe_id] = {
                _label(item["label"])
                for item in semantic.get("objects", [])
                if object_vocabulary is None or _label(item["label"]) in object_vocabulary
            }
        if "interactions" in annotated:
            expected_interactions[keyframe_id] = {
                _relation(item["subject_ref"], item["predicate"], item.get("object_ref"))
                for item in annotated["interactions"]
            }
            predicted_interactions[keyframe_id] = {
                _relation(item["subject_ref"], item["predicate"], item.get("object_ref"))
                for item in semantic.get("interactions", [])
                if interaction_predicates is None or item["predicate"] in interaction_predicates
            }

    person_states = {item["person_state_id"]: item for item in scene["person_states"]}
    spatial_by_frame: dict[str, set] = {}
    for item in scene["spatial_relations"]:
        subject = person_states[item["subject_state_id"]]["face_id"]
        object_ref = person_states[item["object_state_id"]]["face_id"]
        spatial_by_frame.setdefault(item["keyframe_id"], set()).add(
            _relation(subject, item["predicate"], object_ref)
        )
    for keyframe_id, annotated in annotated_frames.items():
        if "spatial_relations" not in annotated:
            continue
        expected_spatial[keyframe_id] = {
            _relation(item["subject_ref"], item["predicate"], item.get("object_ref"))
            for item in annotated["spatial_relations"]
        }
        predicted_spatial[keyframe_id] = spatial_by_frame.get(keyframe_id, set())

    metrics = {
        "identity_binding": {
            "evaluated_bindings": identity_total,
            "correct": identity_correct,
            "accuracy": round(identity_accuracy, 6),
            "details": identity_details,
        },
        "environment": {
            "evaluated_keyframes": environment_total,
            "correct": environment_correct,
            "accuracy": round(environment_accuracy, 6),
            "details": environment_details,
        },
        "objects": _set_metric(expected_objects, predicted_objects),
        "interactions": _set_metric(
            expected_interactions, predicted_interactions, _serialize_relation
        ),
        "spatial_relations": _set_metric(
            expected_spatial, predicted_spatial, _serialize_relation
        ),
    }
    metrics["objects"]["vocabulary"] = (
        sorted(object_vocabulary) if object_vocabulary is not None else None
    )
    metrics["interactions"]["evaluated_predicates"] = (
        sorted(interaction_predicates) if interaction_predicates is not None else None
    )
    scores = []
    if identity_total:
        scores.append(metrics["identity_binding"]["accuracy"])
    if environment_total:
        scores.append(metrics["environment"]["accuracy"])
    for name in ("objects", "interactions", "spatial_relations"):
        if metrics[name]["evaluated_keyframes"]:
            scores.append(metrics[name]["f1"])
    return {
        "metrics": metrics,
        "summary": {
            "evaluated_dimensions": len(scores),
            "macro_score": round(sum(scores) / len(scores), 6),
        },
    }


def evaluate_vision(
    annotation_path: Path,
    scene_path: Path,
    visual_path: Optional[Path] = None,
    semantics_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> dict:
    annotation_path = annotation_path.expanduser().resolve()
    scene_path = scene_path.expanduser().resolve()
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    scene = json.loads(scene_path.read_text(encoding="utf-8"))

    scene_errors = validate_scene_context(scene)
    if scene_errors:
        raise ValueError("invalid scene context: " + "; ".join(scene_errors))
    visual_path = (
        visual_path.expanduser().resolve()
        if visual_path
        else Path(scene["source"]["visual_tracks_path"]).expanduser().resolve()
    )
    semantics_path = (
        semantics_path.expanduser().resolve()
        if semantics_path
        else scene_path.parent / "scene_semantics.json"
    )
    visual = json.loads(visual_path.read_text(encoding="utf-8"))
    semantics = json.loads(semantics_path.read_text(encoding="utf-8"))
    visual_errors = validate_output(visual)
    if visual_errors:
        raise ValueError("invalid visual tracks: " + "; ".join(visual_errors))
    semantic_errors = validate_scene_semantics(semantics)
    if semantic_errors:
        raise ValueError("invalid scene semantics: " + "; ".join(semantic_errors))

    source = annotations.get("source", {})
    if source.get("video_sha256") != scene["source"]["video_sha256"]:
        raise ValueError("annotation video SHA-256 does not match scene context")
    if source.get("duration_ms") != scene["source"]["duration_ms"]:
        raise ValueError("annotation duration does not match scene context")
    if visual["session_id"] != scene["session_id"] or semantics["session_id"] != scene["session_id"]:
        raise ValueError("prediction files do not belong to the same session")
    if file_sha256(visual_path) != scene["source"]["visual_tracks_sha256"]:
        raise ValueError("visual tracks have changed since the scene context was created")
    if semantics["source"]["scene_context_sha256"] != file_sha256(scene_path):
        raise ValueError("scene semantics do not reference this scene context")

    scored = score_vision_predictions(annotations, visual, scene, semantics)
    report = {
        "schema_version": 1,
        "context_type": "vision_evaluation_report",
        "sample_id": annotations["sample_id"],
        "created_at": _now(),
        "source": {
            "annotations_path": str(annotation_path),
            "annotations_sha256": file_sha256(annotation_path),
            "visual_tracks_path": str(visual_path),
            "visual_tracks_sha256": file_sha256(visual_path),
            "scene_context_path": str(scene_path),
            "scene_context_sha256": file_sha256(scene_path),
            "scene_semantics_path": str(semantics_path),
            "scene_semantics_sha256": file_sha256(semantics_path),
        },
        **scored,
    }
    output_path = (
        output_path.expanduser().resolve()
        if output_path
        else scene_path.parent / "vision_evaluation_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
