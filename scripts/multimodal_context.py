from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import file_sha256


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _visual_entity_ref(visual_session_id: str, face_id: str) -> str:
    return f"{visual_session_id}:face:{face_id}"


def _validate_upstream(context: dict) -> None:
    required = {
        "schema_version",
        "session_id",
        "source",
        "policy",
        "participants",
        "observations",
        "evidence",
        "associations",
        "extraction",
        "stats",
    }
    missing = required - set(context)
    if missing:
        raise ValueError("participant context is missing: " + ", ".join(sorted(missing)))
    if context["schema_version"] != 1:
        raise ValueError("only participant context schema_version=1 is supported")
    participants = context["participants"]
    if not isinstance(participants, list):
        raise ValueError("participant context participants must be an array")
    participant_ids = [item.get("participant_id") for item in participants]
    speaker_labels = [item.get("speaker_label") for item in participants]
    if None in participant_ids or len(participant_ids) != len(set(participant_ids)):
        raise ValueError("participant context contains missing or duplicate participant IDs")
    if None in speaker_labels or len(speaker_labels) != len(set(speaker_labels)):
        raise ValueError("participant context contains missing or duplicate speaker labels")


def _join_text(previous: str, current: str) -> str:
    if not previous:
        return current.strip()
    if (
        previous[-1:].isascii()
        and previous[-1:].isalnum()
        and current[:1].isascii()
        and current[:1].isalnum()
    ):
        return previous.rstrip() + " " + current.lstrip()
    return previous.rstrip() + current.lstrip()


def _normalized_text(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _verify_context_timeline(context: dict, visual: dict) -> None:
    binding = visual.get("processing", {}).get("speaker_face_binding")
    if not isinstance(binding, dict):
        raise ValueError("visual tracks are missing speaker-face binding provenance")
    timeline_path = Path(binding.get("speech_timeline_path", "")).expanduser().resolve()
    expected_hash = binding.get("speech_timeline_sha256")
    if not timeline_path.is_file() or file_sha256(timeline_path) != expected_hash:
        raise ValueError("speaker-face binding speech timeline is unavailable or has changed")
    with timeline_path.open(encoding="utf-8") as handle:
        timeline = json.load(handle)
    if isinstance(timeline, dict):
        for key in ("speech_spans", "utterances", "segments"):
            if isinstance(timeline.get(key), list):
                timeline = timeline[key]
                break
    if not isinstance(timeline, list):
        raise ValueError("speaker-face binding speech timeline has an invalid format")
    duration_ms = visual.get("source", {}).get("duration_ms")
    if not isinstance(duration_ms, int) or duration_ms <= 0:
        raise ValueError("visual tracks are missing a valid source duration")
    offset_ms = int(binding.get("timeline_offset_ms", 0))
    visible_speech_start = -offset_ms / 1000
    visible_speech_end = (duration_ms - offset_ms) / 1000

    spans = []
    for item in timeline:
        if not isinstance(item, dict) or item.get("speaker") is None:
            continue
        if item.get("attribution", "single_speaker") != "single_speaker":
            continue
        if item.get("status", "final") != "final":
            continue
        start = item.get("start_s", item.get("start"))
        end = item.get("end_s", item.get("end"))
        text = item.get("text")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if not isinstance(text, str):
            text = ""
        spans.append((item["speaker"], float(start), float(end), text))

    verified = 0
    for evidence in context["evidence"]:
        if evidence.get("source_type") != "completed_turn":
            continue
        speaker = evidence.get("speaker_label")
        start, end = evidence.get("start_s"), evidence.get("end_s")
        if speaker is None or start is None or end is None:
            continue
        if float(end) <= visible_speech_start or float(start) >= visible_speech_end:
            continue
        matching = [
            item
            for item in spans
            if item[0] == speaker and item[2] > float(start) and item[1] < float(end)
        ]
        matching.sort(key=lambda item: (item[1], item[2]))
        if not matching:
            raise ValueError(f"participant evidence {evidence['evidence_id']} is absent from the binding timeline")
        combined = ""
        for item in matching:
            combined = _join_text(combined, item[3])
        same_bounds = matching[0][1] <= float(start) + 0.25 and matching[-1][2] >= float(end) - 0.25
        combined_text = _normalized_text(combined)
        evidence_text = _normalized_text(evidence["content"])
        same_text = bool(evidence_text) and evidence_text in combined_text
        if not same_bounds or not same_text:
            raise ValueError(
                f"participant evidence {evidence['evidence_id']} does not match the binding timeline"
            )
        verified += 1
    if not verified:
        raise ValueError("participant context has no completed-turn evidence to verify against the binding timeline")


def _binding_state(binding_status: str, primary: bool) -> str:
    if primary and binding_status == "confirmed":
        return "confirmed"
    if binding_status == "ambiguous":
        return "disputed"
    return "candidate"


def _evidence_record(
    session_id: str,
    participant_id: str,
    speaker_label: str,
    visual_entity_ref: str,
    binding_status: str,
    overlap: dict,
) -> dict:
    payload = {
        "type": "speaker_face_temporal_overlap",
        "speaker_label": speaker_label,
        "visual_entity_ref": visual_entity_ref,
        "binding_status": binding_status,
        "active_speaker_segment_id": overlap["active_speaker_segment_id"],
        "start_ms": overlap["start_ms"],
        "end_ms": overlap["end_ms"],
        "duration_ms": overlap["duration_ms"],
    }
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    identity = "\x1f".join(
        (
            participant_id,
            visual_entity_ref,
            overlap["active_speaker_segment_id"],
            str(overlap["start_ms"]),
            str(overlap["end_ms"]),
        )
    )
    return {
        "evidence_id": f"{session_id}:evidence:multimodal:{_sha256_text(identity)[:16]}",
        "participant_id": participant_id,
        "modality": "multimodal",
        "source_type": "video_track",
        "source_id": overlap["active_speaker_segment_id"],
        "speaker_label": speaker_label,
        "start_s": overlap["start_ms"] / 1000,
        "end_s": overlap["end_ms"] / 1000,
        "content": content,
        "content_sha256": _sha256_text(content),
        "contribution_ids": list(
            dict.fromkeys(overlap["speech_span_ids"] + [overlap["active_speaker_segment_id"]])
        ),
    }


def _visibility_evidence_record(session_id: str, participant_id: str, segment: dict) -> dict:
    payload = {
        "type": "speaker_visibility_state",
        "speaker_label": segment["speaker_label"],
        "face_id": segment["face_id"],
        "state": segment["state"],
        "reason": segment["reason"],
        "start_ms": segment["start_ms"],
        "end_ms": segment["end_ms"],
    }
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    identity = "\x1f".join((participant_id, "visibility", segment["visibility_id"]))
    return {
        "evidence_id": f"{session_id}:evidence:multimodal:{_sha256_text(identity)[:16]}",
        "participant_id": participant_id,
        "modality": "multimodal",
        "source_type": "video_track",
        "source_id": segment["visibility_id"],
        "speaker_label": segment["speaker_label"],
        "start_s": segment["start_ms"] / 1000,
        "end_s": segment["end_ms"] / 1000,
        "content": content,
        "content_sha256": _sha256_text(content),
        "contribution_ids": list(
            dict.fromkeys(
                segment["source_speech_span_ids"]
                + segment["tracklet_ids"]
                + [segment["visibility_id"]]
            )
        ),
    }


def project_multimodal_context(
    visual_path: Path,
    participant_context_path: Path,
    output_path: Optional[Path] = None,
) -> dict:
    visual_path = visual_path.expanduser().resolve()
    participant_context_path = participant_context_path.expanduser().resolve()
    output_path = (
        output_path.expanduser().resolve()
        if output_path
        else visual_path.parent / "multimodal_participant_context.json"
    )
    if output_path in {visual_path, participant_context_path}:
        raise ValueError("multimodal output must not overwrite either source file")
    with visual_path.open(encoding="utf-8") as handle:
        visual = json.load(handle)
    with participant_context_path.open(encoding="utf-8") as handle:
        upstream = json.load(handle)
    _validate_upstream(upstream)
    _verify_context_timeline(upstream, visual)
    bindings = visual.get("speaker_face_associations", [])
    if not bindings:
        raise RuntimeError("visual tracks contain no speaker-face bindings; run ./lab bind-speakers first")

    context = copy.deepcopy(upstream)
    context["context_type"] = "multimodal_participant_context"
    context["created_at"] = datetime.now(timezone.utc).isoformat()
    context["source"] = {
        "participant_context": {
            "path": str(participant_context_path),
            "sha256": file_sha256(participant_context_path),
        },
        "visual_tracks": {"path": str(visual_path), "sha256": file_sha256(visual_path)},
        "upstream": copy.deepcopy(upstream["source"]),
    }
    context["policy"] = copy.deepcopy(upstream["policy"])
    context["policy"].update(
        {
            "visual_identity_scope": "current_session",
            "confirmed_bindings_are_identity_links": True,
            "ambiguous_bindings_remain_disputed": True,
        }
    )

    visual_session_id = str(visual["session_id"])
    context["visual_entities"] = [
        {
            "visual_entity_ref": _visual_entity_ref(visual_session_id, face["face_id"]),
            "source_face_id": face["face_id"],
            "first_seen_s": face["first_seen_ms"] / 1000,
            "last_seen_s": face["last_seen_ms"] / 1000,
            "visible_duration_s": face["visible_duration_ms"] / 1000,
            "observation_count": face["observation_count"],
            "evidence_image": face["evidence_image"],
        }
        for face in visual["faces"]
    ]
    visual_refs = {item["source_face_id"]: item["visual_entity_ref"] for item in context["visual_entities"]}
    participants = {item["speaker_label"]: item for item in context["participants"]}
    evidence_by_id = {item["evidence_id"]: item for item in context["evidence"]}
    associations = list(context["associations"])
    existing_association_ids = {item["association_id"] for item in associations}
    minimum_evidence = int(
        visual.get("processing", {}).get("speaker_face_binding", {}).get("min_evidence_ms", 1000)
    )
    added_evidence_ids = set()
    added_associations = []

    for binding in bindings:
        speaker = binding["speaker_label"]
        participant = participants.get(speaker)
        if participant is None:
            raise ValueError(f"visual binding speaker {speaker} has no participant")
        primary_face = binding.get("face_id")
        for candidate in binding.get("candidate_faces", []):
            face_id = candidate["face_id"]
            primary = face_id == primary_face
            if not primary and candidate["evidence_duration_ms"] < minimum_evidence:
                continue
            visual_ref = visual_refs.get(face_id)
            if visual_ref is None:
                raise ValueError(f"visual binding references unknown face {face_id}")
            evidence_ids = []
            for overlap in candidate["evidence"]:
                evidence = _evidence_record(
                    context["session_id"],
                    participant["participant_id"],
                    speaker,
                    visual_ref,
                    binding["status"],
                    overlap,
                )
                existing = evidence_by_id.get(evidence["evidence_id"])
                if existing is not None and existing != evidence:
                    raise ValueError(f"evidence ID collision: {evidence['evidence_id']}")
                if existing is None:
                    context["evidence"].append(evidence)
                    evidence_by_id[evidence["evidence_id"]] = evidence
                    added_evidence_ids.add(evidence["evidence_id"])
                evidence_ids.append(evidence["evidence_id"])
                if evidence["evidence_id"] not in participant["evidence_ids"]:
                    participant["evidence_ids"].append(evidence["evidence_id"])

            association_id = (
                f"{context['session_id']}:association:speaker-face:{speaker}:{face_id}"
            )
            if association_id in existing_association_ids:
                raise ValueError(f"association ID collision: {association_id}")
            state = _binding_state(binding["status"], primary)
            confidence = binding["confidence"] if primary else candidate["speaker_coverage"]
            association = {
                "association_id": association_id,
                "left_ref": participant["participant_id"],
                "right_ref": visual_ref,
                "relation": "same_session_identity",
                "confidence": round(float(confidence), 6),
                "state": state,
                "evidence_ids": evidence_ids,
                "source_binding_status": binding["status"],
                "speaker_coverage": candidate["speaker_coverage"],
                "evidence_duration_ms": candidate["evidence_duration_ms"],
            }
            associations.append(association)
            added_associations.append(association)
            existing_association_ids.add(association_id)

    context["associations"] = associations
    context["visibility_events"] = []
    for segment in visual.get("speaker_visibility_segments", []):
        participant = participants.get(segment["speaker_label"])
        if participant is None:
            raise ValueError(
                f"visibility segment speaker {segment['speaker_label']} has no participant"
            )
        visual_ref = visual_refs.get(segment["face_id"]) if segment["face_id"] else None
        evidence = _visibility_evidence_record(
            context["session_id"], participant["participant_id"], segment
        )
        if evidence["evidence_id"] in evidence_by_id:
            raise ValueError(f"evidence ID collision: {evidence['evidence_id']}")
        context["evidence"].append(evidence)
        evidence_by_id[evidence["evidence_id"]] = evidence
        added_evidence_ids.add(evidence["evidence_id"])
        participant["evidence_ids"].append(evidence["evidence_id"])
        context["visibility_events"].append(
            {
                "visibility_event_id": f"{context['session_id']}:visibility:{segment['visibility_id']}",
                "participant_id": participant["participant_id"],
                "speaker_label": segment["speaker_label"],
                "visual_entity_ref": visual_ref,
                "state": segment["state"],
                "confidence": segment["confidence"],
                "reason": segment["reason"],
                "start_s": segment["start_ms"] / 1000,
                "end_s": segment["end_ms"] / 1000,
                "evidence_id": evidence["evidence_id"],
                "source_visibility_id": segment["visibility_id"],
            }
        )
    context["stats"] = copy.deepcopy(upstream["stats"])
    context["stats"].update(
        {
            "participants": len(context["participants"]),
            "observations": len(context["observations"]),
            "evidence": len(context["evidence"]),
            "associations": len(context["associations"]),
            "visual_entities": len(context["visual_entities"]),
            "multimodal_evidence": len(added_evidence_ids),
            "visual_associations": len(added_associations),
            "confirmed_visual_associations": sum(
                item["state"] == "confirmed" for item in added_associations
            ),
            "disputed_visual_associations": sum(
                item["state"] == "disputed" for item in added_associations
            ),
            "visibility_events": len(context["visibility_events"]),
        }
    )
    errors = validate_multimodal_context(context)
    if errors:
        raise RuntimeError("multimodal context validation failed: " + "; ".join(errors))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(context, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output_path)
    return context


def validate_multimodal_context(context: dict) -> List[str]:
    errors: List[str] = []
    required = {
        "schema_version",
        "context_type",
        "session_id",
        "created_at",
        "source",
        "policy",
        "participants",
        "visual_entities",
        "visibility_events",
        "observations",
        "evidence",
        "associations",
        "extraction",
        "stats",
    }
    missing = required - set(context)
    if missing:
        return ["missing top-level fields: " + ", ".join(sorted(missing))]
    for source_name in ("participant_context", "visual_tracks"):
        source = context.get("source", {}).get(source_name, {})
        path = Path(source.get("path", "")).expanduser()
        if not path.is_file():
            errors.append(f"{source_name} source file is unavailable")
        elif file_sha256(path) != source.get("sha256"):
            errors.append(f"{source_name} source SHA-256 mismatch")
    participant_ids = [item["participant_id"] for item in context["participants"]]
    speaker_labels = [item["speaker_label"] for item in context["participants"]]
    visual_refs = [item["visual_entity_ref"] for item in context["visual_entities"]]
    evidence_ids = [item["evidence_id"] for item in context["evidence"]]
    observation_ids = [item["observation_id"] for item in context["observations"]]
    association_ids = [item["association_id"] for item in context["associations"]]
    visibility_event_ids = [item["visibility_event_id"] for item in context["visibility_events"]]
    for label, values in (
        ("participant", participant_ids),
        ("speaker", speaker_labels),
        ("visual entity", visual_refs),
        ("evidence", evidence_ids),
        ("observation", observation_ids),
        ("association", association_ids),
        ("visibility event", visibility_event_ids),
    ):
        if len(values) != len(set(values)):
            errors.append(f"duplicate {label} IDs")
    participant_set = set(participant_ids)
    visual_set = set(visual_refs)
    evidence_set = set(evidence_ids)
    observation_set = set(observation_ids)
    for participant in context["participants"]:
        if not set(participant["observation_ids"]).issubset(observation_set):
            errors.append(f"{participant['participant_id']} references unknown observations")
        if not set(participant["evidence_ids"]).issubset(evidence_set):
            errors.append(f"{participant['participant_id']} references unknown evidence")
    for observation in context["observations"]:
        if observation["participant_id"] not in participant_set:
            errors.append(f"{observation['observation_id']} references unknown participant")
        referenced = set(observation["evidence_ids"] + observation["context_evidence_ids"])
        if not referenced.issubset(evidence_set):
            errors.append(f"{observation['observation_id']} references unknown evidence")
    for evidence in context["evidence"]:
        if evidence["participant_id"] not in participant_set:
            errors.append(f"{evidence['evidence_id']} references unknown participant")
        if _sha256_text(evidence["content"]) != evidence["content_sha256"]:
            errors.append(f"{evidence['evidence_id']} content hash mismatch")
        if evidence["start_s"] is not None and evidence["end_s"] is not None:
            if evidence["start_s"] > evidence["end_s"]:
                errors.append(f"{evidence['evidence_id']} has reversed time")
    confirmed_left = []
    confirmed_right = []
    for association in context["associations"]:
        if not set(association["evidence_ids"]).issubset(evidence_set):
            errors.append(f"{association['association_id']} references unknown evidence")
        if association["relation"] != "same_session_identity":
            continue
        if association["left_ref"] not in participant_set:
            errors.append(f"{association['association_id']} references unknown participant")
        if association["right_ref"] not in visual_set:
            errors.append(f"{association['association_id']} references unknown visual entity")
        if association["state"] == "confirmed":
            confirmed_left.append(association["left_ref"])
            confirmed_right.append(association["right_ref"])
    if len(confirmed_left) != len(set(confirmed_left)):
        errors.append("a participant has more than one confirmed visual identity")
    if len(confirmed_right) != len(set(confirmed_right)):
        errors.append("a visual identity is confirmed for more than one participant")
    for event in context["visibility_events"]:
        if event["participant_id"] not in participant_set:
            errors.append(f"{event['visibility_event_id']} references unknown participant")
        if event["visual_entity_ref"] is not None and event["visual_entity_ref"] not in visual_set:
            errors.append(f"{event['visibility_event_id']} references unknown visual entity")
        if event["evidence_id"] not in evidence_set:
            errors.append(f"{event['visibility_event_id']} references unknown evidence")
        if event["start_s"] >= event["end_s"]:
            errors.append(f"{event['visibility_event_id']} has invalid time range")
    return errors
