from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .models import file_sha256
from .speaker_face_binding import (
    CANONICAL_SPEAKERS,
    _interval_duration,
    _time_value,
    bind_speakers_to_faces,
)


ALGORITHM = "confirmed_face_anchor_span_relabel_v1"


def _visual_evidence_sha256(visual: dict) -> str:
    processing = visual.get("processing", {})
    evidence = {
        "session_id": visual.get("session_id"),
        "source": visual.get("source"),
        "faces": visual.get("faces", []),
        "tracklets": visual.get("tracklets", []),
        "active_speaker_segments": visual.get("active_speaker_segments", []),
        "active_speaker": processing.get("active_speaker"),
        "active_speaker_parameters": processing.get("active_speaker_parameters"),
    }
    encoded = json.dumps(
        evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _speech_spans(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("speech_spans", "utterances", "segments"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(
        "speech timeline must be a JSON array or contain "
        "speech_spans/utterances/segments"
    )


def _span_face_evidence(
    start_ms: int,
    end_ms: int,
    active_segments: Sequence[dict],
) -> List[dict]:
    by_face: Dict[str, List[Tuple[int, int, str]]] = {}
    for segment in active_segments:
        face_id = segment.get("face_id")
        start = max(start_ms, int(segment["start_ms"]))
        end = min(end_ms, int(segment["end_ms"]))
        if face_id and end > start:
            by_face.setdefault(face_id, []).append(
                (start, end, str(segment["segment_id"]))
            )
    candidates = []
    for face_id, evidence in by_face.items():
        overlap = _interval_duration([(start, end) for start, end, _ in evidence])
        candidates.append(
            {
                "face_id": face_id,
                "overlap_ms": overlap,
                "active_speaker_segment_ids": list(
                    dict.fromkeys(identifier for _, _, identifier in evidence)
                ),
            }
        )
    return sorted(candidates, key=lambda item: (-item["overlap_ms"], item["face_id"]))


def reconcile_speakers(
    visual_path: Path,
    speech_path: Path,
    output_path: Optional[Path] = None,
    timeline_offset_ms: int = 0,
    min_anchor_confidence: float = 0.70,
    min_overlap_ms: int = 1200,
    min_span_coverage: float = 0.60,
    min_face_dominance: float = 0.60,
) -> dict:
    if min_overlap_ms < 0:
        raise ValueError("min_overlap_ms cannot be negative")
    for name, value in (
        ("min_anchor_confidence", min_anchor_confidence),
        ("min_span_coverage", min_span_coverage),
        ("min_face_dominance", min_face_dominance),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between zero and one")

    visual_path = visual_path.expanduser().resolve()
    speech_path = speech_path.expanduser().resolve()
    output_path = (
        output_path.expanduser().resolve()
        if output_path
        else visual_path.parent / "reconciled_speech_spans.json"
    )
    if output_path == speech_path:
        raise ValueError("reconciled output must not overwrite the source speech timeline")

    with visual_path.open(encoding="utf-8") as handle:
        visual = json.load(handle)
    active_segments = visual.get("active_speaker_segments", [])
    if not active_segments:
        raise RuntimeError("visual tracks contain no active-speaker segments")
    anchors = {
        item["face_id"]: item
        for item in visual.get("speaker_face_associations", [])
        if item.get("status") == "confirmed"
        and item.get("face_id")
        and float(item.get("confidence", 0)) >= min_anchor_confidence
    }
    if not anchors:
        raise RuntimeError("visual tracks contain no sufficiently confident speaker-face anchors")

    duration_ms = int(visual["source"]["duration_ms"])
    spans = [dict(item) for item in _speech_spans(speech_path)]
    corrections = []
    eligible = visually_evidenced = 0
    for index, span in enumerate(spans, start=1):
        span_id = str(span.get("id") or f"speech-interval-{index:06d}")
        span["id"] = span_id
        speaker = span.get("speaker")
        if (
            speaker not in CANONICAL_SPEAKERS
            or span.get("attribution", "single_speaker") != "single_speaker"
            or span.get("status", "final") != "final"
        ):
            continue
        start_ms = max(
            0,
            int(round(_time_value(span, "start_s", "start") * 1000))
            + timeline_offset_ms,
        )
        end_ms = min(
            duration_ms,
            int(round(_time_value(span, "end_s", "end") * 1000))
            + timeline_offset_ms,
        )
        if end_ms <= start_ms:
            continue
        eligible += 1
        candidates = _span_face_evidence(start_ms, end_ms, active_segments)
        if not candidates:
            continue
        visually_evidenced += 1
        top = candidates[0]
        runner_up_overlap = candidates[1]["overlap_ms"] if len(candidates) > 1 else 0
        coverage = top["overlap_ms"] / (end_ms - start_ms)
        dominance = (top["overlap_ms"] - runner_up_overlap) / top["overlap_ms"]
        anchor = anchors.get(top["face_id"])
        resolved = anchor.get("speaker_label") if anchor else None
        if (
            resolved
            and resolved != speaker
            and top["overlap_ms"] >= min_overlap_ms
            and coverage >= min_span_coverage
            and dominance >= min_face_dominance
        ):
            correction_id = f"speaker-correction-{len(corrections) + 1:04d}"
            span["original_speaker"] = speaker
            span["speaker"] = resolved
            span["speaker_reconciliation"] = {
                "correction_id": correction_id,
                "algorithm": ALGORITHM,
                "face_id": top["face_id"],
            }
            corrections.append(
                {
                    "correction_id": correction_id,
                    "speech_span_id": span_id,
                    "original_speaker": speaker,
                    "resolved_speaker": resolved,
                    "face_id": top["face_id"],
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "overlap_ms": top["overlap_ms"],
                    "span_coverage": round(coverage, 6),
                    "face_dominance": round(dominance, 6),
                    "anchor_confidence": anchor["confidence"],
                    "active_speaker_segment_ids": top[
                        "active_speaker_segment_ids"
                    ],
                    "candidate_faces": candidates,
                }
            )

    result = {
        "schema_version": 1,
        "context_type": "multimodal_speaker_timeline",
        "session_id": visual["session_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "visual_tracks": {
                "path": str(visual_path),
                "evidence_sha256": _visual_evidence_sha256(visual),
            },
            "speech_timeline": {
                "path": str(speech_path),
                "sha256": file_sha256(speech_path),
            },
        },
        "processing": {
            "algorithm": ALGORITHM,
            "timeline_offset_ms": timeline_offset_ms,
            "min_anchor_confidence": min_anchor_confidence,
            "min_overlap_ms": min_overlap_ms,
            "min_span_coverage": min_span_coverage,
            "min_face_dominance": min_face_dominance,
            "anchors": [
                {
                    "speaker_label": item["speaker_label"],
                    "face_id": face_id,
                    "confidence": item["confidence"],
                }
                for face_id, item in sorted(anchors.items())
            ],
        },
        "speech_spans": spans,
        "corrections": corrections,
        "statistics": {
            "input_spans": len(spans),
            "eligible_spans": eligible,
            "visually_evidenced_spans": visually_evidenced,
            "corrected_spans": len(corrections),
            "corrected_duration_ms": sum(
                item["end_ms"] - item["start_ms"] for item in corrections
            ),
        },
        "warnings": (
            [
                "Voice-derived participant context and dialogue memory still use the original "
                "speaker labels and must be regenerated from this reconciled timeline"
            ]
            if corrections
            else []
        ),
    }
    errors = validate_reconciled_timeline(result)
    if errors:
        raise RuntimeError("reconciled timeline validation failed: " + "; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output_path)
    return result


def reconcile_and_rebind(
    visual_path: Path,
    speech_path: Path,
    output_path: Optional[Path] = None,
    timeline_offset_ms: int = 0,
    min_anchor_confidence: float = 0.70,
    min_overlap_ms: int = 1200,
    min_span_coverage: float = 0.60,
    min_face_dominance: float = 0.60,
    max_passes: int = 4,
) -> Tuple[dict, dict, int]:
    if max_passes < 2:
        raise ValueError("max_passes must be at least two")
    visual_path = visual_path.expanduser().resolve()
    speech_path = speech_path.expanduser().resolve()
    output_path = (
        output_path.expanduser().resolve()
        if output_path
        else visual_path.parent / "reconciled_speech_spans.json"
    )
    original_visual = visual_path.read_bytes()
    previous_signature = None
    try:
        for pass_number in range(1, max_passes + 1):
            timeline = reconcile_speakers(
                visual_path,
                speech_path,
                output_path=output_path,
                timeline_offset_ms=timeline_offset_ms,
                min_anchor_confidence=min_anchor_confidence,
                min_overlap_ms=min_overlap_ms,
                min_span_coverage=min_span_coverage,
                min_face_dominance=min_face_dominance,
            )
            visual = bind_speakers_to_faces(
                visual_path,
                output_path,
                timeline_offset_ms=timeline_offset_ms,
            )
            signature = tuple(
                (
                    item["speech_span_id"],
                    item["original_speaker"],
                    item["resolved_speaker"],
                )
                for item in timeline["corrections"]
            )
            if signature == previous_signature:
                return timeline, visual, pass_number
            previous_signature = signature
    except Exception:
        visual_path.write_bytes(original_visual)
        raise
    timeline["warnings"].append(
        f"Speaker reconciliation did not stabilize within {max_passes} passes"
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(timeline, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return timeline, visual, max_passes


def validate_reconciled_timeline(data: dict) -> List[str]:
    errors = []
    required = {
        "schema_version",
        "context_type",
        "session_id",
        "created_at",
        "source",
        "processing",
        "speech_spans",
        "corrections",
        "statistics",
        "warnings",
    }
    missing = required - set(data)
    if missing:
        return ["missing top-level fields: " + ", ".join(sorted(missing))]
    span_ids = [str(item.get("id")) for item in data["speech_spans"]]
    if len(span_ids) != len(set(span_ids)):
        errors.append("duplicate speech span IDs")
    correction_ids = [item.get("correction_id") for item in data["corrections"]]
    if len(correction_ids) != len(set(correction_ids)):
        errors.append("duplicate speaker correction IDs")
    for item in data["corrections"]:
        if item.get("speech_span_id") not in span_ids:
            errors.append(f"{item.get('correction_id')} references unknown speech span")
        if item.get("original_speaker") == item.get("resolved_speaker"):
            errors.append(f"{item.get('correction_id')} does not change the speaker")
        if item.get("resolved_speaker") not in CANONICAL_SPEAKERS:
            errors.append(f"{item.get('correction_id')} has invalid resolved speaker")
        if int(item.get("end_ms", 0)) <= int(item.get("start_ms", 0)):
            errors.append(f"{item.get('correction_id')} has invalid time range")
    expected = {
        "input_spans": len(data["speech_spans"]),
        "eligible_spans": data["statistics"]["eligible_spans"],
        "visually_evidenced_spans": data["statistics"][
            "visually_evidenced_spans"
        ],
        "corrected_spans": len(data["corrections"]),
        "corrected_duration_ms": sum(
            item["end_ms"] - item["start_ms"] for item in data["corrections"]
        ),
    }
    if data["statistics"] != expected:
        errors.append("statistics do not match reconciled timeline contents")
    return errors
