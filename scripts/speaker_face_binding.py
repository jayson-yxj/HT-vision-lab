from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .models import file_sha256


CANONICAL_SPEAKERS = frozenset("ABCD")


@dataclass(frozen=True)
class SpeechInterval:
    start_ms: int
    end_ms: int
    source_ids: Tuple[str, ...]


def _time_value(item: dict, seconds_name: str, fallback_name: str) -> float:
    if seconds_name in item:
        return float(item[seconds_name])
    if fallback_name in item:
        return float(item[fallback_name])
    raise ValueError(f"speech interval is missing {seconds_name}/{fallback_name}")


def _merge_speech_intervals(items: Sequence[SpeechInterval]) -> List[SpeechInterval]:
    merged: List[SpeechInterval] = []
    for item in sorted(items, key=lambda value: (value.start_ms, value.end_ms)):
        if merged and item.start_ms <= merged[-1].end_ms:
            previous = merged[-1]
            merged[-1] = SpeechInterval(
                previous.start_ms,
                max(previous.end_ms, item.end_ms),
                tuple(dict.fromkeys(previous.source_ids + item.source_ids)),
            )
        else:
            merged.append(item)
    return merged


def _load_speech_timeline(path: Path, duration_ms: int, offset_ms: int) -> Dict[str, List[SpeechInterval]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        for key in ("speech_spans", "utterances", "segments"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError("speech timeline must be a JSON array or contain speech_spans/utterances/segments")

    grouped: Dict[str, List[SpeechInterval]] = {}
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError("speech timeline entries must be objects")
        speaker = item.get("speaker")
        if speaker is None or speaker == "unresolved":
            continue
        if speaker not in CANONICAL_SPEAKERS:
            raise ValueError(
                f"speech timeline contains non-canonical speaker {speaker!r}; use stabilized A/B/C/D output"
            )
        if item.get("attribution", "single_speaker") != "single_speaker":
            continue
        if item.get("status", "final") != "final":
            continue
        start_ms = int(round(_time_value(item, "start_s", "start") * 1000)) + offset_ms
        end_ms = int(round(_time_value(item, "end_s", "end") * 1000)) + offset_ms
        start_ms = max(0, start_ms)
        end_ms = min(duration_ms, end_ms)
        if end_ms <= start_ms:
            continue
        source_id = str(item.get("id") or f"speech-interval-{index:06d}")
        grouped.setdefault(speaker, []).append(SpeechInterval(start_ms, end_ms, (source_id,)))
    return {speaker: _merge_speech_intervals(items) for speaker, items in sorted(grouped.items())}


def _interval_duration(intervals: Sequence[Tuple[int, int]]) -> int:
    if not intervals:
        return 0
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def _overlap_evidence(
    speech: Sequence[SpeechInterval], active_segments: Sequence[dict]
) -> Tuple[int, List[dict]]:
    evidence = []
    for speech_interval in speech:
        for segment in active_segments:
            start = max(speech_interval.start_ms, int(segment["start_ms"]))
            end = min(speech_interval.end_ms, int(segment["end_ms"]))
            if end <= start:
                continue
            evidence.append(
                {
                    "speech_span_ids": list(speech_interval.source_ids),
                    "active_speaker_segment_id": segment["segment_id"],
                    "start_ms": start,
                    "end_ms": end,
                    "duration_ms": end - start,
                }
            )
    duration = _interval_duration([(item["start_ms"], item["end_ms"]) for item in evidence])
    return duration, evidence


def _maximum_overlap_assignment(
    speakers: Sequence[str], faces: Sequence[str], overlaps: Dict[Tuple[str, str], int]
) -> Dict[str, Optional[str]]:
    best_total = -1
    best_signature: Optional[Tuple[str, ...]] = None
    best_mapping: Dict[str, Optional[str]] = {}

    def visit(index: int, used: set, total: int, mapping: Dict[str, Optional[str]]) -> None:
        nonlocal best_total, best_signature, best_mapping
        if index == len(speakers):
            signature = tuple(mapping[speaker] or "~" for speaker in speakers)
            if total > best_total or (total == best_total and (best_signature is None or signature < best_signature)):
                best_total = total
                best_signature = signature
                best_mapping = dict(mapping)
            return
        speaker = speakers[index]
        options = [face for face in faces if face not in used and overlaps.get((speaker, face), 0) > 0]
        options.sort(key=lambda face: (-overlaps[(speaker, face)], face))
        options.append(None)
        for face in options:
            mapping[speaker] = face
            if face is None:
                visit(index + 1, used, total, mapping)
            else:
                used.add(face)
                visit(index + 1, used, total + overlaps[(speaker, face)], mapping)
                used.remove(face)
        mapping.pop(speaker, None)

    visit(0, set(), 0, {})
    return best_mapping


def bind_speakers_to_faces(
    visual_path: Path,
    speech_path: Path,
    output_path: Optional[Path] = None,
    timeline_offset_ms: int = 0,
    min_evidence_ms: int = 1000,
    min_speaker_coverage: float = 0.50,
    min_margin: float = 0.40,
) -> dict:
    if min_evidence_ms < 0:
        raise ValueError("min_evidence_ms cannot be negative")
    for name, value in (("min_speaker_coverage", min_speaker_coverage), ("min_margin", min_margin)):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between zero and one")
    visual_path = visual_path.expanduser().resolve()
    speech_path = speech_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve() if output_path else visual_path
    with visual_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    active_segments = data.get("active_speaker_segments", [])
    if not active_segments:
        raise RuntimeError("visual tracks contain no active-speaker segments; run ./lab active-speaker first")

    duration_ms = int(data["source"]["duration_ms"])
    speech_by_speaker = _load_speech_timeline(speech_path, duration_ms, timeline_offset_ms)
    if not speech_by_speaker:
        raise RuntimeError("speech timeline contains no final single-speaker A/B/C/D intervals in the visual range")
    active_by_face: Dict[str, List[dict]] = {}
    for segment in active_segments:
        face_id = segment.get("face_id")
        if face_id:
            active_by_face.setdefault(face_id, []).append(segment)

    speakers = sorted(speech_by_speaker)
    faces = sorted(active_by_face)
    overlaps: Dict[Tuple[str, str], int] = {}
    evidence_by_pair: Dict[Tuple[str, str], List[dict]] = {}
    for speaker in speakers:
        for face_id in faces:
            duration, evidence = _overlap_evidence(speech_by_speaker[speaker], active_by_face[face_id])
            overlaps[(speaker, face_id)] = duration
            evidence_by_pair[(speaker, face_id)] = evidence
    assignment = _maximum_overlap_assignment(speakers, faces, overlaps)

    associations = []
    for index, speaker in enumerate(speakers, start=1):
        speech_duration = _interval_duration(
            [(item.start_ms, item.end_ms) for item in speech_by_speaker[speaker]]
        )
        candidates = []
        for face_id in faces:
            overlap = overlaps[(speaker, face_id)]
            if overlap <= 0:
                continue
            candidates.append(
                {
                    "face_id": face_id,
                    "evidence_duration_ms": overlap,
                    "speaker_coverage": round(overlap / speech_duration, 6),
                    "evidence": evidence_by_pair[(speaker, face_id)],
                }
            )
        candidates.sort(key=lambda item: (-item["evidence_duration_ms"], item["face_id"]))
        face_id = assignment.get(speaker)
        overlap = overlaps.get((speaker, face_id), 0) if face_id else 0
        other_candidates = [item for item in candidates if item["face_id"] != face_id]
        runner_up = other_candidates[0] if other_candidates else None
        runner_up_overlap = runner_up["evidence_duration_ms"] if runner_up else 0
        coverage = overlap / speech_duration if speech_duration else 0.0
        face_duration = (
            _interval_duration(
                [(int(item["start_ms"]), int(item["end_ms"])) for item in active_by_face[face_id]]
            )
            if face_id
            else 0
        )
        purity = overlap / face_duration if face_duration else 0.0
        margin = (overlap - runner_up_overlap) / speech_duration if speech_duration else 0.0
        dominance = max(0.0, (overlap - runner_up_overlap) / overlap) if overlap else 0.0
        confidence = max(0.0, min(1.0, 0.60 * coverage + 0.25 * purity + 0.15 * dominance))
        if face_id is None:
            status = "ambiguous" if candidates else "offscreen"
        elif overlap < min_evidence_ms:
            status = "candidate"
        elif margin < min_margin:
            status = "ambiguous"
        elif coverage < min_speaker_coverage:
            status = "candidate"
        else:
            status = "confirmed"
        associations.append(
            {
                "association_id": f"speaker-face-{index:02d}",
                "speaker_label": speaker,
                "face_id": face_id,
                "confidence": round(confidence, 6),
                "evidence_duration_ms": overlap,
                "speaker_speech_duration_ms": speech_duration,
                "face_active_duration_ms": face_duration,
                "speaker_coverage": round(coverage, 6),
                "face_purity": round(purity, 6),
                "margin": round(margin, 6),
                "status": status,
                "runner_up_face_id": runner_up["face_id"] if runner_up else None,
                "runner_up_evidence_duration_ms": runner_up_overlap,
                "candidate_faces": candidates,
                "evidence": evidence_by_pair.get((speaker, face_id), []) if face_id else [],
            }
        )

    data["speaker_face_associations"] = associations
    data["processing"]["speaker_face_binding"] = {
        "algorithm": "one_to_one_maximum_temporal_overlap_v1",
        "speech_timeline_path": str(speech_path),
        "speech_timeline_sha256": file_sha256(speech_path),
        "timeline_offset_ms": timeline_offset_ms,
        "min_evidence_ms": min_evidence_ms,
        "min_speaker_coverage": min_speaker_coverage,
        "min_margin": min_margin,
    }
    data["statistics"]["speaker_face_association_count"] = len(associations)
    data["statistics"]["confirmed_speaker_face_association_count"] = sum(
        item["status"] == "confirmed" for item in associations
    )
    data["warnings"] = [item for item in data["warnings"] if not item.startswith("Speaker-face binding")]
    unresolved = [item["speaker_label"] for item in associations if item["status"] != "confirmed"]
    if unresolved:
        data["warnings"].append(
            "Speaker-face binding needs more evidence for: " + ", ".join(unresolved)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output_path)
    return data
