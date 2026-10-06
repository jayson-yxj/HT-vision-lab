from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.speaker_reconciliation import (
    reconcile_and_rebind,
    reconcile_speakers,
    validate_reconciled_timeline,
)


def test_confirmed_face_anchor_corrects_conflicting_span_and_rebinds() -> None:
    visual = {
        "session_id": "visual-test",
        "source": {"duration_ms": 4000},
        "processing": {},
        "active_speaker_segments": [
            {"segment_id": "active-1", "face_id": "Face-01", "start_ms": 0, "end_ms": 1000},
            {"segment_id": "active-2", "face_id": "Face-02", "start_ms": 1000, "end_ms": 2000},
            {"segment_id": "active-3", "face_id": "Face-01", "start_ms": 2000, "end_ms": 3000},
            {"segment_id": "active-4", "face_id": "Face-01", "start_ms": 3000, "end_ms": 3400},
        ],
        "speaker_face_associations": [
            {"speaker_label": "A", "face_id": "Face-01", "status": "confirmed", "confidence": 0.9},
            {"speaker_label": "B", "face_id": "Face-02", "status": "confirmed", "confidence": 0.9},
        ],
        "statistics": {},
        "warnings": [],
    }
    speech = [
        {"id": "span-a", "speaker": "A", "start_s": 0.0, "end_s": 1.0},
        {"id": "span-b", "speaker": "B", "start_s": 1.0, "end_s": 2.0},
        {"id": "span-wrong", "speaker": "B", "start_s": 2.0, "end_s": 3.0},
        {"id": "span-short", "speaker": "B", "start_s": 3.0, "end_s": 3.4},
    ]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        visual_path = root / "visual.json"
        speech_path = root / "speech.json"
        visual_path.write_text(json.dumps(visual), encoding="utf-8")
        speech_path.write_text(json.dumps(speech), encoding="utf-8")
        result = reconcile_speakers(
            visual_path,
            speech_path,
            min_overlap_ms=500,
            min_span_coverage=0.6,
            min_face_dominance=0.6,
        )
        assert not validate_reconciled_timeline(result)
        assert result["statistics"]["corrected_spans"] == 1
        correction = result["corrections"][0]
        assert correction["speech_span_id"] == "span-wrong"
        assert correction["original_speaker"] == "B"
        assert correction["resolved_speaker"] == "A"
        assert correction["face_id"] == "Face-01"
        assert result["speech_spans"][2]["speaker"] == "A"
        assert result["speech_spans"][3]["speaker"] == "B"

        final_timeline, rebound, passes = reconcile_and_rebind(
            visual_path,
            speech_path,
            min_overlap_ms=500,
            min_span_coverage=0.6,
            min_face_dominance=0.6,
        )
        assert passes == 2
        assert (
            result["source"]["visual_tracks"]["evidence_sha256"]
            == final_timeline["source"]["visual_tracks"]["evidence_sha256"]
        )
    bindings = {
        item["speaker_label"]: item for item in rebound["speaker_face_associations"]
    }
    assert bindings["A"]["face_id"] == "Face-01"
    assert bindings["A"]["status"] == "confirmed"
    assert bindings["B"]["face_id"] == "Face-02"
    assert bindings["B"]["status"] == "confirmed"


def test_reconciliation_adds_new_anchors_until_corrections_stabilize() -> None:
    visual = {
        "session_id": "iterative-test",
        "source": {"duration_ms": 5000},
        "processing": {},
        "active_speaker_segments": [
            {
                "segment_id": f"active-{index}",
                "face_id": face_id,
                "start_ms": (index - 1) * 1000,
                "end_ms": index * 1000,
            }
            for index, face_id in enumerate(
                ["Face-01", "Face-02", "Face-02", "Face-03", "Face-03"],
                start=1,
            )
        ],
        "speaker_face_associations": [
            {
                "speaker_label": "A",
                "face_id": "Face-01",
                "status": "confirmed",
                "confidence": 0.9,
            },
            {
                "speaker_label": "B",
                "face_id": "Face-02",
                "status": "confirmed",
                "confidence": 0.9,
            },
            {
                "speaker_label": "C",
                "face_id": "Face-03",
                "status": "ambiguous",
                "confidence": 0.6,
            },
        ],
        "statistics": {},
        "warnings": [],
    }
    speech = [
        {"id": "a-correct", "speaker": "A", "start_s": 0.0, "end_s": 1.0},
        {"id": "b-correct", "speaker": "B", "start_s": 1.0, "end_s": 2.0},
        {"id": "c-is-b", "speaker": "C", "start_s": 2.0, "end_s": 3.0},
        {"id": "c-correct", "speaker": "C", "start_s": 3.0, "end_s": 4.0},
        {"id": "a-is-c", "speaker": "A", "start_s": 4.0, "end_s": 5.0},
    ]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        visual_path = root / "visual.json"
        speech_path = root / "speech.json"
        visual_path.write_text(json.dumps(visual), encoding="utf-8")
        speech_path.write_text(json.dumps(speech), encoding="utf-8")
        timeline, rebound, passes = reconcile_and_rebind(
            visual_path,
            speech_path,
            min_overlap_ms=500,
        )
    assert passes == 3
    assert [
        (item["speech_span_id"], item["resolved_speaker"])
        for item in timeline["corrections"]
    ] == [("c-is-b", "B"), ("a-is-c", "C")]
    assert all(
        item["status"] == "confirmed"
        for item in rebound["speaker_face_associations"]
    )


if __name__ == "__main__":
    test_confirmed_face_anchor_corrects_conflicting_span_and_rebinds()
    test_reconciliation_adds_new_anchors_until_corrections_stabilize()
    print("PASS: visual anchors reconcile conflicting speaker labels")
