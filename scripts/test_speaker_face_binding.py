from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.speaker_face_binding import _maximum_overlap_assignment, bind_speakers_to_faces


def test_global_assignment_avoids_greedy_face_conflict() -> None:
    overlaps = {
        ("A", "Face-01"): 900,
        ("A", "Face-02"): 800,
        ("B", "Face-01"): 850,
        ("B", "Face-02"): 100,
    }
    mapping = _maximum_overlap_assignment(["A", "B"], ["Face-01", "Face-02"], overlaps)
    assert mapping == {"A": "Face-02", "B": "Face-01"}


def test_one_to_one_binding_and_offscreen_status() -> None:
    visual = {
        "source": {"duration_ms": 4000},
        "processing": {},
        "active_speaker_segments": [
            {"segment_id": "active-1", "face_id": "Face-01", "start_ms": 0, "end_ms": 900},
            {"segment_id": "active-2", "face_id": "Face-02", "start_ms": 1000, "end_ms": 2000},
            {"segment_id": "active-3", "face_id": "Face-01", "start_ms": 2000, "end_ms": 2400},
        ],
        "speaker_face_associations": [],
        "statistics": {},
        "warnings": [],
    }
    speech = [
        {"id": "span-a", "speaker": "A", "start_s": 0.0, "end_s": 1.0, "status": "final"},
        {"id": "span-b", "speaker": "B", "start_s": 1.0, "end_s": 2.0, "status": "final"},
        {"id": "span-c", "speaker": "C", "start_s": 3.0, "end_s": 4.0, "status": "final"},
    ]
    with tempfile.TemporaryDirectory() as directory:
        visual_path = Path(directory) / "visual.json"
        speech_path = Path(directory) / "speech.json"
        visual_path.write_text(json.dumps(visual), encoding="utf-8")
        speech_path.write_text(json.dumps(speech), encoding="utf-8")
        result = bind_speakers_to_faces(
            visual_path, speech_path, min_evidence_ms=500, min_speaker_coverage=0.5, min_margin=0.2
        )
    associations = {item["speaker_label"]: item for item in result["speaker_face_associations"]}
    assert associations["A"]["face_id"] == "Face-01"
    assert associations["A"]["status"] == "confirmed"
    assert associations["A"]["candidate_faces"][0]["evidence"][0]["duration_ms"] == 900
    assert associations["B"]["face_id"] == "Face-02"
    assert associations["B"]["status"] == "confirmed"
    assert associations["C"]["face_id"] is None
    assert associations["C"]["status"] == "offscreen"
    assert associations["A"]["evidence"][0]["speech_span_ids"] == ["span-a"]


def test_rejects_raw_sortformer_slots() -> None:
    visual = {
        "source": {"duration_ms": 1000},
        "processing": {},
        "active_speaker_segments": [
            {"segment_id": "active-1", "face_id": "Face-01", "start_ms": 0, "end_ms": 1000}
        ],
        "speaker_face_associations": [],
        "statistics": {},
        "warnings": [],
    }
    with tempfile.TemporaryDirectory() as directory:
        visual_path = Path(directory) / "visual.json"
        speech_path = Path(directory) / "speech.json"
        visual_path.write_text(json.dumps(visual), encoding="utf-8")
        speech_path.write_text(
            json.dumps([{"speaker": "speaker0", "start": 0, "end": 1}]), encoding="utf-8"
        )
        try:
            bind_speakers_to_faces(visual_path, speech_path)
        except ValueError as error:
            assert "stabilized A/B/C/D" in str(error)
        else:
            raise AssertionError("raw Sortformer slot was accepted")


def test_default_keeps_short_visual_evidence_as_candidate() -> None:
    visual = {
        "source": {"duration_ms": 2500},
        "processing": {},
        "active_speaker_segments": [
            {"segment_id": "active-1", "face_id": "Face-01", "start_ms": 0, "end_ms": 2500}
        ],
        "speaker_face_associations": [],
        "statistics": {},
        "warnings": [],
    }
    speech = [{"id": "span-a", "speaker": "A", "start_s": 0.0, "end_s": 2.5}]
    with tempfile.TemporaryDirectory() as directory:
        visual_path = Path(directory) / "visual.json"
        speech_path = Path(directory) / "speech.json"
        visual_path.write_text(json.dumps(visual), encoding="utf-8")
        speech_path.write_text(json.dumps(speech), encoding="utf-8")
        result = bind_speakers_to_faces(visual_path, speech_path)
    association = result["speaker_face_associations"][0]
    assert association["face_id"] == "Face-01"
    assert association["status"] == "candidate"


if __name__ == "__main__":
    test_global_assignment_avoids_greedy_face_conflict()
    test_one_to_one_binding_and_offscreen_status()
    test_rejects_raw_sortformer_slots()
    test_default_keeps_short_visual_evidence_as_candidate()
    print("PASS: speaker-face temporal binding")
