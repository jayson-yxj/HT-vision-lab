from __future__ import annotations

from scripts.speaker_face_binding import SpeechInterval
from scripts.visibility import build_visibility_segments


def test_visibility_states_distinguish_short_gap_offscreen_and_unknown() -> None:
    data = {
        "source": {"duration_ms": 7000},
        "processing": {"sample_fps": 10.0},
        "tracklets": [
            {"tracklet_id": "track-a1", "face_id": "Face-01", "start_ms": 0, "end_ms": 900},
            {"tracklet_id": "track-a2", "face_id": "Face-01", "start_ms": 1500, "end_ms": 2900},
        ],
        "speaker_face_associations": [
            {"speaker_label": "A", "face_id": "Face-01", "status": "confirmed", "confidence": 0.95},
            {"speaker_label": "B", "face_id": None, "status": "offscreen", "confidence": 0.0},
            {"speaker_label": "C", "face_id": "Face-02", "status": "ambiguous", "confidence": 0.6},
        ],
    }
    speech = {
        "A": [
            SpeechInterval(0, 3000, ("span-a1",)),
            SpeechInterval(6000, 7000, ("span-a2",)),
        ],
        "B": [SpeechInterval(3000, 4000, ("span-b",))],
        "C": [SpeechInterval(4000, 5000, ("span-c",))],
    }
    segments = build_visibility_segments(
        data, speech, cuts=[], occlusion_max_ms=1200, boundary_tolerance_ms=100
    )
    states = [(item["speaker_label"], item["state"], item["start_ms"], item["end_ms"]) for item in segments]
    assert states == [
        ("A", "visible", 0, 1000),
        ("A", "occluded", 1000, 1500),
        ("A", "visible", 1500, 3000),
        ("B", "offscreen", 3000, 4000),
        ("C", "unknown", 4000, 5000),
        ("A", "offscreen", 6000, 7000),
    ]


def test_camera_cut_turns_short_gap_into_offscreen() -> None:
    data = {
        "source": {"duration_ms": 3000},
        "processing": {"sample_fps": 10.0},
        "tracklets": [
            {"tracklet_id": "left", "face_id": "Face-01", "start_ms": 0, "end_ms": 900},
            {"tracklet_id": "right", "face_id": "Face-01", "start_ms": 1500, "end_ms": 2900},
        ],
        "speaker_face_associations": [
            {"speaker_label": "A", "face_id": "Face-01", "status": "confirmed", "confidence": 0.95}
        ],
    }
    speech = {"A": [SpeechInterval(0, 3000, ("span-a",))]}
    segments = build_visibility_segments(
        data, speech, cuts=[1200], occlusion_max_ms=1200, boundary_tolerance_ms=100
    )
    assert [item["state"] for item in segments] == ["visible", "offscreen", "visible"]


if __name__ == "__main__":
    test_visibility_states_distinguish_short_gap_offscreen_and_unknown()
    test_camera_cut_turns_short_gap_into_offscreen()
    print("PASS: speaker visibility state classification")
