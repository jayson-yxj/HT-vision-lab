from __future__ import annotations

import numpy as np

from scripts.active_speaker import TrackSequence, _assign_activity


def _sequence(face_id: str, scores) -> TrackSequence:
    count = len(scores)
    sequence = TrackSequence(
        tracklet_id=f"track-{face_id}",
        face_id=face_id,
        times_ms=np.arange(count, dtype=np.int64) * 40,
        source_frames=np.arange(count, dtype=np.int64),
        boxes=np.zeros((count, 4), dtype=np.float32),
    )
    sequence.raw_scores = np.asarray(scores, dtype=np.float32)
    sequence.probabilities = 1.0 / (1.0 + np.exp(-sequence.raw_scores))
    return sequence


def test_exclusive_activity_keeps_strongest_visible_face() -> None:
    speaker = _sequence("Face-01", [2.0] * 10)
    listener = _sequence("Face-02", [0.2] * 10)
    _assign_activity([speaker, listener], threshold=0.0, min_frames=3, bridge_frames=2, exclusive_speaker=True)
    assert speaker.speaking.tolist() == [True] * 10
    assert listener.speaking.tolist() == [False] * 10


def test_overlap_mode_preserves_two_positive_faces() -> None:
    left = _sequence("Face-01", [1.0] * 5)
    right = _sequence("Face-02", [0.5] * 5)
    _assign_activity([left, right], threshold=0.0, min_frames=1, bridge_frames=0, exclusive_speaker=False)
    assert left.speaking.all()
    assert right.speaking.all()


if __name__ == "__main__":
    test_exclusive_activity_keeps_strongest_visible_face()
    test_overlap_mode_preserves_two_positive_faces()
    print("PASS: active-speaker exclusivity")
