from __future__ import annotations

import numpy as np

from scripts.face_tracks import Detection
from scripts.live_video import OnlineFaceTracker


def _detection(box, embedding) -> Detection:
    return Detection(
        box=list(box),
        landmarks=[[0.0, 0.0]] * 5,
        score=0.95,
        quality=0.9,
        embedding=np.asarray(embedding, dtype=np.float32),
    )


def test_online_tracker_reuses_faces_and_bounds_identity_count() -> None:
    tracker = OnlineFaceTracker(
        5.0,
        max_gap_seconds=0.8,
        reid_threshold=0.75,
        cluster_threshold=0.70,
        min_observations=2,
        max_people=2,
    )
    first = _detection((10, 10, 40, 40), (1, 0, 0, 0))
    second = _detection((100, 10, 40, 40), (0, 1, 0, 0))
    assert all(track.face_id is None for _, track in tracker.update([first, second], 0))
    assigned = tracker.update([first, second], 1)
    assert [track.face_id for _, track in assigned] == ["Face-01", "Face-02"]

    tracker.update([], 10)
    assert not tracker.tracks
    assert tracker.update([first], 11)[0][1].face_id is None
    assert tracker.update([first], 12)[0][1].face_id == "Face-01"

    third = _detection((190, 10, 40, 40), (0, 0, 1, 0))
    tracker.update([first, second, third], 13)
    current = tracker.update([first, second, third], 14)
    assert len(tracker.face_profiles) == 2
    assert current[2][1].face_id is None
    assert tracker.dropped_identity_candidates == 1


if __name__ == "__main__":
    test_online_tracker_reuses_faces_and_bounds_identity_count()
    print("PASS: bounded online face tracking and cross-gap re-identification")
