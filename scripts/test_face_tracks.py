from __future__ import annotations

import numpy as np

from scripts.face_tracks import Detection, TrackManager, cluster_tracklets


def _detection(x: int, embedding, score: float = 0.95) -> Detection:
    return Detection(
        box=[x, 20, 80, 80],
        landmarks=[[x + 20, 40], [x + 60, 40], [x + 40, 55], [x + 25, 75], [x + 55, 75]],
        score=score,
        quality=score,
        embedding=np.asarray(embedding, dtype=np.float32),
    )


def test_crossing_faces_keep_identity() -> None:
    manager = TrackManager(fps=10.0, max_gap_seconds=1.0, reid_threshold=0.36)
    person_a = [1.0, 0.0, 0.0]
    person_b = [0.0, 1.0, 0.0]
    manager.update([_detection(10, person_a), _detection(200, person_b)], 0, 0, 400, 200)
    manager.update([_detection(190, person_a), _detection(20, person_b)], 1, 100, 400, 200)
    assert len(manager.tracklets) == 2
    assert manager.observations[2]["tracklet_id"] == manager.observations[0]["tracklet_id"]
    assert manager.observations[3]["tracklet_id"] == manager.observations[1]["tracklet_id"]


def test_fragmented_tracklets_merge_but_overlapping_people_do_not() -> None:
    manager = TrackManager(fps=10.0, max_gap_seconds=0.2, reid_threshold=0.36)
    person_a = [1.0, 0.0, 0.0]
    person_b = [0.0, 1.0, 0.0]
    for frame in (0, 1, 10, 11):
        manager.update([_detection(10, person_a), _detection(200, person_b)], frame, frame * 100, 400, 200)
    clusters = cluster_tracklets(manager.tracklets, manager.observations, 2, 0.40, 100)
    assert len(manager.tracklets) == 4
    assert len(clusters) == 2
    assert sorted(len(cluster) for cluster in clusters) == [2, 2]


if __name__ == "__main__":
    test_crossing_faces_keep_identity()
    test_fragmented_tracklets_merge_but_overlapping_people_do_not()
    print("PASS: identity matching and fragmented-track clustering")
