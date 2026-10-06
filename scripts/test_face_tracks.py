from __future__ import annotations

import numpy as np

from scripts.face_tracks import Detection, TrackManager, Tracklet, cluster_tracklets


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


def _tracklet(tracklet_id: str, start_ms: int, end_ms: int, embedding) -> Tracklet:
    return Tracklet(
        tracklet_id=tracklet_id,
        start_ms=start_ms,
        end_ms=end_ms,
        last_frame=end_ms // 100,
        last_box=[0, 0, 80, 80],
        embedding_sum=np.asarray(embedding, dtype=np.float32),
        embedding_weight=1.0,
        observation_ids=[f"observation-{tracklet_id}"],
        detection_scores=[0.95],
    )


def test_global_clustering_uses_later_cooccurrence_to_separate_lookalikes() -> None:
    # The two close-ups are similar enough to cross the cluster threshold, but
    # each has a stronger match in a later group shot. Those group-shot tracks
    # overlap, proving the two identities cannot be merged.
    tracks = [
        _tracklet("a-close", 0, 900, [1.0, 0.0]),
        _tracklet("b-close", 1000, 1900, [0.8, 0.6]),
        _tracklet("a-group", 3000, 3900, [0.99, 0.05]),
        _tracklet("b-group", 3000, 3900, [0.82, 0.57]),
    ]
    observations = [
        {"observation_id": track.observation_ids[0], "face_id": None}
        for track in tracks
    ]
    clusters = cluster_tracklets(tracks, observations, 1, 0.75, 100)
    assert {tuple(item.tracklet_id for item in cluster) for cluster in clusters} == {
        ("a-close", "a-group"),
        ("b-close", "b-group"),
    }


def test_camera_cut_does_not_continue_track_on_box_overlap_alone() -> None:
    manager = TrackManager(fps=10.0, max_gap_seconds=1.0, reid_threshold=0.75)
    manager.update([_detection(10, [1.0, 0.0])], 0, 0, 400, 200)
    manager.update([_detection(10, [0.0, 1.0])], 1, 100, 400, 200)
    assert len(manager.tracklets) == 2


def test_short_low_quality_fragment_bridges_back_to_surrounding_track() -> None:
    target = _tracklet("main", 0, 400, [1.0, 0.0])
    target.observation_ids = [
        "main-before-1",
        "main-before-2",
        "main-before-3",
        "main-after-1",
        "main-after-2",
        "main-after-3",
    ]
    fragment = _tracklet("fragment", 100, 300, [0.0, 1.0])
    fragment.observation_ids = ["fragment-1", "fragment-2", "fragment-3"]
    observations = [
        {
            "observation_id": name,
            "timestamp_ms": timestamp,
            "bbox_px": box,
            "face_id": None,
        }
        for name, timestamp, box in [
            ("main-before-1", 0, [10, 20, 80, 80]),
            ("main-before-2", 25, [10, 20, 80, 80]),
            ("main-before-3", 50, [11, 20, 80, 80]),
            ("fragment-1", 100, [12, 21, 80, 80]),
            ("fragment-2", 200, [13, 22, 80, 80]),
            ("fragment-3", 300, [14, 22, 80, 80]),
            ("main-after-1", 350, [15, 23, 80, 80]),
            ("main-after-2", 375, [15, 23, 80, 80]),
            ("main-after-3", 400, [15, 23, 80, 80]),
        ]
    ]
    clusters = cluster_tracklets([target, fragment], observations, 1, 0.45, 100)
    assert [[item.tracklet_id for item in cluster] for cluster in clusters] == [
        ["main", "fragment"]
    ]


if __name__ == "__main__":
    test_crossing_faces_keep_identity()
    test_fragmented_tracklets_merge_but_overlapping_people_do_not()
    test_global_clustering_uses_later_cooccurrence_to_separate_lookalikes()
    test_camera_cut_does_not_continue_track_on_box_overlap_alone()
    test_short_low_quality_fragment_bridges_back_to_surrounding_track()
    print("PASS: identity matching and fragmented-track clustering")
