from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.face_tracks import Detection
from scripts.live_video import OnlineFaceTracker, _portrait_score, _speaker_position_records


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


def test_online_tracker_allows_more_than_four_session_identities() -> None:
    tracker = OnlineFaceTracker(
        5.0,
        max_gap_seconds=0.8,
        reid_threshold=0.75,
        cluster_threshold=0.70,
        min_observations=2,
        max_people=12,
    )
    identities = []
    for index in range(5):
        embedding = np.zeros(5, dtype=np.float32)
        embedding[index] = 1
        detection = _detection((10 + index * 50, 10, 40, 40), embedding)
        tracker.update([detection], index * 10)
        identities.append(tracker.update([detection], index * 10 + 1)[0][1].face_id)
    assert identities == ["Face-01", "Face-02", "Face-03", "Face-04", "Face-05"]


def test_scene_cut_breaks_a_positional_track() -> None:
    tracker = OnlineFaceTracker(
        5.0,
        reid_threshold=0.75,
        cluster_threshold=0.70,
        min_observations=2,
    )
    teacher = _detection((100, 10, 40, 40), (1, 0, 0, 0))
    student = _detection((100, 10, 40, 40), (0.3, 0.954, 0, 0))
    tracker.update([teacher], 0)
    assert tracker.update([teacher], 1)[0][1].face_id == "Face-01"

    tracker.start_new_scene()
    assert tracker.update([student], 2)[0][1].face_id is None
    assert tracker.update([student], 3)[0][1].face_id == "Face-02"


def test_portrait_score_prefers_a_frontal_face() -> None:
    frontal = _detection((10, 10, 80, 80), (1, 0, 0, 0))
    frontal.landmarks = [[30, 35], [70, 35], [50, 50], [35, 68], [65, 68]]
    turned = _detection((10, 10, 80, 80), (1, 0, 0, 0))
    turned.landmarks = [[30, 35], [70, 41], [66, 50], [45, 68], [72, 68]]
    assert _portrait_score(frontal) > _portrait_score(turned)


def test_speaker_positions_keep_only_active_samples_and_normalize_boxes() -> None:
    chunk = SimpleNamespace(
        chunk_id="chunk-1",
        observations={
            "Face-01": (
                {"timestamp_ms": 1000, "bbox_px": [100, 50, 200, 100]},
                {"timestamp_ms": 1200, "bbox_px": [120, 60, 200, 100]},
                {"timestamp_ms": 1400, "bbox_px": [140, 70, 200, 100]},
            ),
            "Face-02": (
                {"timestamp_ms": 1200, "bbox_px": [400, 50, 100, 100]},
            ),
        },
    )
    records = _speaker_position_records(
        chunk,
        [{"face_id": "Face-01", "start_ms": 1100, "end_ms": 1400, "score": 0.82}],
        1000,
        500,
    )

    assert records == [{
        "schema_version": 1,
        "timestamp_ms": 1200,
        "face_id": "Face-01",
        "bbox_normalized": [0.12, 0.12, 0.2, 0.2],
        "score": 0.82,
        "chunk_id": "chunk-1",
    }]


if __name__ == "__main__":
    test_online_tracker_reuses_faces_and_bounds_identity_count()
    test_online_tracker_allows_more_than_four_session_identities()
    test_scene_cut_breaks_a_positional_track()
    test_portrait_score_prefers_a_frontal_face()
    test_speaker_positions_keep_only_active_samples_and_normalize_boxes()
    print("PASS: bounded online face tracking and cross-gap re-identification")
