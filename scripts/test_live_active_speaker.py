from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path

from scripts.live_active_speaker import (
    ActiveSpeakerChunk,
    FaceSpeakingStateMachine,
    LiveActiveSpeakerWorker,
    OnlineSpeakerBinder,
    _sequence_specs,
)


def _chunk(index: int, face_id: str = "Face-01") -> ActiveSpeakerChunk:
    start = (index - 1) * 4000
    observations = tuple(
        {
            "timestamp_ms": start + offset,
            "bbox_px": [10 + offset // 200, 20, 40, 40],
        }
        for offset in range(0, 4001, 200)
    )
    return ActiveSpeakerChunk(
        chunk_id=f"chunk-{index}",
        start_ms=start,
        publish_start_ms=start,
        end_ms=start + 4000,
        observations={face_id: observations},
    )


def test_face_observations_interpolate_to_25fps() -> None:
    specs = _sequence_specs(_chunk(1), 25.0)
    assert len(specs) == 1
    assert specs[0]["face_id"] == "Face-01"
    assert specs[0]["times_ms"].tolist() == list(range(0, 4001, 40))
    assert specs[0]["boxes"].shape == (101, 4)


def test_online_binding_requires_accumulated_timeline_overlap() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "speech.json"
        path.write_text(
            json.dumps(
                [
                    {"id": "a", "speaker": "A", "start_s": 0, "end_s": 2},
                    {"id": "b", "speaker": "B", "start_s": 2, "end_s": 4},
                ]
            ),
            encoding="utf-8",
        )
        binder = OnlineSpeakerBinder(
            path,
            4000,
            min_evidence_ms=1000,
            min_speaker_coverage=0.5,
            min_margin=0.2,
        )
        associations = binder.associations(
            [
                {"segment_id": "s1", "face_id": "Face-01", "start_ms": 0, "end_ms": 1900},
                {"segment_id": "s2", "face_id": "Face-02", "start_ms": 2100, "end_ms": 3900},
            ]
        )
    by_speaker = {item["speaker_label"]: item for item in associations}
    assert by_speaker["A"]["face_id"] == "Face-01"
    assert by_speaker["A"]["status"] == "confirmed"
    assert by_speaker["B"]["face_id"] == "Face-02"
    assert by_speaker["B"]["status"] == "confirmed"


def test_worker_bounds_chunks_without_loading_torch() -> None:
    gate = threading.Event()
    started = threading.Event()

    def processor(chunk):
        started.set()
        gate.wait(2)
        face_id = next(iter(chunk.observations))
        return [
            {
                "segment_id": "local",
                "face_id": face_id,
                "start_ms": chunk.start_ms,
                "end_ms": chunk.end_ms,
                "score": 0.9,
                "raw_score_mean": 2.0,
            }
        ]

    results, dropped = [], []
    finished = threading.Event()
    worker = LiveActiveSpeakerWorker(
        Path("unused.mp4"),
        25.0,
        on_result=lambda chunk, segments, elapsed: results.append(chunk.chunk_id),
        on_error=lambda _chunk, error: (_ for _ in ()).throw(error),
        on_drop=lambda chunk: dropped.append(chunk.chunk_id),
        on_ready=lambda _device: None,
        on_finished=finished.set,
        queue_size=1,
        processor=processor,
    )
    worker.start()
    worker.submit(_chunk(1))
    assert started.wait(1)
    worker.submit(_chunk(2))
    worker.submit(_chunk(3))
    worker.close()
    gate.set()
    assert finished.wait(2)
    assert results == ["chunk-1", "chunk-3"]
    assert dropped == ["chunk-2"]


def test_speaking_state_is_independent_for_each_face() -> None:
    machine = FaceSpeakingStateMachine(start_confirm_ms=200, end_silence_ms=600)
    first = machine.update(
        [
            {
                "face_id": "Face-01",
                "timestamp_ms": timestamp,
                "is_speaking": True,
            }
            for timestamp in range(0, 1000, 40)
        ],
        publish_start_ms=0,
        end_ms=1000,
    )
    second = machine.update(
        [
            {
                "face_id": "Face-02",
                "timestamp_ms": timestamp,
                "is_speaking": True,
            }
            for timestamp in range(1000, 2000, 40)
        ],
        publish_start_ms=1000,
        end_ms=2000,
    )
    assert first == [
        {"face_id": "Face-01", "phase": "started", "timestamp_ms": 0}
    ]
    assert {item["face_id"]: item["phase"] for item in second} == {
        "Face-01": "ended",
        "Face-02": "started",
    }
    assert next(item for item in second if item["face_id"] == "Face-01")[
        "timestamp_ms"
    ] == 1000


def test_subsecond_boundary_gap_does_not_end_speaker() -> None:
    machine = FaceSpeakingStateMachine(start_confirm_ms=200, end_silence_ms=1000)
    machine.update(
        [
            {"face_id": "Face-01", "timestamp_ms": timestamp, "is_speaking": True}
            for timestamp in range(0, 1000, 40)
        ],
        publish_start_ms=0,
        end_ms=1000,
    )
    assert machine.update([], publish_start_ms=1000, end_ms=1960) == []
    resumed = machine.update(
        [
            {"face_id": "Face-01", "timestamp_ms": timestamp, "is_speaking": True}
            for timestamp in range(1960, 2960, 40)
        ],
        publish_start_ms=1960,
        end_ms=2960,
    )
    assert [item["phase"] for item in resumed] == ["continued"]


if __name__ == "__main__":
    test_face_observations_interpolate_to_25fps()
    test_online_binding_requires_accumulated_timeline_overlap()
    test_worker_bounds_chunks_without_loading_torch()
    test_speaking_state_is_independent_for_each_face()
    test_subsecond_boundary_gap_does_not_end_speaker()
    print("PASS: chunked active speaker interpolation, queue bounds and A/B binding")
