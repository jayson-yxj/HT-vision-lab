from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from scripts.live_semantics import LiveSemanticWorker, SemanticTask
from scripts.scene_semantics import normalize_live_analysis


def _raw(keyframe_id: str) -> dict:
    return {
        "keyframe_id": keyframe_id,
        "environment": {
            "category": "indoor_meeting",
            "description": "Two people beside a screen.",
            "confidence": 0.9,
        },
        "objects": [
            {"label": "screen", "count": 1, "region": "background", "confidence": 0.8}
        ],
        "interactions": [
            {
                "subject_ref": "Face-01",
                "predicate": "pointing_to",
                "object_ref": "object:screen",
                "description": "Face-01 points to the screen.",
                "evidence_basis": ["image"],
                "epistemic_status": "inferred",
                "confidence": 0.75,
            },
            {
                "subject_ref": "Face-01",
                "predicate": "speaking",
                "object_ref": "group",
                "description": "Face-01 may be speaking.",
                "evidence_basis": ["image"],
                "epistemic_status": "inferred",
                "confidence": 0.5,
            },
        ],
    }


class FakeAnalyzer:
    model = "fake/qwen"

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.calls = 0
        self.started = threading.Event()
        self.gate = gate

    def analyze(self, metadata, _image_paths):
        self.calls += 1
        self.started.set()
        if self.gate:
            self.gate.wait(2)
        return [_raw(item["keyframe_id"]) for item in metadata]


def _task(root: Path, index: int) -> SemanticTask:
    image = root / f"frame-{index}.jpg"
    image.write_bytes(f"image-{index}".encode())
    keyframe_id = f"live-keyframe-{index:05d}"
    return SemanticTask(
        keyframe_id=keyframe_id,
        scene_id="live-scene:0001",
        timestamp_ms=index * 1000,
        metadata={
            "keyframe_id": keyframe_id,
            "visible_people": [{"face_id": "Face-01"}],
        },
        image_path=image,
    )


def test_live_normalization_uses_only_visible_image_evidence() -> None:
    normalized = normalize_live_analysis(_raw("frame-1"), ["Face-01"])
    assert normalized["environment"]["category"] == "indoor_meeting"
    assert normalized["objects"][0]["label"] == "screen"
    assert [item["predicate"] for item in normalized["interactions"]] == ["pointing_to"]


def test_worker_bounds_queue_and_reuses_cache() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        gate = threading.Event()
        analyzer = FakeAnalyzer(gate)
        results, dropped = [], []
        finished = threading.Event()
        worker = LiveSemanticWorker(
            analyzer,
            root / "cache",
            on_result=lambda task, raw, request_hash, cache_hit: results.append(
                (task.keyframe_id, raw, request_hash, cache_hit)
            ),
            on_error=lambda _task, error: (_ for _ in ()).throw(error),
            on_drop=lambda task: dropped.append(task.keyframe_id),
            on_finished=finished.set,
            queue_size=1,
        )
        tasks = [_task(root, index) for index in range(1, 4)]
        worker.start()
        worker.submit(tasks[0])
        assert analyzer.started.wait(1)
        worker.submit(tasks[1])
        worker.submit(tasks[2])
        worker.close()
        gate.set()
        assert finished.wait(2)
        assert [item[0] for item in results] == [tasks[0].keyframe_id, tasks[2].keyframe_id]
        assert dropped == [tasks[1].keyframe_id]

        cached = []
        cached_finished = threading.Event()
        second = LiveSemanticWorker(
            analyzer,
            root / "cache",
            on_result=lambda task, raw, request_hash, cache_hit: cached.append(cache_hit),
            on_error=lambda _task, error: (_ for _ in ()).throw(error),
            on_drop=lambda _task: None,
            on_finished=cached_finished.set,
        )
        second.start()
        second.submit(tasks[0])
        second.close()
        assert cached_finished.wait(2)
        assert cached == [True]
        assert analyzer.calls == 2


def test_provider_failure_stops_cloud_work_without_blocking_completion() -> None:
    class BrokenAnalyzer:
        model = "fake/broken"

        def analyze(self, _metadata, _image_paths):
            raise OSError("provider unavailable")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        errors, dropped = [], []
        finished = threading.Event()
        worker = LiveSemanticWorker(
            BrokenAnalyzer(),
            root / "cache",
            on_result=lambda *_args: None,
            on_error=lambda task, error: errors.append((task.keyframe_id, str(error))),
            on_drop=lambda task: dropped.append(task.keyframe_id),
            on_finished=finished.set,
            queue_size=2,
        )
        tasks = [_task(root, index) for index in range(1, 3)]
        worker.submit(tasks[0])
        worker.submit(tasks[1])
        worker.start()
        worker.close()
        assert finished.wait(2)
        assert worker.finished
        assert errors == [(tasks[0].keyframe_id, "provider unavailable")]
        assert dropped == [tasks[1].keyframe_id]


if __name__ == "__main__":
    test_live_normalization_uses_only_visible_image_evidence()
    test_worker_bounds_queue_and_reuses_cache()
    test_provider_failure_stops_cloud_work_without_blocking_completion()
    print("PASS: bounded asynchronous Qwen worker and live semantic normalization")
