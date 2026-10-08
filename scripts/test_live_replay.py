from __future__ import annotations

from scripts.live_replay import VisionReplay, graph_events


class FakeClock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now


def _graph() -> dict:
    return {
        "nodes": [
            {"id": "person:A", "type": "person", "label": "A", "evidence_refs": [], "attributes": {}},
            {"id": "scene:1", "type": "scene", "label": "Room", "confidence": 0.9, "evidence_refs": ["frame:1"], "attributes": {"start_ms": 0, "end_ms": 1000}},
            {"id": "scene:2", "type": "scene", "label": "Hall", "confidence": 0.8, "evidence_refs": ["frame:2"], "attributes": {"start_ms": 1000, "end_ms": 2000}},
        ],
        "edges": [
            {"id": "edge-000001", "source": "person:A", "target": "scene:1", "predicate": "speaking", "epistemic_status": "observed", "confidence": 0.9, "start_ms": 200, "end_ms": 800, "evidence_refs": ["turn:1"]},
            {"id": "edge-000002", "source": "person:A", "target": "scene:2", "predicate": "present_in", "epistemic_status": "observed", "confidence": 0.9, "start_ms": 1200, "end_ms": 1800, "evidence_refs": ["frame:2"]},
        ],
    }


def test_events_and_replay_controls() -> None:
    events = graph_events(_graph())
    assert [event["timestamp_ms"] for event in events] == [0, 200, 1000, 1200]
    assert [event["type"] for event in events] == [
        "scene_changed",
        "person_speaking",
        "scene_changed",
        "person_seen",
    ]

    clock = FakeClock()
    replay = VisionReplay(_graph(), speed=2, event_window=2, clock=clock)
    assert replay.snapshot()["visible_node_ids"] == ["scene:1"]
    clock.now += 0.6
    snapshot = replay.snapshot()
    assert snapshot["current_ms"] == 1200
    assert snapshot["current_scene_id"] == "scene:2"
    assert snapshot["visible_edge_ids"] == ["edge-000001", "edge-000002"]
    assert len(snapshot["events"]) == 2

    paused = replay.control("pause")
    clock.now += 4
    assert replay.snapshot()["current_ms"] == paused["current_ms"]
    assert replay.control("seek", 500)["current_ms"] == 500
    assert replay.control("resume")["status"] == "playing"
    assert replay.control("restart")["current_ms"] == 0


if __name__ == "__main__":
    test_events_and_replay_controls()
    print("PASS: visual events and bounded replay controls")
