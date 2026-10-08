from __future__ import annotations

import time
from bisect import bisect_right
from threading import Lock
from typing import Callable, Dict, List, Optional


DIALOGUE_PREDICATES = {
    "expresses",
    "about",
    "has_intent",
    "occurred_in",
    "discussed_in",
}
STRUCTURAL_PREDICATES = {"followed_by"}


def _edge_event_type(edge: dict, nodes: Dict[str, dict]) -> str:
    predicate = edge["predicate"]
    if predicate == "present_in":
        return "person_seen"
    if predicate == "contains":
        return "object_detected"
    if predicate == "speaking":
        return "person_speaking"
    if predicate == "identity_candidate":
        return "identity_candidate"
    if predicate in DIALOGUE_PREDICATES:
        return "dialogue_linked"
    if predicate in STRUCTURAL_PREDICATES:
        return "relation_observed"
    source_type = nodes.get(edge["source"], {}).get("type")
    return "interaction_detected" if source_type in {"person", "visual_identity"} else "relation_observed"


def graph_events(graph: dict) -> List[dict]:
    """Project a completed graph into deterministic, time-ordered visual events."""
    nodes = {node["id"]: node for node in graph["nodes"]}
    pending = []
    connected = set()
    for node_index, node in enumerate(graph["nodes"]):
        if node["type"] != "scene":
            continue
        timestamp = int(node.get("attributes", {}).get("start_ms", 0))
        pending.append(
            (
                timestamp,
                0,
                node_index,
                {
                    "type": "scene_changed",
                    "timestamp_ms": timestamp,
                    "subject_ref": node["id"],
                    "object_ref": None,
                    "evidence_refs": node.get("evidence_refs", []),
                    "payload": {
                        "label": node["label"],
                        "confidence": node.get("confidence"),
                        "reveal_node_ids": [node["id"]],
                        "reveal_edge_ids": [],
                    },
                },
            )
        )
    for edge_index, edge in enumerate(graph["edges"]):
        timestamp = int(edge.get("start_ms", 0))
        connected.update((edge["source"], edge["target"]))
        pending.append(
            (
                timestamp,
                1,
                edge_index,
                {
                    "type": _edge_event_type(edge, nodes),
                    "timestamp_ms": timestamp,
                    "subject_ref": edge["source"],
                    "object_ref": edge["target"],
                    "evidence_refs": edge.get("evidence_refs", []),
                    "payload": {
                        "predicate": edge["predicate"],
                        "confidence": edge.get("confidence"),
                        "epistemic_status": edge.get("epistemic_status"),
                        "end_ms": edge.get("end_ms", timestamp),
                        "reveal_node_ids": [edge["source"], edge["target"]],
                        "reveal_edge_ids": [edge["id"]],
                    },
                },
            )
        )
    for node_index, node in enumerate(graph["nodes"]):
        if node["id"] in connected or node["type"] == "scene":
            continue
        timestamp = int(node.get("attributes", {}).get("start_ms", 0))
        pending.append(
            (
                timestamp,
                2,
                node_index,
                {
                    "type": "node_available",
                    "timestamp_ms": timestamp,
                    "subject_ref": node["id"],
                    "object_ref": None,
                    "evidence_refs": node.get("evidence_refs", []),
                    "payload": {
                        "label": node["label"],
                        "reveal_node_ids": [node["id"]],
                        "reveal_edge_ids": [],
                    },
                },
            )
        )
    events = []
    for sequence, (_, _, _, event) in enumerate(sorted(pending), 1):
        events.append(
            {
                "schema_version": 1,
                "event_id": f"vision-event-{sequence:06d}",
                "sequence": sequence,
                **event,
            }
        )
    return events


class VisionReplay:
    """A monotonic-clock replay controller with a bounded API event window."""

    def __init__(
        self,
        graph: dict,
        *,
        speed: float = 1.0,
        start_ms: int = 0,
        event_window: int = 100,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.1 <= speed <= 100:
            raise ValueError("replay speed must be between 0.1 and 100")
        if not 1 <= event_window <= 1000:
            raise ValueError("event window must be between 1 and 1000")
        self.graph = graph
        self.events = graph_events(graph)
        self._timestamps = [event["timestamp_ms"] for event in self.events]
        self.duration_ms = max(
            [0]
            + [int(edge.get("end_ms", 0)) for edge in graph["edges"]]
            + [int(node.get("attributes", {}).get("end_ms", 0)) for node in graph["nodes"]]
        )
        if not 0 <= start_ms <= self.duration_ms:
            raise ValueError("replay start must be inside the graph timeline")
        self.speed = float(speed)
        self.event_window = event_window
        self._clock = clock
        self._lock = Lock()
        self._anchor_clock = clock()
        self._anchor_ms = float(start_ms)
        self._playing = start_ms < self.duration_ms
        self._node_order = [node["id"] for node in graph["nodes"]]
        self._edge_order = [edge["id"] for edge in graph["edges"]]
        self._scene_ranges = sorted(
            (
                int(node.get("attributes", {}).get("start_ms", 0)),
                int(node.get("attributes", {}).get("end_ms", 0)),
                node["id"],
            )
            for node in graph["nodes"]
            if node["type"] == "scene"
        )

    def _position(self, now: float) -> int:
        elapsed = (now - self._anchor_clock) * 1000 * self.speed if self._playing else 0
        return min(self.duration_ms, max(0, round(self._anchor_ms + elapsed)))

    def snapshot(self) -> dict:
        with self._lock:
            current_ms = self._position(self._clock())
            emitted = bisect_right(self._timestamps, current_ms)
            recent = self.events[max(0, emitted - self.event_window) : emitted]
            visible_nodes = set()
            visible_edges = set()
            for event in self.events[:emitted]:
                visible_nodes.update(event["payload"]["reveal_node_ids"])
                visible_edges.update(event["payload"]["reveal_edge_ids"])
            current_scene_id: Optional[str] = None
            for start, end, scene_id in self._scene_ranges:
                if start <= current_ms < end or current_ms == self.duration_ms == end:
                    current_scene_id = scene_id
            status = (
                "complete"
                if current_ms >= self.duration_ms
                else "playing"
                if self._playing
                else "paused"
            )
            return {
                "schema_version": 1,
                "status": status,
                "current_ms": current_ms,
                "duration_ms": self.duration_ms,
                "speed": self.speed,
                "events_emitted": emitted,
                "events_total": len(self.events),
                "event_window": self.event_window,
                "current_scene_id": current_scene_id,
                "visible_node_ids": [item for item in self._node_order if item in visible_nodes],
                "visible_edge_ids": [item for item in self._edge_order if item in visible_edges],
                "events": recent,
            }

    def control(self, action: str, timestamp_ms: Optional[int] = None) -> dict:
        with self._lock:
            now = self._clock()
            current_ms = self._position(now)
            if action == "pause":
                self._anchor_ms = current_ms
                self._anchor_clock = now
                self._playing = False
            elif action == "resume":
                self._anchor_ms = current_ms
                self._anchor_clock = now
                self._playing = current_ms < self.duration_ms
            elif action == "restart":
                self._anchor_ms = 0
                self._anchor_clock = now
                self._playing = self.duration_ms > 0
            elif action == "seek":
                if timestamp_ms is None:
                    raise ValueError("seek requires timestamp_ms")
                if not 0 <= timestamp_ms <= self.duration_ms:
                    raise ValueError("seek timestamp must be inside the graph timeline")
                self._anchor_ms = timestamp_ms
                self._anchor_clock = now
                self._playing = timestamp_ms < self.duration_ms and self._playing
            else:
                raise ValueError("unknown replay action: " + action)
        return self.snapshot()
