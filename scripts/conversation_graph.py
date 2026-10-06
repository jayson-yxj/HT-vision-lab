from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .models import file_sha256
from .scene_graph import STRUCTURAL_PREDICATES, validate_scene_graph


NODE_TYPES = {
    "person",
    "visual_identity",
    "scene",
    "object",
    "topic",
    "opinion",
    "intent",
}
DIALOGUE_PREDICATES = {
    "expresses",
    "about",
    "has_intent",
    "occurred_in",
    "discussed_in",
}
MIN_SCENE_OVERLAP_MS = 120


def _source_file(path: Path) -> dict:
    return {"path": str(path), "sha256": file_sha256(path)}


def _milliseconds(value: object) -> int:
    return int(round(float(value) * 1000))


def _append_node(nodes: List[dict], node: dict) -> None:
    if any(existing["id"] == node["id"] for existing in nodes):
        raise ValueError("duplicate graph node: " + node["id"])
    nodes.append(node)


def _append_edge(
    edges: List[dict],
    source: str,
    target: str,
    predicate: str,
    start_ms: int,
    end_ms: int,
    evidence_refs: Sequence[str],
    attributes: Optional[dict] = None,
) -> None:
    edges.append(
        {
            "id": f"edge-{len(edges) + 1:06d}",
            "source": source,
            "target": target,
            "predicate": predicate,
            "epistemic_status": "inferred",
            "confidence": 1.0,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "evidence_refs": list(
                dict.fromkeys(str(value) for value in evidence_refs if str(value))
            ),
            "attributes": attributes or {},
        }
    )


def validate_session_memory(data: dict) -> List[str]:
    if not isinstance(data, dict):
        return ["session memory must be an object"]
    errors: List[str] = []
    if data.get("schema_version") != 1:
        errors.append("only session memory schema_version=1 is supported")
    if not isinstance(data.get("session_id"), str) or not data["session_id"].strip():
        errors.append("session memory requires a non-empty session_id")
    participants = data.get("participants")
    if not isinstance(participants, list) or any(
        not isinstance(value, str) or not value.strip() for value in participants
    ):
        errors.append("session memory participants must be an array of non-empty strings")
    turns = data.get("turns")
    if not isinstance(turns, list):
        return errors + ["session memory turns must be an array"]
    turn_ids = []
    for index, turn in enumerate(turns):
        prefix = f"turns[{index}]"
        if not isinstance(turn, dict):
            errors.append(prefix + " must be an object")
            continue
        identifier = turn.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            errors.append(prefix + " requires a non-empty id")
        else:
            turn_ids.append(identifier)
        if not isinstance(turn.get("speaker"), str) or not turn["speaker"].strip():
            errors.append(prefix + " requires a non-empty speaker")
        try:
            start = float(turn.get("start_s"))
            end = float(turn.get("end_s"))
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                errors.append(prefix + " has an invalid time range")
        except (TypeError, ValueError):
            errors.append(prefix + " requires numeric start_s and end_s")
        for name in ("contribution_ids", "topic_names", "entities"):
            value = turn.get(name)
            if not isinstance(value, list):
                errors.append(prefix + f".{name} must be an array")
    if len(turn_ids) != len(set(turn_ids)):
        errors.append("session memory contains duplicate turn IDs")
    scene = data.get("scene")
    if not isinstance(scene, dict):
        errors.append("session memory scene must be an object")
    elif not isinstance(scene.get("topics", []), list):
        errors.append("session memory scene.topics must be an array")
    return errors


def _load_participant_context(
    scene_graph: dict, memory_path: Path, memory: dict
) -> Tuple[Path, dict]:
    source = scene_graph.get("source", {}).get("participant_context")
    if not isinstance(source, dict):
        raise ValueError(
            "scene graph must be built with --participant-context before conversation fusion"
        )
    path = Path(source.get("path", "")).expanduser().resolve()
    if not path.is_file() or file_sha256(path) != source.get("sha256"):
        raise ValueError("scene graph participant_context source is unavailable or has changed")
    context = json.loads(path.read_text(encoding="utf-8"))
    if context.get("session_id") != memory["session_id"]:
        raise ValueError("participant context and session memory belong to different sessions")
    upstream = context.get("source", {}).get("upstream", {}).get("session_memory", {})
    if upstream.get("sha256") != file_sha256(memory_path):
        raise ValueError(
            "participant context does not reference this exact session_memory.json"
        )
    return path, context


def _topic_catalog(memory: dict) -> Tuple[List[dict], Dict[str, List[str]]]:
    catalog: Dict[str, dict] = {}
    evidence_topics: Dict[str, List[str]] = {}
    for item in memory.get("scene", {}).get("topics", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        evidence = [
            str(value)
            for value in item.get("evidence_turn_ids", [])
            if str(value).strip()
        ]
        catalog.setdefault(
            name,
            {
                "name": name,
                "summary": str(item.get("summary") or "").strip(),
                "evidence_turn_ids": [],
            },
        )
        catalog[name]["evidence_turn_ids"] = list(
            dict.fromkeys(catalog[name]["evidence_turn_ids"] + evidence)
        )
        for turn_id in evidence:
            evidence_topics.setdefault(turn_id, []).append(name)
    for turn in memory["turns"]:
        names = [
            str(value).strip()
            for value in turn.get("topic_names", [])
            if str(value).strip()
        ] or evidence_topics.get(turn["id"], []) or ["未归类话题"]
        for name in names:
            topic = catalog.setdefault(
                name, {"name": name, "summary": "", "evidence_turn_ids": []}
            )
            if turn["id"] not in topic["evidence_turn_ids"]:
                topic["evidence_turn_ids"].append(turn["id"])
    return list(catalog.values()), evidence_topics


def _turn_topics(turn: dict, evidence_topics: Dict[str, List[str]]) -> List[str]:
    names = [
        str(value).strip()
        for value in turn.get("topic_names", [])
        if str(value).strip()
    ]
    return list(
        dict.fromkeys(names or evidence_topics.get(turn["id"], []) or ["未归类话题"])
    )


def _node_time_range(
    turn_ids: Sequence[str], turns_by_id: Dict[str, dict]
) -> Tuple[Optional[int], Optional[int]]:
    turns = [turns_by_id[identifier] for identifier in turn_ids if identifier in turns_by_id]
    if not turns:
        return None, None
    return min(_milliseconds(item["start_s"]) for item in turns), max(
        _milliseconds(item["end_s"]) for item in turns
    )


def project_conversation_graph(
    scene_graph_path: Path,
    session_memory_path: Path,
    output_path: Optional[Path] = None,
) -> dict:
    scene_graph_path = scene_graph_path.expanduser().resolve()
    session_memory_path = session_memory_path.expanduser().resolve()
    scene_graph = json.loads(scene_graph_path.read_text(encoding="utf-8"))
    graph_errors = validate_scene_graph(scene_graph)
    if graph_errors:
        raise ValueError("invalid scene graph: " + "; ".join(graph_errors))
    if scene_graph.get("context_type") != "multimodal_scene_graph":
        raise ValueError("input must be a multimodal_scene_graph")
    memory = json.loads(session_memory_path.read_text(encoding="utf-8"))
    memory_errors = validate_session_memory(memory)
    if memory_errors:
        raise ValueError("invalid session memory: " + "; ".join(memory_errors))
    participant_path, _ = _load_participant_context(
        scene_graph, session_memory_path, memory
    )
    output_path = (
        output_path.expanduser().resolve()
        if output_path
        else scene_graph_path.parent / "multimodal_conversation_graph.json"
    )
    if output_path in {scene_graph_path, session_memory_path, participant_path}:
        raise ValueError("conversation graph output must not overwrite a source file")

    nodes = copy.deepcopy(scene_graph["nodes"])
    edges = copy.deepcopy(scene_graph["edges"])
    people_by_speaker = {
        node.get("attributes", {}).get("speaker_label"): node["id"]
        for node in nodes
        if node["type"] == "person" and node.get("attributes", {}).get("speaker_label")
    }
    speakers = list(
        dict.fromkeys(
            [*memory["participants"], *(turn["speaker"] for turn in memory["turns"])]
        )
    )
    missing_speakers = [speaker for speaker in speakers if speaker not in people_by_speaker]
    if missing_speakers:
        raise ValueError(
            "scene graph is missing voice participants: " + ", ".join(missing_speakers)
        )

    turns_by_id = {turn["id"]: turn for turn in memory["turns"]}
    topics, evidence_topics = _topic_catalog(memory)
    topic_ids: Dict[str, str] = {}
    for index, topic in enumerate(topics, start=1):
        identifier = f"topic:{index:04d}"
        topic_ids[topic["name"]] = identifier
        evidence_ids = list(dict.fromkeys(topic["evidence_turn_ids"]))
        start_ms, end_ms = _node_time_range(evidence_ids, turns_by_id)
        _append_node(
            nodes,
            {
                "id": identifier,
                "type": "topic",
                "label": topic["name"],
                "epistemic_status": "inferred",
                "confidence": None,
                "evidence_refs": evidence_ids,
                "attributes": {
                    "summary": topic["summary"],
                    "evidence_turn_ids": evidence_ids,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                },
            },
        )

    scenes = [node for node in nodes if node["type"] == "scene"]
    intent_ids: Dict[str, str] = {}
    topic_scene: Dict[Tuple[str, str], dict] = {}
    linked_turn_ids = set()
    for turn in memory["turns"]:
        turn_id = turn["id"]
        start_ms = _milliseconds(turn["start_s"])
        end_ms = _milliseconds(turn["end_s"])
        opinion_id = "opinion:" + turn_id
        contribution_ids = [str(value) for value in turn.get("contribution_ids", [])]
        _append_node(
            nodes,
            {
                "id": opinion_id,
                "type": "opinion",
                "label": str(turn.get("summary") or "观点待识别").strip(),
                "epistemic_status": "inferred",
                "confidence": None,
                "evidence_refs": [turn_id, *contribution_ids],
                "attributes": {
                    "turn_id": turn_id,
                    "speaker_label": turn["speaker"],
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": end_ms - start_ms,
                    "speech_act": turn.get("speech_act") or "other",
                    "stance": turn.get("stance") or "unclear",
                    "entities": turn.get("entities", []),
                    "contribution_ids": contribution_ids,
                    "reply_requested": bool(turn.get("reply_requested", False)),
                    "source_text_sha256": turn.get("source_text_sha256"),
                },
            },
        )
        _append_edge(
            edges,
            people_by_speaker[turn["speaker"]],
            opinion_id,
            "expresses",
            start_ms,
            end_ms,
            [turn_id, *contribution_ids],
            {"turn_id": turn_id},
        )

        turn_topic_names = _turn_topics(turn, evidence_topics)
        for name in turn_topic_names:
            topic_id = topic_ids[name]
            _append_edge(
                edges,
                topic_id,
                opinion_id,
                "about",
                start_ms,
                end_ms,
                [turn_id],
                {"turn_id": turn_id},
            )

        intent = str(turn.get("intent") or "").strip()
        if intent:
            if intent not in intent_ids:
                intent_id = f"intent:{len(intent_ids) + 1:04d}"
                intent_ids[intent] = intent_id
                _append_node(
                    nodes,
                    {
                        "id": intent_id,
                        "type": "intent",
                        "label": intent,
                        "epistemic_status": "inferred",
                        "confidence": None,
                        "evidence_refs": [],
                        "attributes": {
                            "speaker_labels": [],
                            "evidence_turn_ids": [],
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                        },
                    },
                )
            intent_id = intent_ids[intent]
            intent_node = next(node for node in nodes if node["id"] == intent_id)
            intent_node["evidence_refs"].append(turn_id)
            intent_node["attributes"]["evidence_turn_ids"].append(turn_id)
            if turn["speaker"] not in intent_node["attributes"]["speaker_labels"]:
                intent_node["attributes"]["speaker_labels"].append(turn["speaker"])
            intent_node["attributes"]["start_ms"] = min(
                intent_node["attributes"]["start_ms"], start_ms
            )
            intent_node["attributes"]["end_ms"] = max(
                intent_node["attributes"]["end_ms"], end_ms
            )
            _append_edge(
                edges,
                opinion_id,
                intent_id,
                "has_intent",
                start_ms,
                end_ms,
                [turn_id],
                {"turn_id": turn_id},
            )

        for scene in scenes:
            scene_start = int(scene["attributes"]["start_ms"])
            scene_end = int(scene["attributes"]["end_ms"])
            overlap_start = max(start_ms, scene_start)
            overlap_end = min(end_ms, scene_end)
            if overlap_end - overlap_start < MIN_SCENE_OVERLAP_MS:
                continue
            linked_turn_ids.add(turn_id)
            overlap_ms = overlap_end - overlap_start
            turn_duration = max(1, end_ms - start_ms)
            scene_duration = max(1, scene_end - scene_start)
            _append_edge(
                edges,
                opinion_id,
                scene["id"],
                "occurred_in",
                overlap_start,
                overlap_end,
                [turn_id, *scene["evidence_refs"]],
                {
                    "turn_id": turn_id,
                    "overlap_ms": overlap_ms,
                    "turn_coverage": round(overlap_ms / turn_duration, 6),
                    "scene_coverage": round(overlap_ms / scene_duration, 6),
                },
            )
            for name in turn_topic_names:
                accumulator = topic_scene.setdefault(
                    (topic_ids[name], scene["id"]),
                    {
                        "turn_ids": [],
                        "start_ms": overlap_start,
                        "end_ms": overlap_end,
                        "overlap_ms": 0,
                    },
                )
                if turn_id not in accumulator["turn_ids"]:
                    accumulator["turn_ids"].append(turn_id)
                accumulator["start_ms"] = min(accumulator["start_ms"], overlap_start)
                accumulator["end_ms"] = max(accumulator["end_ms"], overlap_end)
                accumulator["overlap_ms"] += overlap_ms

    for (topic_id, scene_id), value in topic_scene.items():
        _append_edge(
            edges,
            topic_id,
            scene_id,
            "discussed_in",
            value["start_ms"],
            value["end_ms"],
            value["turn_ids"],
            {"turn_ids": value["turn_ids"], "overlap_ms": value["overlap_ms"]},
        )

    unlinked_turns = [
        turn["id"] for turn in memory["turns"] if turn["id"] not in linked_turn_ids
    ]
    scene_ranges = [
        (node["attributes"]["start_ms"], node["attributes"]["end_ms"])
        for node in scenes
    ]
    coverage_start = min((value[0] for value in scene_ranges), default=None)
    coverage_end = max((value[1] for value in scene_ranges), default=None)
    data = {
        "schema_version": 1,
        "context_type": "multimodal_conversation_graph",
        "session_id": memory["session_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "scene_graph": _source_file(scene_graph_path),
            "session_memory": _source_file(session_memory_path),
            "scene_context": copy.deepcopy(scene_graph["source"]["scene_context"]),
            "participant_context": _source_file(participant_path),
        },
        "processing": {
            "speaker_alignment": "participant_context_session_id_and_hash_v1",
            "temporal_alignment": {
                "algorithm": "minimum_interval_overlap_v1",
                "minimum_overlap_ms": MIN_SCENE_OVERLAP_MS,
                "scene_coverage_start_ms": coverage_start,
                "scene_coverage_end_ms": coverage_end,
                "linked_turn_ids": sorted(linked_turn_ids),
                "unlinked_turn_ids": unlinked_turns,
            },
            "dialogue_semantics_are_inferred": True,
        },
        "nodes": nodes,
        "edges": edges,
        "statistics": {},
        "warnings": (
            [
                f"{len(unlinked_turns)} dialogue turns do not overlap "
                "the available visual scene range"
            ]
            if unlinked_turns
            else []
        ),
    }
    data["statistics"] = _statistics(data)
    errors = validate_conversation_graph(data)
    if errors:
        raise RuntimeError("conversation graph validation failed: " + "; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return data


def _statistics(data: dict) -> dict:
    nodes = data["nodes"]
    edges = data["edges"]
    linked_turns = {
        edge["source"] for edge in edges if edge.get("predicate") == "occurred_in"
    }
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "persons": sum(node.get("type") == "person" for node in nodes),
        "visual_identities": sum(
            node.get("type") == "visual_identity" for node in nodes
        ),
        "scenes": sum(node.get("type") == "scene" for node in nodes),
        "objects": sum(node.get("type") == "object" for node in nodes),
        "topics": sum(node.get("type") == "topic" for node in nodes),
        "opinions": sum(node.get("type") == "opinion" for node in nodes),
        "intents": sum(node.get("type") == "intent" for node in nodes),
        "visual_interactions": sum(
            edge.get("predicate") not in STRUCTURAL_PREDICATES | DIALOGUE_PREDICATES
            for edge in edges
        ),
        "dialogue_relations": sum(
            edge.get("predicate") in DIALOGUE_PREDICATES for edge in edges
        ),
        "temporally_linked_turns": len(linked_turns),
        "unlinked_turns": sum(node.get("type") == "opinion" for node in nodes)
        - len(linked_turns),
        "observed_edges": sum(
            edge.get("epistemic_status") == "observed" for edge in edges
        ),
        "inferred_edges": sum(
            edge.get("epistemic_status") == "inferred" for edge in edges
        ),
    }


def validate_conversation_graph(data: dict) -> List[str]:
    required = {
        "schema_version",
        "context_type",
        "session_id",
        "created_at",
        "source",
        "processing",
        "nodes",
        "edges",
        "statistics",
        "warnings",
    }
    if not isinstance(data, dict):
        return ["conversation graph must be an object"]
    missing = required - set(data)
    if missing:
        return ["missing top-level fields: " + ", ".join(sorted(missing))]
    errors: List[str] = []
    if data.get("schema_version") != 1:
        errors.append("only schema_version=1 is supported")
    if data.get("context_type") != "multimodal_conversation_graph":
        errors.append("context_type must be multimodal_conversation_graph")
    if not isinstance(data.get("nodes"), list) or not isinstance(data.get("edges"), list):
        return errors + ["nodes and edges must be arrays"]
    node_ids = [node.get("id") for node in data["nodes"] if isinstance(node, dict)]
    edge_ids = [edge.get("id") for edge in data["edges"] if isinstance(edge, dict)]
    if len(node_ids) != len(data["nodes"]):
        errors.append("every node must be an object")
    if len(edge_ids) != len(data["edges"]):
        errors.append("every edge must be an object")
    if len(node_ids) != len(data["nodes"]) or len(edge_ids) != len(data["edges"]):
        return errors
    if len(node_ids) != len(set(node_ids)):
        errors.append("duplicate node IDs")
    if len(edge_ids) != len(set(edge_ids)):
        errors.append("duplicate edge IDs")
    known_nodes = set(node_ids)
    node_types = {
        node.get("id"): node.get("type")
        for node in data["nodes"]
        if isinstance(node, dict)
    }
    for node in data["nodes"]:
        if not isinstance(node, dict):
            continue
        if node.get("type") not in NODE_TYPES:
            errors.append(f"{node.get('id')} has an invalid node type")
        if node.get("epistemic_status") not in {"observed", "inferred"}:
            errors.append(f"{node.get('id')} has an invalid epistemic status")
        confidence = node.get("confidence")
        if confidence is not None and (
            not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1
        ):
            errors.append(f"{node.get('id')} has invalid confidence")
    expresses = []
    for edge in data["edges"]:
        if not isinstance(edge, dict):
            continue
        identifier = edge.get("id")
        if edge.get("source") not in known_nodes or edge.get("target") not in known_nodes:
            errors.append(f"{identifier} references an unknown node")
        if edge.get("epistemic_status") not in {"observed", "inferred"}:
            errors.append(f"{identifier} has an invalid epistemic status")
        confidence = edge.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append(f"{identifier} has invalid confidence")
        start = edge.get("start_ms")
        end = edge.get("end_ms")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            errors.append(f"{identifier} has an invalid time range")
        predicate = edge.get("predicate")
        source_type = node_types.get(edge.get("source"))
        target_type = node_types.get(edge.get("target"))
        expected_types = {
            "expresses": ("person", "opinion"),
            "about": ("topic", "opinion"),
            "has_intent": ("opinion", "intent"),
            "occurred_in": ("opinion", "scene"),
            "discussed_in": ("topic", "scene"),
        }
        if predicate in expected_types and (source_type, target_type) != expected_types[predicate]:
            errors.append(f"{identifier} has invalid endpoints for {predicate}")
        if predicate == "expresses":
            expresses.append(edge.get("target"))
    opinion_ids = [
        node["id"]
        for node in data["nodes"]
        if isinstance(node, dict) and node.get("type") == "opinion"
    ]
    if sorted(expresses) != sorted(opinion_ids):
        errors.append("every opinion must have exactly one expresses edge")
    if data.get("statistics") != _statistics(data):
        errors.append("statistics do not match graph contents")
    return errors
