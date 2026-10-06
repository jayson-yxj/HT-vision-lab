from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .models import file_sha256
from .multimodal_context import validate_multimodal_context
from .scene_context import validate_scene_context
from .scene_semantics import validate_scene_semantics


MERGER_ALGORITHM = "adjacent_semantic_similarity_v1"
DESCRIPTION_WEIGHT = 0.55
OBJECT_WEIGHT = 0.30
PERSON_WEIGHT = 0.15
NODE_TYPES = {"person", "visual_identity", "scene", "object"}
STRUCTURAL_PREDICATES = {"present_in", "contains", "followed_by", "identity_candidate"}


def _source_file(path: Path) -> dict:
    return {"path": str(path), "sha256": file_sha256(path)}


def _jaccard(left: Set[str], right: Set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _character_bigrams(value: str) -> Set[str]:
    compact = "".join(character.lower() for character in value if character.isalnum())
    if not compact:
        return set()
    if len(compact) == 1:
        return {compact}
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def _object_key(label: str) -> str:
    return "_".join(label.strip().lower().split())


def _unit(analysis: dict, states: Dict[str, dict], keyframes: Dict[str, dict], shots: Dict[str, dict]) -> dict:
    keyframe = keyframes[analysis["keyframe_id"]]
    shot = shots[analysis["shot_id"]]
    face_ids = {
        states[state_id]["face_id"] for state_id in keyframe["person_state_ids"]
    }
    return {
        "analysis": analysis,
        "keyframe": keyframe,
        "shot": shot,
        "description_terms": _character_bigrams(analysis["environment"]["description"]),
        "object_terms": {_object_key(item["label"]) for item in analysis["objects"]},
        "face_ids": face_ids,
    }


def _similarity(left: dict, right: dict) -> float:
    if left["analysis"]["environment"]["category"] != right["analysis"]["environment"]["category"]:
        return 0.0
    return (
        DESCRIPTION_WEIGHT * _jaccard(left["description_terms"], right["description_terms"])
        + OBJECT_WEIGHT * _jaccard(left["object_terms"], right["object_terms"])
        + PERSON_WEIGHT * _jaccard(left["face_ids"], right["face_ids"])
    )


def _group_units(units: Sequence[dict], threshold: float) -> List[List[dict]]:
    groups: List[List[dict]] = []
    for unit in units:
        same_category = bool(groups) and (
            groups[-1][-1]["analysis"]["environment"]["category"]
            == unit["analysis"]["environment"]["category"]
        )
        if not same_category or _similarity(groups[-1][-1], unit) < threshold:
            groups.append([unit])
        else:
            groups[-1].append(unit)
    return groups


def _participant_context(path: Optional[Path], expected_visual_hash: str) -> Tuple[dict, Optional[dict]]:
    if path is None:
        return {}, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_multimodal_context(payload)
    if errors:
        raise ValueError("invalid multimodal participant context: " + "; ".join(errors))
    visual_source = payload.get("source", {}).get("visual_tracks", {})
    if visual_source.get("sha256") != expected_visual_hash:
        raise ValueError("participant context and scene semantics do not reference the same visual tracks")
    return {item["speaker_label"]: item for item in payload["participants"]}, payload


def _binding_confidences(visual_path: Path, expected_hash: str) -> Dict[Tuple[str, str], float]:
    if not visual_path.is_file() or file_sha256(visual_path) != expected_hash:
        raise ValueError("scene graph visual_tracks source is unavailable or has changed")
    visual = json.loads(visual_path.read_text(encoding="utf-8"))
    return {
        (item["speaker_label"], item["face_id"]): float(item["confidence"])
        for item in visual.get("speaker_face_associations", [])
        if item.get("status") == "confirmed" and item.get("face_id")
    }


def _append_node(nodes: List[dict], node: dict) -> None:
    if any(existing["id"] == node["id"] for existing in nodes):
        raise ValueError("duplicate graph node: " + node["id"])
    nodes.append(node)


def _append_edge(
    edges: List[dict],
    source: str,
    target: str,
    predicate: str,
    status: str,
    confidence: float,
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
            "epistemic_status": status,
            "confidence": round(max(0.0, min(1.0, confidence)), 6),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "evidence_refs": list(dict.fromkeys(evidence_refs)),
            "attributes": attributes or {},
        }
    )


def project_scene_graph(
    semantics_path: Path,
    participant_context_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    similarity_threshold: float = 0.20,
) -> dict:
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("scene similarity threshold must be between zero and one")
    semantics_path = semantics_path.expanduser().resolve()
    semantics = json.loads(semantics_path.read_text(encoding="utf-8"))
    if semantics.get("status") != "complete":
        raise ValueError("scene semantics must be complete before building the graph")
    semantic_errors = validate_scene_semantics(semantics)
    if semantic_errors:
        raise ValueError("invalid scene semantics: " + "; ".join(semantic_errors))

    scene_path = Path(semantics["source"]["scene_context_path"]).expanduser().resolve()
    if not scene_path.is_file() or file_sha256(scene_path) != semantics["source"]["scene_context_sha256"]:
        raise ValueError("scene_context source is unavailable or has changed")
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene_errors = validate_scene_context(scene)
    if scene_errors:
        raise ValueError("invalid scene context: " + "; ".join(scene_errors))
    if scene["session_id"] != semantics["session_id"]:
        raise ValueError("scene context and semantics session IDs do not match")

    participant_context_path = (
        participant_context_path.expanduser().resolve() if participant_context_path else None
    )
    participants, participant_context = _participant_context(
        participant_context_path, semantics["source"]["visual_tracks_sha256"]
    )
    visual_path = Path(scene["source"]["visual_tracks_path"]).expanduser().resolve()
    binding_confidences = _binding_confidences(
        visual_path, scene["source"]["visual_tracks_sha256"]
    )
    output_path = (
        output_path.expanduser().resolve()
        if output_path
        else semantics_path.parent / "multimodal_scene_graph.json"
    )
    protected = {semantics_path, scene_path, visual_path}
    if participant_context_path:
        protected.add(participant_context_path)
    if output_path in protected:
        raise ValueError("scene graph output must not overwrite a source file")

    states = {item["person_state_id"]: item for item in scene["person_states"]}
    keyframes = {item["keyframe_id"]: item for item in scene["keyframes"]}
    shots = {item["shot_id"]: item for item in scene["shots"]}
    analyses = sorted(
        semantics["analyses"], key=lambda item: (shots[item["shot_id"]]["start_ms"], item["keyframe_id"])
    )
    units = [_unit(item, states, keyframes, shots) for item in analyses]
    groups = _group_units(units, similarity_threshold)

    nodes: List[dict] = []
    edges: List[dict] = []
    person_by_face: Dict[str, str] = {}
    known_speakers = set()
    for entity in sorted(scene["visual_entities"], key=lambda item: item["face_id"]):
        face_id = entity["face_id"]
        speaker = entity["speaker_label"]
        participant = participants.get(speaker) if speaker else None
        node_id = (
            f"person:speaker:{speaker}"
            if speaker
            else f"visual_identity:{face_id}"
        )
        if speaker:
            known_speakers.add(speaker)
        person_by_face[face_id] = node_id
        _append_node(
            nodes,
            {
                "id": node_id,
                "type": "person" if speaker else "visual_identity",
                "label": (participant or {}).get("display_name") or speaker or face_id,
                "epistemic_status": "observed",
                "confidence": binding_confidences.get((speaker, face_id)),
                "evidence_refs": [face_id],
                "attributes": {
                    "face_id": face_id,
                    "speaker_label": speaker,
                    "participant_id": (participant or {}).get("participant_id"),
                    "display_name": (participant or {}).get("display_name"),
                    "identity_status": entity["identity_status"],
                },
            },
        )
    for speaker, participant in sorted(participants.items()):
        if speaker in known_speakers:
            continue
        _append_node(
            nodes,
            {
                "id": f"person:speaker:{speaker}",
                "type": "person",
                "label": participant.get("display_name") or speaker,
                "epistemic_status": "observed",
                "confidence": None,
                "evidence_refs": list(participant.get("evidence_ids", [])),
                "attributes": {
                    "face_id": None,
                    "speaker_label": speaker,
                    "participant_id": participant["participant_id"],
                    "display_name": participant.get("display_name"),
                    "identity_status": "unbound",
                },
            },
        )

    if participant_context:
        speaker_by_participant = {
            item["participant_id"]: item["speaker_label"]
            for item in participant_context["participants"]
        }
        evidence_by_id = {
            item["evidence_id"]: item for item in participant_context["evidence"]
        }
        for association in participant_context["associations"]:
            if (
                association.get("relation") != "same_session_identity"
                or association.get("state") != "disputed"
            ):
                continue
            speaker = speaker_by_participant.get(association["left_ref"])
            face_id = association["right_ref"].rsplit(":", 1)[-1]
            source = f"person:speaker:{speaker}" if speaker else None
            target = person_by_face.get(face_id)
            if source not in {item["id"] for item in nodes} or target is None:
                raise ValueError("disputed participant association references an unknown person")
            association_evidence = [
                evidence_by_id[identifier]
                for identifier in association["evidence_ids"]
                if identifier in evidence_by_id
            ]
            starts = [item["start_s"] for item in association_evidence if item["start_s"] is not None]
            ends = [item["end_s"] for item in association_evidence if item["end_s"] is not None]
            association_start_ms = int(round(min(starts) * 1000)) if starts else 0
            association_end_ms = (
                int(round(max(ends) * 1000)) if ends else association_start_ms
            )
            _append_edge(
                edges,
                source,
                target,
                "identity_candidate",
                "inferred",
                association["confidence"],
                association_start_ms,
                association_end_ms,
                [association["association_id"]] + association["evidence_ids"],
                {
                    "association_state": "disputed",
                    "source_binding_status": association.get("source_binding_status"),
                    "speaker_coverage": association.get("speaker_coverage"),
                    "evidence_duration_ms": association.get("evidence_duration_ms"),
                },
            )

    scene_ids = []
    interaction_edges = 0
    for event_index, group in enumerate(groups, start=1):
        scene_id = f"scene:{event_index:04d}"
        scene_ids.append(scene_id)
        event_analyses = [unit["analysis"] for unit in group]
        start_ms = min(unit["shot"]["start_ms"] for unit in group)
        end_ms = max(unit["shot"]["end_ms"] for unit in group)
        descriptions = list(
            dict.fromkeys(item["environment"]["description"] for item in event_analyses)
        )
        environment_confidence = sum(
            item["environment"]["confidence"] for item in event_analyses
        ) / len(event_analyses)
        _append_node(
            nodes,
            {
                "id": scene_id,
                "type": "scene",
                "label": descriptions[0],
                "epistemic_status": "inferred",
                "confidence": round(environment_confidence, 6),
                "evidence_refs": [item["analysis_id"] for item in event_analyses],
                "attributes": {
                    "environment_category": event_analyses[0]["environment"]["category"],
                    "descriptions": descriptions,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": end_ms - start_ms,
                    "shot_ids": [unit["shot"]["shot_id"] for unit in group],
                    "keyframe_ids": [unit["keyframe"]["keyframe_id"] for unit in group],
                },
            },
        )

        face_evidence: Dict[str, List[Tuple[dict, dict]]] = {}
        for unit in group:
            for state_id in unit["keyframe"]["person_state_ids"]:
                state = states[state_id]
                face_evidence.setdefault(state["face_id"], []).append((state, unit["keyframe"]))
        for face_id, evidence in sorted(face_evidence.items()):
            _append_edge(
                edges,
                person_by_face[face_id],
                scene_id,
                "present_in",
                "observed",
                sum(item[0]["detection_confidence"] for item in evidence) / len(evidence),
                start_ms,
                end_ms,
                [value for state, frame in evidence for value in (state["person_state_id"], frame["keyframe_id"])],
            )

        object_values: Dict[str, List[dict]] = {}
        synthetic_object_evidence: Dict[str, List[dict]] = {}
        for analysis in event_analyses:
            for item in analysis["objects"]:
                object_values.setdefault(_object_key(item["label"]), []).append(item)
            for interaction in analysis["interactions"]:
                target = interaction.get("object_ref")
                if target and target.startswith("object:"):
                    key = _object_key(target.split(":", 1)[1])
                    object_values.setdefault(key, [])
                    synthetic_object_evidence.setdefault(key, []).append(interaction)
        object_nodes = {}
        for object_index, (key, values) in enumerate(sorted(object_values.items()), start=1):
            object_id = f"{scene_id}:object:{object_index:03d}"
            object_nodes[key] = object_id
            supporting_interactions = synthetic_object_evidence.get(key, [])
            confidence_values = [item["confidence"] for item in values] or [
                item["confidence"] for item in supporting_interactions
            ]
            confidence = sum(confidence_values) / len(confidence_values)
            label = values[0]["label"] if values else key
            evidence_refs = [item["semantic_object_id"] for item in values] + [
                item["interaction_id"] for item in supporting_interactions
            ]
            _append_node(
                nodes,
                {
                    "id": object_id,
                    "type": "object",
                    "label": label,
                    "epistemic_status": "inferred",
                    "confidence": round(confidence, 6),
                    "evidence_refs": evidence_refs,
                    "attributes": {
                        "scene_id": scene_id,
                        "count": max((item["count"] for item in values), default=1),
                        "regions": sorted({item["region"] for item in values}) or ["unknown"],
                    },
                },
            )
            _append_edge(
                edges,
                scene_id,
                object_id,
                "contains",
                "inferred",
                confidence,
                start_ms,
                end_ms,
                evidence_refs,
            )

        grouped_interactions: Dict[Tuple[str, str, Optional[str]], List[Tuple[dict, dict]]] = {}
        for unit in group:
            analysis = unit["analysis"]
            for item in analysis["interactions"]:
                key = (item["subject_ref"], item["predicate"], item.get("object_ref"))
                grouped_interactions.setdefault(key, []).append((item, unit["shot"]))
        for (subject_ref, predicate, object_ref), values in sorted(
            grouped_interactions.items(), key=lambda item: tuple(str(value) for value in item[0])
        ):
            if object_ref and object_ref.startswith("Face-"):
                target = person_by_face[object_ref]
            elif object_ref and object_ref.startswith("object:"):
                target = object_nodes[_object_key(object_ref.split(":", 1)[1])]
            else:
                target = scene_id
            _append_edge(
                edges,
                person_by_face[subject_ref],
                target,
                predicate,
                "inferred",
                sum(item[0]["confidence"] for item in values) / len(values),
                min(item[1]["start_ms"] for item in values),
                max(item[1]["end_ms"] for item in values),
                [item[0]["interaction_id"] for item in values],
                {
                    "object_scope": object_ref,
                    "descriptions": list(
                        dict.fromkeys(item[0]["description"] for item in values)
                    ),
                    "evidence_basis": sorted(
                        {basis for item in values for basis in item[0]["evidence_basis"]}
                    ),
                },
            )
            interaction_edges += 1

    scene_nodes = {item["id"]: item for item in nodes if item["type"] == "scene"}
    for left, right in zip(scene_ids, scene_ids[1:]):
        _append_edge(
            edges,
            left,
            right,
            "followed_by",
            "observed",
            1.0,
            scene_nodes[left]["attributes"]["end_ms"],
            scene_nodes[right]["attributes"]["start_ms"],
            scene_nodes[left]["evidence_refs"] + scene_nodes[right]["evidence_refs"],
        )

    data = {
        "schema_version": 1,
        "context_type": "multimodal_scene_graph",
        "session_id": scene["session_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "scene_context": _source_file(scene_path),
            "scene_semantics": _source_file(semantics_path),
            "visual_tracks": _source_file(visual_path),
            "participant_context": _source_file(participant_context_path)
            if participant_context_path
            else None,
        },
        "processing": {
            "scene_merger": {
                "algorithm": MERGER_ALGORITHM,
                "similarity_threshold": similarity_threshold,
                "description_weight": DESCRIPTION_WEIGHT,
                "object_weight": OBJECT_WEIGHT,
                "person_weight": PERSON_WEIGHT,
            },
            "confirmed_speaker_bindings_only": True,
            "semantic_claims_are_inferred": True,
        },
        "nodes": nodes,
        "edges": edges,
        "statistics": {
            "nodes": len(nodes),
            "edges": len(edges),
            "persons": sum(item["type"] == "person" for item in nodes),
            "visual_identities": sum(
                item["type"] == "visual_identity" for item in nodes
            ),
            "scenes": len(scene_ids),
            "objects": sum(item["type"] == "object" for item in nodes),
            "interactions": interaction_edges,
            "observed_edges": sum(item["epistemic_status"] == "observed" for item in edges),
            "inferred_edges": sum(item["epistemic_status"] == "inferred" for item in edges),
        },
        "warnings": [],
    }
    errors = validate_scene_graph(data)
    if errors:
        raise RuntimeError("scene graph validation failed: " + "; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return data


def validate_scene_graph(data: dict) -> List[str]:
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
    missing = required - set(data)
    if missing:
        return ["missing top-level fields: " + ", ".join(sorted(missing))]
    errors = []
    node_ids = [item.get("id") for item in data["nodes"]]
    edge_ids = [item.get("id") for item in data["edges"]]
    if len(node_ids) != len(set(node_ids)):
        errors.append("duplicate node IDs")
    if len(edge_ids) != len(set(edge_ids)):
        errors.append("duplicate edge IDs")
    known_nodes = set(node_ids)
    for node in data["nodes"]:
        if node.get("type") not in NODE_TYPES:
            errors.append(f"{node.get('id')} has an invalid node type")
        if node.get("epistemic_status") not in {"observed", "inferred"}:
            errors.append(f"{node.get('id')} has an invalid epistemic status")
        confidence = node.get("confidence")
        if confidence is not None and not 0 <= confidence <= 1:
            errors.append(f"{node.get('id')} has invalid confidence")
    for edge in data["edges"]:
        if edge.get("source") not in known_nodes or edge.get("target") not in known_nodes:
            errors.append(f"{edge.get('id')} references an unknown node")
        if edge.get("epistemic_status") not in {"observed", "inferred"}:
            errors.append(f"{edge.get('id')} has an invalid epistemic status")
        if not 0 <= edge.get("confidence", -1) <= 1:
            errors.append(f"{edge.get('id')} has invalid confidence")
        if edge.get("end_ms", -1) < edge.get("start_ms", 0):
            errors.append(f"{edge.get('id')} has an invalid time range")
    expected = {
        "nodes": len(data["nodes"]),
        "edges": len(data["edges"]),
        "persons": sum(item.get("type") == "person" for item in data["nodes"]),
        "visual_identities": sum(
            item.get("type") == "visual_identity" for item in data["nodes"]
        ),
        "scenes": sum(item.get("type") == "scene" for item in data["nodes"]),
        "objects": sum(item.get("type") == "object" for item in data["nodes"]),
        "interactions": sum(
            item.get("predicate") not in STRUCTURAL_PREDICATES
            for item in data["edges"]
        ),
        "observed_edges": sum(
            item.get("epistemic_status") == "observed" for item in data["edges"]
        ),
        "inferred_edges": sum(
            item.get("epistemic_status") == "inferred" for item in data["edges"]
        ),
    }
    if data["statistics"] != expected:
        errors.append("statistics do not match graph contents")
    return errors
