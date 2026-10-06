from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path
from urllib.request import urlopen

from scripts.conversation_graph import (
    project_conversation_graph,
    validate_conversation_graph,
)
from scripts.models import file_sha256
from scripts.scene_graph import project_scene_graph
from scripts.scene_graph_server import create_server
from scripts.scene_semantics import analyze_scene_semantics
from scripts.test_scene_graph import _semantic_response
from scripts.test_scene_semantics import _write_fixture


def _write_sources(root: Path) -> tuple[Path, Path]:
    scene_path = _write_fixture(root)
    analyze_scene_semantics(scene_path, transport=lambda payload: _semantic_response())
    scene_graph_path = root / "multimodal_scene_graph.json"
    graph = project_scene_graph(
        root / "scene_semantics.json",
        output_path=scene_graph_path,
        similarity_threshold=1.0,
    )
    assert graph["statistics"]["scenes"] == 2
    second_person = next(
        node
        for node in graph["nodes"]
        if node["type"] == "visual_identity"
    )
    old_person_id = second_person["id"]
    second_person["id"] = "person:speaker:B"
    second_person["type"] = "person"
    second_person["label"] = "B"
    second_person["attributes"]["speaker_label"] = "B"
    second_person["attributes"]["participant_id"] = "voice-test:participant:B"
    second_person["attributes"]["identity_status"] = "confirmed"
    for edge in graph["edges"]:
        if edge["source"] == old_person_id:
            edge["source"] = second_person["id"]
        if edge["target"] == old_person_id:
            edge["target"] = second_person["id"]
    graph["statistics"]["persons"] += 1
    graph["statistics"]["visual_identities"] -= 1

    memory_path = root / "session_memory.json"
    memory = {
        "schema_version": 1,
        "session_id": "voice-test",
        "created_at": "2026-10-06T00:00:00+00:00",
        "source": {},
        "participants": ["A", "B"],
        "turns": [
            {
                "id": "turn-000001",
                "speaker": "A",
                "start_s": 0.1,
                "end_s": 0.8,
                "contribution_ids": ["contribution-000001"],
                "summary": "A 提议先做一个小规模演示。",
                "speech_act": "proposal",
                "intent": "推动演示",
                "stance": "supportive",
                "topic_names": ["演示方案"],
                "entities": [],
            },
            {
                "id": "turn-000002",
                "speaker": "B",
                "start_s": 0.8,
                "end_s": 1.4,
                "contribution_ids": ["contribution-000002"],
                "summary": "B 同意并补充需要记录结果。",
                "speech_act": "agreement",
                "intent": "确认记录要求",
                "stance": "supportive",
                "topic_names": ["演示方案"],
                "entities": [],
            },
        ],
        "working_memory": [],
        "scene": {
            "session_summary": "两人讨论演示方案。",
            "topics": [
                {
                    "name": "演示方案",
                    "summary": "先演示并记录结果。",
                    "evidence_turn_ids": ["turn-000001", "turn-000002"],
                }
            ],
        },
        "stats": {},
    }
    memory_path.write_text(json.dumps(memory, ensure_ascii=False), encoding="utf-8")
    participant_path = root / "multimodal_participant_context.json"
    participant_path.write_text(
        json.dumps(
            {
                "session_id": "voice-test",
                "source": {
                    "upstream": {
                        "session_memory": {
                            "path": str(memory_path),
                            "sha256": file_sha256(memory_path),
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    graph["source"]["participant_context"] = {
        "path": str(participant_path),
        "sha256": file_sha256(participant_path),
    }
    scene_graph_path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    return scene_graph_path, memory_path


def test_voice_graph_reuses_people_and_aligns_turns_to_scenes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scene_graph_path, memory_path = _write_sources(root)
        graph = project_conversation_graph(scene_graph_path, memory_path)
        assert not validate_conversation_graph(graph)
        assert graph["statistics"]["topics"] == 1
        assert graph["statistics"]["opinions"] == 2
        assert graph["statistics"]["intents"] == 2
        assert graph["statistics"]["temporally_linked_turns"] == 2
        assert graph["statistics"]["unlinked_turns"] == 0
        assert len([node for node in graph["nodes"] if node["type"] == "person"]) == 2
        assert graph["statistics"]["visual_identities"] == 0
        assert not any(node["id"].startswith("person:voice") for node in graph["nodes"])

        links = [edge for edge in graph["edges"] if edge["predicate"] == "occurred_in"]
        assert len(links) == 3
        crossing = [
            edge for edge in links if edge["source"] == "opinion:turn-000002"
        ]
        assert [(edge["start_ms"], edge["end_ms"]) for edge in crossing] == [
            (800, 1000),
            (1000, 1400),
        ]
        assert [edge["attributes"]["turn_coverage"] for edge in crossing] == [
            0.333333,
            0.666667,
        ]
        assert len(
            [edge for edge in graph["edges"] if edge["predicate"] == "discussed_in"]
        ) == 2
        assert (root / "multimodal_conversation_graph.json").is_file()


def test_memory_not_referenced_by_participant_context_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scene_graph_path, memory_path = _write_sources(root)
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        memory["scene"]["session_summary"] = "changed"
        memory_path.write_text(json.dumps(memory), encoding="utf-8")
        try:
            project_conversation_graph(scene_graph_path, memory_path)
        except ValueError as error:
            assert "exact session_memory" in str(error)
        else:
            raise AssertionError("a changed session memory was accepted")


def test_scene_graph_server_accepts_conversation_graph() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scene_graph_path, memory_path = _write_sources(root)
        conversation_path = root / "multimodal_conversation_graph.json"
        project_conversation_graph(
            scene_graph_path, memory_path, output_path=conversation_path
        )
        server, _, url = create_server(conversation_path, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(url + "api/graph", timeout=3) as response:
                payload = json.load(response)
            assert payload["graph"]["context_type"] == "multimodal_conversation_graph"
            assert payload["graph"]["statistics"]["opinions"] == 2
            assert len(payload["assets"]["keyframes"]) == 2
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    test_voice_graph_reuses_people_and_aligns_turns_to_scenes()
    test_memory_not_referenced_by_participant_context_is_rejected()
    test_scene_graph_server_accepts_conversation_graph()
    print("PASS: voice topics, opinions and intents fuse with visual scenes")
