from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.scene_graph import project_scene_graph, validate_scene_graph
from scripts.scene_semantics import analyze_scene_semantics
from scripts.test_scene_semantics import _write_fixture


def _semantic_response() -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "keyframes": [
                                {
                                    "keyframe_id": "keyframe-00001",
                                    "environment": {
                                        "category": "indoor_meeting",
                                        "description": "室内会议空间，有一块展示屏幕",
                                        "confidence": 0.9,
                                    },
                                    "objects": [
                                        {
                                            "label": "screen",
                                            "count": 1,
                                            "region": "background",
                                            "confidence": 0.85,
                                        }
                                    ],
                                    "interactions": [
                                        {
                                            "subject_ref": "Face-01",
                                            "predicate": "presenting",
                                            "object_ref": "object:screen",
                                            "description": "A 正在展示屏幕内容",
                                            "evidence_basis": ["image", "transcript"],
                                            "epistemic_status": "inferred",
                                            "confidence": 0.8,
                                        }
                                    ],
                                },
                                {
                                    "keyframe_id": "keyframe-00002",
                                    "environment": {
                                        "category": "indoor_meeting",
                                        "description": "室内会议空间，可见展示屏幕",
                                        "confidence": 0.88,
                                    },
                                    "objects": [
                                        {
                                            "label": "screen",
                                            "count": 1,
                                            "region": "background",
                                            "confidence": 0.8,
                                        }
                                    ],
                                    "interactions": [],
                                },
                            ]
                        },
                        ensure_ascii=False,
                    )
                }
            }
        ]
    }


def test_adjacent_semantics_merge_into_evidence_graph() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scene_path = _write_fixture(root)
        semantics = analyze_scene_semantics(
            scene_path, transport=lambda payload: _semantic_response()
        )
        semantics_path = root / "scene_semantics.json"
        assert semantics["status"] == "complete"
        graph = project_scene_graph(semantics_path)
        assert not validate_scene_graph(graph)
        assert graph["statistics"] == {
            "nodes": 4,
            "edges": 4,
            "persons": 2,
            "scenes": 1,
            "objects": 1,
            "interactions": 1,
            "observed_edges": 2,
            "inferred_edges": 2,
        }
        scene = next(item for item in graph["nodes"] if item["type"] == "scene")
        assert scene["attributes"]["shot_ids"] == ["shot-00001", "shot-00002"]
        assert scene["attributes"]["start_ms"] == 0
        assert scene["attributes"]["end_ms"] == 2000
        person = next(item for item in graph["nodes"] if item["id"] == "person:speaker:A")
        assert person["attributes"]["face_id"] == "Face-01"
        presenting = next(item for item in graph["edges"] if item["predicate"] == "presenting")
        assert presenting["source"] == "person:speaker:A"
        assert graph["nodes"][[item["id"] for item in graph["nodes"]].index(presenting["target"])]["type"] == "object"
        assert (root / "multimodal_scene_graph.json").is_file()


def test_incomplete_semantics_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scene_path = _write_fixture(root)
        analyze_scene_semantics(scene_path, transport=lambda payload: _semantic_response())
        semantics_path = root / "scene_semantics.json"
        data = json.loads(semantics_path.read_text(encoding="utf-8"))
        data["status"] = "partial"
        data["analyses"] = data["analyses"][:1]
        data["statistics"]["analyzed_keyframes"] = 1
        data["statistics"]["semantic_objects"] = 1
        data["statistics"]["interactions"] = 1
        semantics_path.write_text(json.dumps(data), encoding="utf-8")
        try:
            project_scene_graph(semantics_path)
        except ValueError as error:
            assert "must be complete" in str(error)
        else:
            raise AssertionError("partial scene semantics were accepted")


if __name__ == "__main__":
    test_adjacent_semantics_merge_into_evidence_graph()
    test_incomplete_semantics_are_rejected()
    print("PASS: stable scene merging and multimodal graph projection")
