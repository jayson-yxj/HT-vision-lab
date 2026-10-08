from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scripts.live_replay import VisionReplay
from scripts.scene_graph import project_scene_graph
from scripts.scene_graph_server import create_server
from scripts.scene_semantics import analyze_scene_semantics
from scripts.test_scene_graph import _semantic_response
from scripts.test_scene_semantics import _write_fixture


def test_server_exposes_graph_page_and_allowlisted_keyframes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scene_path = _write_fixture(root)
        analyze_scene_semantics(scene_path, transport=lambda payload: _semantic_response())
        graph_path = root / "multimodal_scene_graph.json"
        project_scene_graph(root / "scene_semantics.json", output_path=graph_path)
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        replay = VisionReplay(graph, speed=100)
        server, source, url = create_server(graph_path, port=0, replay=replay)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            assert source == graph_path.resolve()
            with urlopen(url, timeout=3) as response:
                page = response.read().decode("utf-8")
                assert response.status == 200
                assert "多模态场景图" in page
                assert "场景时间轴" in page
            with urlopen(url + "api/graph", timeout=3) as response:
                payload = json.load(response)
                assert payload["graph"]["statistics"]["scenes"] == 1
                assert len(payload["assets"]["keyframes"]) == 2
                keyframe_url = payload["assets"]["keyframes"][0]["url"].lstrip("/")
            with urlopen(url + keyframe_url, timeout=3) as response:
                assert response.status == 200
                assert response.headers["Content-Type"] == "image/jpeg"
                assert response.read(2) == b"\xff\xd8"
            with urlopen(url + "api/replay", timeout=3) as response:
                replay_payload = json.load(response)
                assert replay_payload["events_total"] > 0
                assert replay_payload["event_window"] == 100
            request = Request(
                url + "api/replay/control",
                data=b'{"action":"pause"}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=3) as response:
                assert json.load(response)["status"] in {"paused", "complete"}
            try:
                urlopen(url + "api/keyframe?id=../../secret", timeout=3)
            except HTTPError as error:
                assert error.code == 404
            else:
                raise AssertionError("unknown keyframe was served")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    test_server_exposes_graph_page_and_allowlisted_keyframes()
    print("PASS: scene graph server page, API and keyframe allowlist")
