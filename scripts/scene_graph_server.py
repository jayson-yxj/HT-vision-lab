from __future__ import annotations

import json
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .models import file_sha256
from .scene_context import validate_scene_context
from .scene_graph import validate_scene_graph


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "web" / "scene_graph.html"


def _load_sources(graph_path: Path) -> Tuple[dict, dict, Dict[str, Path]]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    if graph.get("context_type") == "multimodal_conversation_graph":
        from .conversation_graph import validate_conversation_graph

        errors = validate_conversation_graph(graph)
    else:
        errors = validate_scene_graph(graph)
    if errors:
        raise ValueError("invalid graph: " + "; ".join(errors))
    scene_source = graph["source"]["scene_context"]
    scene_path = Path(scene_source["path"]).expanduser().resolve()
    if not scene_path.is_file() or file_sha256(scene_path) != scene_source["sha256"]:
        raise ValueError("scene graph scene_context source is unavailable or has changed")
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene_errors = validate_scene_context(scene)
    if scene_errors:
        raise ValueError("invalid scene context: " + "; ".join(scene_errors))
    keyframe_paths = {}
    keyframes = []
    asset_root = scene_path.parent.resolve()
    for item in scene["keyframes"]:
        image_path = (scene_path.parent / item["annotated_image_path"]).resolve()
        try:
            image_path.relative_to(asset_root)
        except ValueError:
            continue
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"} or not image_path.is_file():
            continue
        keyframe_paths[item["keyframe_id"]] = image_path
        keyframes.append(
            {
                "id": item["keyframe_id"],
                "shot_id": item["shot_id"],
                "timestamp_ms": item["timestamp_ms"],
                "url": "/api/keyframe?id=" + item["keyframe_id"],
            }
        )
    return graph, {"keyframes": keyframes}, keyframe_paths


def create_server(
    graph_path: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> Tuple[ThreadingHTTPServer, Path, str]:
    source = graph_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("graph is missing: " + str(source))
    if not PAGE.is_file():
        raise FileNotFoundError("scene graph page is missing: " + str(PAGE))
    _load_sources(source)
    page = PAGE.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(
            self,
            status: int,
            content_type: str,
            body: bytes = b"",
            etag: Optional[str] = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            if etag:
                self.send_header("ETag", etag)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send(200, "text/html; charset=utf-8", page)
                return
            if parsed.path == "/favicon.ico":
                self._send(204, "image/x-icon")
                return
            try:
                graph, assets, keyframe_paths = _load_sources(source)
                if parsed.path == "/api/graph":
                    stat = source.stat()
                    etag = f'"{stat.st_mtime_ns}-{stat.st_size}"'
                    if self.headers.get("If-None-Match") == etag:
                        self._send(304, "application/json", etag=etag)
                        return
                    body = json.dumps(
                        {"graph": graph, "assets": assets}, ensure_ascii=False
                    ).encode("utf-8")
                    self._send(200, "application/json; charset=utf-8", body, etag)
                    return
                if parsed.path == "/api/keyframe":
                    identifier = parse_qs(parsed.query).get("id", [""])[0]
                    image_path = keyframe_paths.get(identifier)
                    if image_path is None:
                        self._send(404, "text/plain; charset=utf-8", b"Unknown keyframe\n")
                        return
                    content_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
                    self._send(200, content_type, image_path.read_bytes())
                    return
                self._send(404, "text/plain; charset=utf-8", b"Not found\n")
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                body = json.dumps({"error": str(error)}, ensure_ascii=False).encode("utf-8")
                self._send(422, "application/json; charset=utf-8", body)

        def log_message(self, _format: str, *_args) -> None:
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{server.server_address[1]}/"
    return server, source, url


def serve_scene_graph(
    graph_path: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> int:
    server, source, url = create_server(graph_path, host, port)
    print("Watching " + str(source), flush=True)
    print(url, flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
