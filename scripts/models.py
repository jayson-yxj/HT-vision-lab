from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "models.json"
MODEL_DIR = ROOT / "models"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def model_path(name: str) -> Path:
    spec = manifest()["models"][name]
    return MODEL_DIR / spec["filename"]


def verify_model(name: str) -> tuple[bool, str]:
    spec = manifest()["models"][name]
    path = MODEL_DIR / spec["filename"]
    if not path.is_file():
        return False, "missing"
    if path.stat().st_size != spec["size"]:
        return False, "wrong size"
    if file_sha256(path) != spec["sha256"]:
        return False, "wrong SHA-256"
    return True, "ok"


def fetch_models() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = manifest()
    for name, spec in data["models"].items():
        path = MODEL_DIR / spec["filename"]
        valid, _ = verify_model(name)
        if valid:
            print(f"[models] {name}: verified {path}")
            continue

        part = path.with_suffix(path.suffix + ".part")
        if part.exists():
            part.unlink()
        print(f"[models] downloading {name} ({spec['size']} bytes)…")
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "HT-vision-lab/1"})
        with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        os.replace(part, path)
        valid, reason = verify_model(name)
        if not valid:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"downloaded {name} failed verification: {reason}")
        print(f"[models] {name}: downloaded and verified")


def require_models(names=None) -> None:
    failures = []
    selected = names or manifest()["models"].keys()
    for name in selected:
        valid, reason = verify_model(name)
        if not valid:
            failures.append(f"{name} ({reason})")
    if failures:
        raise RuntimeError("models unavailable: " + ", ".join(failures) + "; run ./lab fetch-models")
