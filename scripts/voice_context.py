from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

from .models import file_sha256
from .multimodal_context import project_multimodal_context
from .speaker_reconciliation import validate_reconciled_timeline


Runner = Callable[[Sequence[str], Path], None]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _run(command: Sequence[str], cwd: Path) -> None:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.returncode:
        raise RuntimeError(
            f"HT-voice-lab command failed with exit code {result.returncode}: "
            + " ".join(command[:2])
        )


def _source_file(path: Path) -> dict:
    return {"path": str(path), "sha256": file_sha256(path)}


def rebuild_voice_context(
    visual_path: Path,
    reconciled_timeline_path: Path,
    output_directory: Path,
    voice_lab_root: Path,
    model: str = "qwen/qwen3.8-27b",
    semantic_timeout: float = 20.0,
    memory_timeout: float = 45.0,
    personal_timeout: float = 45.0,
    minimum_interval: float = 0.2,
    proxy: Optional[str] = None,
    assistant_aliases: Sequence[str] = ("小派",),
    extract_personal_info: bool = True,
    runner: Optional[Runner] = None,
) -> dict:
    """Regenerate voice semantics from the corrected timeline, then fuse visual identity."""
    visual_path = visual_path.expanduser().resolve()
    timeline_path = reconciled_timeline_path.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    voice_lab_root = voice_lab_root.expanduser().resolve()
    voice_lab = voice_lab_root / "lab"
    if not visual_path.is_file():
        raise FileNotFoundError(visual_path)
    if not timeline_path.is_file():
        raise FileNotFoundError(timeline_path)
    if not voice_lab.is_file():
        raise FileNotFoundError(f"HT-voice-lab launcher not found: {voice_lab}")
    for name, value in (
        ("semantic_timeout", semantic_timeout),
        ("memory_timeout", memory_timeout),
        ("personal_timeout", personal_timeout),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if minimum_interval < 0:
        raise ValueError("minimum_interval cannot be negative")
    aliases = tuple(dict.fromkeys(item.strip() for item in assistant_aliases if item.strip()))
    if not aliases:
        raise ValueError("at least one assistant alias is required")

    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    if not isinstance(timeline, dict):
        raise ValueError("reconciled timeline must be a JSON object")
    errors = validate_reconciled_timeline(timeline)
    if errors:
        raise ValueError("invalid reconciled timeline: " + "; ".join(errors))
    spans = timeline["speech_spans"]
    if not spans:
        raise ValueError("reconciled timeline contains no speech spans")

    visual = json.loads(visual_path.read_text(encoding="utf-8"))
    binding = visual.get("processing", {}).get("speaker_face_binding", {})
    if Path(binding.get("speech_timeline_path", "")).expanduser().resolve() != timeline_path:
        raise ValueError("visual speaker binding does not reference this reconciled timeline")
    if binding.get("speech_timeline_sha256") != file_sha256(timeline_path):
        raise ValueError("visual speaker binding timeline hash does not match")

    output_directory.mkdir(parents=True, exist_ok=True)
    spans_path = output_directory / "speech_spans.json"
    dialogue_path = output_directory / "dialogue_state.json"
    memory_path = output_directory / "session_memory.json"
    participant_path = output_directory / "participant_context.json"
    multimodal_path = output_directory / "multimodal_participant_context.json"
    _write_json(spans_path, spans)

    execute = runner or _run
    common_proxy = ["--semantic-proxy", proxy] if proxy else []
    dialogue_command = [
        str(voice_lab),
        "analyze-dialogue",
        str(spans_path),
        "--semantic-provider",
        "groq",
        "--semantic-model",
        model,
        "--semantic-timeout",
        str(semantic_timeout),
        "--semantic-minimum-interval",
        str(minimum_interval),
        "--output",
        str(dialogue_path),
        *common_proxy,
    ]
    for alias in aliases:
        dialogue_command.extend(["--assistant-alias", alias])
    execute(dialogue_command, voice_lab_root)
    if not dialogue_path.is_file():
        raise RuntimeError("HT-voice-lab did not create dialogue_state.json")

    memory_command = [
        str(voice_lab),
        "build-memory",
        str(dialogue_path),
        "--session-id",
        str(timeline["session_id"]),
        "--memory-provider",
        "groq",
        "--memory-model",
        model,
        "--memory-timeout",
        str(memory_timeout),
        "--output",
        str(memory_path),
    ]
    if proxy:
        memory_command.extend(["--memory-proxy", proxy])
    execute(memory_command, voice_lab_root)
    if not memory_path.is_file():
        raise RuntimeError("HT-voice-lab did not create session_memory.json")

    participant_command = [
        str(voice_lab),
        "build-participant-context",
        str(memory_path),
        "--personal-provider",
        "groq" if extract_personal_info else "none",
        "--personal-model",
        model,
        "--personal-timeout",
        str(personal_timeout),
        "--personal-minimum-interval",
        str(minimum_interval),
        "--output",
        str(participant_path),
    ]
    if proxy and extract_personal_info:
        participant_command.extend(["--personal-proxy", proxy])
    execute(participant_command, voice_lab_root)
    if not participant_path.is_file():
        raise RuntimeError("HT-voice-lab did not create participant_context.json")

    multimodal = project_multimodal_context(
        visual_path, participant_path, output_path=multimodal_path
    )
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    participant = json.loads(participant_path.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "context_type": "reconciled_voice_context_build",
        "session_id": timeline["session_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "visual_tracks": _source_file(visual_path),
            "reconciled_speaker_timeline": _source_file(timeline_path),
        },
        "generator": {
            "voice_lab": str(voice_lab),
            "model": model,
            "personal_info_extracted": extract_personal_info,
            "assistant_aliases": list(aliases),
        },
        "outputs": {
            name: _source_file(path)
            for name, path in (
                ("speech_spans", spans_path),
                ("dialogue_state", dialogue_path),
                ("session_memory", memory_path),
                ("participant_context", participant_path),
                ("multimodal_participant_context", multimodal_path),
            )
        },
        "statistics": {
            "speech_spans": len(spans),
            "corrected_spans": len(timeline["corrections"]),
            "completed_turns": int(memory.get("stats", {}).get("completed_turns", 0)),
            "participants": len(participant.get("participants", [])),
            "personal_observations": len(participant.get("observations", [])),
            "confirmed_visual_associations": int(
                multimodal.get("stats", {}).get("confirmed_visual_associations", 0)
            ),
        },
    }
    manifest_path = output_directory / "voice_context_build.json"
    _write_json(manifest_path, manifest)
    return manifest
