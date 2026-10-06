from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from scripts.voice_context import rebuild_voice_context


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_rebuild_uses_corrected_speakers_and_projects_faces() -> None:
    text = "This question belongs to B."
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        voice_root = root / "HT-voice-lab"
        voice_root.mkdir()
        (voice_root / "lab").write_text("#!/bin/sh\n", encoding="utf-8")
        timeline_path = root / "reconciled_speech_spans.json"
        timeline = {
            "schema_version": 1,
            "context_type": "multimodal_speaker_timeline",
            "session_id": "test-session",
            "created_at": "2026-01-01T00:00:00+00:00",
            "source": {
                "visual_tracks": {"path": str(root / "visual.json"), "evidence_sha256": "0" * 64},
                "speech_timeline": {"path": str(root / "original.json"), "sha256": "0" * 64},
            },
            "processing": {
                "algorithm": "confirmed_face_anchor_span_relabel_v1",
                "timeline_offset_ms": 0,
                "min_anchor_confidence": 0.7,
                "min_overlap_ms": 1200,
                "min_span_coverage": 0.6,
                "min_face_dominance": 0.6,
                "anchors": [{"speaker_label": "B", "face_id": "Face-01", "confidence": 0.9}],
            },
            "speech_spans": [
                {
                    "id": "span-wrong",
                    "speaker": "B",
                    "original_speaker": "C",
                    "start_s": 0.0,
                    "end_s": 2.0,
                    "text": text,
                }
            ],
            "corrections": [
                {
                    "correction_id": "speaker-correction-0001",
                    "speech_span_id": "span-wrong",
                    "original_speaker": "C",
                    "resolved_speaker": "B",
                    "face_id": "Face-01",
                    "start_ms": 0,
                    "end_ms": 2000,
                    "overlap_ms": 1900,
                    "span_coverage": 0.95,
                    "face_dominance": 1.0,
                    "anchor_confidence": 0.9,
                    "active_speaker_segment_ids": ["active-1"],
                    "candidate_faces": [
                        {
                            "face_id": "Face-01",
                            "overlap_ms": 1900,
                            "active_speaker_segment_ids": ["active-1"],
                        }
                    ],
                }
            ],
            "statistics": {
                "input_spans": 1,
                "eligible_spans": 1,
                "visually_evidenced_spans": 1,
                "corrected_spans": 1,
                "corrected_duration_ms": 2000,
            },
            "warnings": [],
        }
        _write(timeline_path, timeline)
        visual_path = root / "visual.json"
        visual = {
            "session_id": "visual-test",
            "source": {"duration_ms": 2000},
            "processing": {
                "speaker_face_binding": {
                    "speech_timeline_path": str(timeline_path),
                    "speech_timeline_sha256": hashlib.sha256(timeline_path.read_bytes()).hexdigest(),
                    "timeline_offset_ms": 0,
                    "min_evidence_ms": 1000,
                }
            },
            "faces": [
                {
                    "face_id": "Face-01",
                    "first_seen_ms": 0,
                    "last_seen_ms": 2000,
                    "visible_duration_ms": 2000,
                    "observation_count": 10,
                    "evidence_image": "evidence/Face-01.jpg",
                }
            ],
            "speaker_face_associations": [
                {
                    "speaker_label": "B",
                    "face_id": "Face-01",
                    "confidence": 0.9,
                    "status": "confirmed",
                    "candidate_faces": [
                        {
                            "face_id": "Face-01",
                            "evidence_duration_ms": 1900,
                            "speaker_coverage": 0.95,
                            "evidence": [
                                {
                                    "speech_span_ids": ["span-wrong"],
                                    "active_speaker_segment_id": "active-1",
                                    "start_ms": 0,
                                    "end_ms": 1900,
                                    "duration_ms": 1900,
                                }
                            ],
                        }
                    ],
                }
            ],
            "speaker_visibility_segments": [],
        }
        _write(visual_path, visual)
        output = root / "output"
        commands = []

        def fake_runner(command, _cwd):
            commands.append(list(command))
            target = Path(command[command.index("--output") + 1])
            if command[1] == "analyze-dialogue":
                _write(
                    target,
                    {
                        "schema_version": 1,
                        "contributions": [
                            {
                                "id": "contribution-1",
                                "source_span_id": "span-wrong",
                                "speaker": "B",
                                "start_s": 0.0,
                                "end_s": 2.0,
                                "text": text,
                                "turn_complete": True,
                                "completed_contribution_ids": ["contribution-1"],
                            }
                        ],
                    },
                )
            elif command[1] == "build-memory":
                dialogue = Path(command[2])
                _write(
                    target,
                    {
                        "schema_version": 1,
                        "session_id": "test-session",
                        "source": {
                            "dialogue_state": str(dialogue),
                            "sha256": hashlib.sha256(dialogue.read_bytes()).hexdigest(),
                        },
                        "participants": ["B"],
                        "turns": [],
                        "working_memory": [],
                        "scene": {},
                        "stats": {"completed_turns": 1},
                    },
                )
            else:
                memory = Path(command[2])
                dialogue = output / "dialogue_state.json"
                evidence_id = "test-session:evidence:turn-1"
                participant_id = "test-session:participant:B"
                _write(
                    target,
                    {
                        "schema_version": 1,
                        "session_id": "test-session",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "source": {
                            "session_memory": {
                                "path": str(memory),
                                "sha256": hashlib.sha256(memory.read_bytes()).hexdigest(),
                            },
                            "dialogue_state": {
                                "path": str(dialogue),
                                "sha256": hashlib.sha256(dialogue.read_bytes()).hexdigest(),
                            },
                        },
                        "policy": {"identity_scope": "current_session"},
                        "participants": [
                            {
                                "participant_id": participant_id,
                                "speaker_label": "B",
                                "person_id": None,
                                "display_name": None,
                                "identity_status": "anonymous",
                                "observation_ids": [],
                                "evidence_ids": [evidence_id],
                            }
                        ],
                        "observations": [],
                        "evidence": [
                            {
                                "evidence_id": evidence_id,
                                "participant_id": participant_id,
                                "modality": "speech_text",
                                "source_type": "completed_turn",
                                "source_id": "turn-1",
                                "speaker_label": "B",
                                "start_s": 0.0,
                                "end_s": 2.0,
                                "content": text,
                                "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                "contribution_ids": ["contribution-1"],
                            }
                        ],
                        "associations": [],
                        "extraction": {"provider": "fixture", "model": None, "rejected_claims": []},
                        "stats": {
                            "participants": 1,
                            "completed_turns": 1,
                            "observations": 0,
                            "associations": 0,
                            "rejected_claims": 0,
                        },
                    },
                )

        manifest = rebuild_voice_context(
            visual_path,
            timeline_path,
            output,
            voice_root,
            runner=fake_runner,
        )
        extracted = json.loads((output / "speech_spans.json").read_text())
        multimodal = json.loads((output / "multimodal_participant_context.json").read_text())
        assert extracted[0]["speaker"] == "B"
        assert extracted[0]["original_speaker"] == "C"
        assert [command[1] for command in commands] == [
            "analyze-dialogue",
            "build-memory",
            "build-participant-context",
        ]
        assert "--personal-provider" in commands[2]
        assert commands[2][commands[2].index("--personal-provider") + 1] == "groq"
        assert manifest["statistics"]["corrected_spans"] == 1
        assert manifest["statistics"]["confirmed_visual_associations"] == 1
        assert multimodal["associations"][0]["left_ref"] == "test-session:participant:B"


if __name__ == "__main__":
    test_rebuild_uses_corrected_speakers_and_projects_faces()
    print("PASS: corrected speaker timeline rebuilds voice and multimodal context")
