from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from scripts.multimodal_context import project_multimodal_context, validate_multimodal_context


def _participant(label: str) -> dict:
    return {
        "participant_id": f"test-session:participant:{label}",
        "speaker_label": label,
        "person_id": None,
        "display_name": "林晓" if label == "A" else None,
        "identity_status": "session_identified" if label == "A" else "anonymous",
        "observation_ids": ["observation-name"] if label == "A" else [],
        "evidence_ids": ["evidence-turn-a"] if label == "A" else [],
    }


def _base_context() -> dict:
    return {
        "schema_version": 1,
        "session_id": "test-session",
        "created_at": "2026-01-01T00:00:00+00:00",
        "source": {"session_memory": {}, "dialogue_state": {}},
        "policy": {"identity_scope": "current_session", "unknown_fields_remain_empty": True},
        "participants": [_participant("A"), _participant("B")],
        "observations": [
            {
                "observation_id": "observation-name",
                "participant_id": "test-session:participant:A",
                "modality": "speech_text",
                "kind": "name",
                "value": "林晓",
                "epistemic_status": "explicit",
                "state": "candidate",
                "confidence": 0.95,
                "durability": "stable",
                "evidence_ids": ["evidence-turn-a"],
                "evidence_quotes": [{"evidence_id": "evidence-turn-a", "quote": "我叫林晓"}],
                "context_evidence_ids": [],
                "context_quotes": [],
            }
        ],
        "evidence": [
            {
                "evidence_id": "evidence-turn-a",
                "participant_id": "test-session:participant:A",
                "modality": "speech_text",
                "source_type": "completed_turn",
                "source_id": "turn-a",
                "speaker_label": "A",
                "start_s": 0.0,
                "end_s": 1.0,
                "content": "我叫林晓",
                "content_sha256": "5872f10a1171323941e1d1bc4161b55a61c36b471d092d649fa06cff27e25a5d",
                "contribution_ids": ["turn-a"],
            }
        ],
        "associations": [],
        "extraction": {"provider": "fixture", "model": None, "rejected_claims": []},
        "stats": {"participants": 2, "completed_turns": 1, "observations": 1, "associations": 0},
    }


def _overlap(segment_id: str, start_ms: int, end_ms: int) -> dict:
    return {
        "speech_span_ids": [f"span-{segment_id}"],
        "active_speaker_segment_id": segment_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
    }


def _visual_tracks(speech_path: Path) -> dict:
    overlap_a = _overlap("active-a", 0, 1000)
    overlap_b1 = _overlap("active-b1", 1000, 1800)
    overlap_b2 = _overlap("active-b2", 1800, 2400)
    faces = []
    for index in range(1, 4):
        faces.append(
            {
                "face_id": f"Face-{index:02d}",
                "first_seen_ms": 0,
                "last_seen_ms": 2400,
                "visible_duration_ms": 2400,
                "observation_count": 10,
                "evidence_image": f"evidence/Face-{index:02d}.jpg",
            }
        )
    return {
        "session_id": "visual-session",
        "source": {"duration_ms": 2400},
        "processing": {
            "speaker_face_binding": {
                "min_evidence_ms": 500,
                "speech_timeline_path": str(speech_path),
                "speech_timeline_sha256": hashlib.sha256(speech_path.read_bytes()).hexdigest(),
            }
        },
        "faces": faces,
        "speaker_face_associations": [
            {
                "speaker_label": "A",
                "face_id": "Face-01",
                "confidence": 0.98,
                "status": "confirmed",
                "candidate_faces": [
                    {
                        "face_id": "Face-01",
                        "evidence_duration_ms": 1000,
                        "speaker_coverage": 1.0,
                        "evidence": [overlap_a],
                    }
                ],
            },
            {
                "speaker_label": "B",
                "face_id": "Face-02",
                "confidence": 0.65,
                "status": "ambiguous",
                "candidate_faces": [
                    {
                        "face_id": "Face-02",
                        "evidence_duration_ms": 800,
                        "speaker_coverage": 0.57,
                        "evidence": [overlap_b1],
                    },
                    {
                        "face_id": "Face-03",
                        "evidence_duration_ms": 600,
                        "speaker_coverage": 0.43,
                        "evidence": [overlap_b2],
                    },
                ],
            },
        ],
    }


def test_projection_preserves_personal_information_and_binding_uncertainty() -> None:
    upstream = _base_context()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        visual_path = root / "visual.json"
        participant_path = root / "participant.json"
        speech_path = root / "speech.json"
        speech_path.write_text(
            json.dumps(
                [
                    {"id": "span-active-a", "speaker": "A", "start_s": 0.0, "end_s": 1.0, "text": "我叫林晓"},
                    {"id": "span-active-b1", "speaker": "B", "start_s": 1.0, "end_s": 1.8, "text": ""},
                    {"id": "span-active-b2", "speaker": "B", "start_s": 1.8, "end_s": 2.4, "text": ""},
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        visual_path.write_text(json.dumps(_visual_tracks(speech_path), ensure_ascii=False), encoding="utf-8")
        participant_path.write_text(json.dumps(upstream, ensure_ascii=False), encoding="utf-8")
        result = project_multimodal_context(visual_path, participant_path)
        assert (root / "multimodal_participant_context.json").is_file()
        assert not validate_multimodal_context(result)
    assert result["participants"][0]["display_name"] == "林晓"
    assert result["observations"] == upstream["observations"]
    visual_associations = [
        item for item in result["associations"] if item["relation"] == "same_session_identity"
    ]
    assert [(item["right_ref"], item["state"]) for item in visual_associations] == [
        ("visual-session:face:Face-01", "confirmed"),
        ("visual-session:face:Face-02", "disputed"),
        ("visual-session:face:Face-03", "disputed"),
    ]
    assert result["stats"]["confirmed_visual_associations"] == 1
    assert result["stats"]["disputed_visual_associations"] == 2


def test_missing_participant_rejects_binding() -> None:
    upstream = _base_context()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        visual_path = root / "visual.json"
        participant_path = root / "participant.json"
        speech_path = root / "speech.json"
        speech_path.write_text(
            json.dumps([{"speaker": "A", "start_s": 0, "end_s": 1, "text": "我叫林晓"}], ensure_ascii=False),
            encoding="utf-8",
        )
        visual = _visual_tracks(speech_path)
        visual["speaker_face_associations"][0]["speaker_label"] = "C"
        visual_path.write_text(json.dumps(visual), encoding="utf-8")
        participant_path.write_text(json.dumps(upstream), encoding="utf-8")
        try:
            project_multimodal_context(visual_path, participant_path)
        except ValueError as error:
            assert "has no participant" in str(error)
        else:
            raise AssertionError("binding without a participant was accepted")


def test_mismatched_conversation_is_rejected() -> None:
    upstream = _base_context()
    upstream["evidence"][0]["content"] = "这是另一场对话"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        speech_path = root / "speech.json"
        speech_path.write_text(
            json.dumps([{"speaker": "A", "start_s": 0, "end_s": 1, "text": "我叫林晓"}], ensure_ascii=False),
            encoding="utf-8",
        )
        visual_path = root / "visual.json"
        participant_path = root / "participant.json"
        visual_path.write_text(json.dumps(_visual_tracks(speech_path)), encoding="utf-8")
        participant_path.write_text(json.dumps(upstream, ensure_ascii=False), encoding="utf-8")
        try:
            project_multimodal_context(visual_path, participant_path)
        except ValueError as error:
            assert "does not match the binding timeline" in str(error)
        else:
            raise AssertionError("participant context from another conversation was accepted")


if __name__ == "__main__":
    test_projection_preserves_personal_information_and_binding_uncertainty()
    test_missing_participant_rejects_binding()
    test_mismatched_conversation_is_rejected()
    print("PASS: multimodal participant context projection")
