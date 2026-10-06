from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .face_tracks import analyze_video, validate_output
from .models import fetch_models, manifest, verify_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lab", description="HT Vision Lab")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("fetch-models", help="Download and verify pinned local models")
    commands.add_parser("status", help="Show runtime and model status")
    commands.add_parser("asd-status", help="Show PyTorch and LR-ASD runtime status")
    commands.add_parser("login-groq", help="Securely save a Groq API key for vision analysis")

    track = commands.add_parser("track-faces", help="Build session-local face tracks for a video")
    track.add_argument("video", type=Path)
    track.add_argument("--output", type=Path, required=True)
    track.add_argument("--sample-fps", type=float, default=5.0)
    track.add_argument("--detection-threshold", type=float, default=0.70)
    track.add_argument("--min-face-size", type=int, default=24)
    track.add_argument("--max-gap-seconds", type=float, default=0.8)
    track.add_argument("--reid-threshold", type=float, default=0.75)
    track.add_argument("--cluster-threshold", type=float, default=0.45)
    track.add_argument("--min-track-observations", type=int, default=3)
    track.add_argument("--max-people", type=int, default=4)
    track.add_argument("--no-render", action="store_true", help="Skip annotated video output")

    validate = commands.add_parser("validate", help="Check visual_tracks.json references and ranges")
    validate.add_argument("json_path", type=Path)

    active = commands.add_parser("active-speaker", help="Run LR-ASD over existing visual face tracks")
    active.add_argument("json_path", type=Path)
    active.add_argument("--output", type=Path)
    active.add_argument("--model", choices=("ava", "talkset"), default="talkset")
    active.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    active.add_argument("--threshold", type=float, default=0.0)
    active.add_argument("--min-segment-ms", type=int, default=200)
    active.add_argument("--bridge-gap-ms", type=int, default=160)
    active.add_argument("--crop-scale", type=float, default=0.40)
    active.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow more than one visible active speaker at the same time",
    )
    active.add_argument("--no-render", action="store_true")

    binding = commands.add_parser(
        "bind-speakers", help="Bind stabilized A/B/C/D speech intervals to visual Face IDs"
    )
    binding.add_argument("visual_json", type=Path)
    binding.add_argument("speech_json", type=Path)
    binding.add_argument("--output", type=Path)
    binding.add_argument(
        "--timeline-offset-ms",
        type=int,
        default=0,
        help="Milliseconds added to speech timestamps before matching",
    )
    binding.add_argument("--min-evidence-ms", type=int, default=1000)
    binding.add_argument("--min-speaker-coverage", type=float, default=0.50)
    binding.add_argument("--min-margin", type=float, default=0.40)

    visibility = commands.add_parser(
        "classify-visibility",
        help="Classify speaking participants as visible, offscreen, occluded or unknown",
    )
    visibility.add_argument("visual_json", type=Path)
    visibility.add_argument("--output", type=Path)
    visibility.add_argument("--shot-threshold", type=float, default=0.08)
    visibility.add_argument("--shot-sample-fps", type=float, default=10.0)
    visibility.add_argument("--occlusion-max-ms", type=int, default=1200)
    visibility.add_argument("--boundary-tolerance-ms", type=int, default=240)

    scenes = commands.add_parser(
        "analyze-scenes",
        help="Extract shots, keyframes, person positions and 2D spatial relations",
    )
    scenes.add_argument("visual_json", type=Path)
    scenes.add_argument("--output", type=Path)
    scenes.add_argument("--adaptive-threshold", type=float, default=3.0)
    scenes.add_argument("--min-content-val", type=float, default=15.0)
    scenes.add_argument("--min-shot-seconds", type=float, default=0.5)
    scenes.add_argument("--horizontal-threshold", type=float, default=0.08)
    scenes.add_argument("--vertical-threshold", type=float, default=0.08)
    scenes.add_argument("--near-threshold", type=float, default=0.35)
    scenes.add_argument("--overlap-iou-threshold", type=float, default=0.10)

    validate_scenes = commands.add_parser(
        "validate-scenes", help="Validate a visual scene context"
    )
    validate_scenes.add_argument("json_path", type=Path)

    semantics = commands.add_parser(
        "analyze-scene-semantics",
        help="Use Qwen3.8 to infer environments, objects and interactions from keyframes",
    )
    semantics.add_argument("scene_json", type=Path)
    semantics.add_argument("--output", type=Path)
    semantics.add_argument("--model", default="qwen/qwen3.8-27b")
    semantics.add_argument("--timeout", type=float, default=60.0)
    semantics.add_argument("--groq-proxy")
    semantics.add_argument("--minimum-interval", type=float, default=0.2)
    semantics.add_argument("--batch-size", type=int, choices=(1, 2, 3), default=2)

    validate_semantics = commands.add_parser(
        "validate-scene-semantics", help="Validate Qwen scene semantics"
    )
    validate_semantics.add_argument("json_path", type=Path)

    graph = commands.add_parser(
        "build-scene-graph",
        help="Merge adjacent shots and build a person-scene-object graph",
    )
    graph.add_argument("semantics_json", type=Path)
    graph.add_argument("--participant-context", type=Path)
    graph.add_argument("--output", type=Path)
    graph.add_argument("--scene-similarity-threshold", type=float, default=0.20)

    validate_graph = commands.add_parser(
        "validate-scene-graph", help="Validate a multimodal scene graph"
    )
    validate_graph.add_argument("json_path", type=Path)

    conversation_graph = commands.add_parser(
        "build-conversation-graph",
        help="Fuse voice topics, opinions and intents into a multimodal scene graph",
    )
    conversation_graph.add_argument("scene_graph_json", type=Path)
    conversation_graph.add_argument("session_memory_json", type=Path)
    conversation_graph.add_argument("--output", type=Path)

    validate_conversation_graph = commands.add_parser(
        "validate-conversation-graph", help="Validate a multimodal conversation graph"
    )
    validate_conversation_graph.add_argument("json_path", type=Path)

    visualize_graph = commands.add_parser(
        "visualize-scene-graph", help="Serve the interactive multimodal scene graph"
    )
    visualize_graph.add_argument("graph_json", type=Path)
    visualize_graph.add_argument("--host", default="127.0.0.1")
    visualize_graph.add_argument("--port", type=int, default=8765)
    visualize_graph.add_argument("--no-open", action="store_true")

    visualize_conversation = commands.add_parser(
        "visualize-conversation-graph",
        help="Serve the interactive multimodal conversation graph",
    )
    visualize_conversation.add_argument("graph_json", type=Path)
    visualize_conversation.add_argument("--host", default="127.0.0.1")
    visualize_conversation.add_argument("--port", type=int, default=8765)
    visualize_conversation.add_argument("--no-open", action="store_true")

    participants = commands.add_parser(
        "project-participants",
        help="Project visual identities into an existing participant context",
    )
    participants.add_argument("visual_json", type=Path)
    participants.add_argument("participant_context_json", type=Path)
    participants.add_argument("--output", type=Path)

    validate_participants = commands.add_parser(
        "validate-participants", help="Validate a multimodal participant context"
    )
    validate_participants.add_argument("json_path", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "fetch-models":
            fetch_models()
            return 0
        if args.command == "status":
            data = manifest()
            for name in data["models"]:
                valid, reason = verify_model(name)
                print(f"{name}: {reason if valid else 'unavailable: ' + reason}")
            return 0
        if args.command == "asd-status":
            import torch

            print(f"torch: {torch.__version__}")
            print(f"cuda_available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"cuda_device: {torch.cuda.get_device_name(0)}")
            for name in ("lr_asd_ava", "lr_asd_talkset"):
                valid, reason = verify_model(name)
                print(f"{name}: {reason if valid else 'unavailable: ' + reason}")
            return 0
        if args.command == "login-groq":
            from .groq_client import login

            return login()
        if args.command == "track-faces":
            data = analyze_video(
                args.video,
                args.output,
                sample_fps=args.sample_fps,
                detection_threshold=args.detection_threshold,
                min_face_size=args.min_face_size,
                max_gap_seconds=args.max_gap_seconds,
                reid_threshold=args.reid_threshold,
                cluster_threshold=args.cluster_threshold,
                min_track_observations=args.min_track_observations,
                max_people=args.max_people,
                render=not args.no_render,
            )
            errors = validate_output(data)
            if errors:
                raise RuntimeError("output validation failed: " + "; ".join(errors))
            stats = data["statistics"]
            print(
                f"[done] {stats['retained_faces']} faces, {stats['retained_tracklets']} retained tracklets, "
                f"{stats['detections']} observations"
            )
            print(f"[done] {args.output.expanduser().resolve() / 'visual_tracks.json'}")
            if not args.no_render:
                print(f"[done] {args.output.expanduser().resolve() / 'annotated.mp4'}")
            for warning in data["warnings"]:
                print(f"[warning] {warning}")
            return 0
        if args.command == "validate":
            with args.json_path.open(encoding="utf-8") as handle:
                errors = validate_output(json.load(handle))
            if errors:
                for error in errors:
                    print(f"ERROR: {error}", file=sys.stderr)
                return 1
            print(f"PASS: {args.json_path}")
            return 0
        if args.command == "active-speaker":
            from .active_speaker import analyze_active_speakers

            data = analyze_active_speakers(
                args.json_path,
                output_path=args.output,
                model_name=args.model,
                device_name=args.device,
                threshold=args.threshold,
                min_segment_ms=args.min_segment_ms,
                bridge_gap_ms=args.bridge_gap_ms,
                crop_scale=args.crop_scale,
                exclusive_speaker=not args.allow_overlap,
                render=not args.no_render,
            )
            errors = validate_output(data)
            if errors:
                raise RuntimeError("output validation failed: " + "; ".join(errors))
            output = args.output.expanduser().resolve() if args.output else args.json_path.expanduser().resolve()
            print(f"[done] {output}")
            if not args.no_render:
                print(f"[done] {output.parent / 'active_speaker.mp4'}")
            return 0
        if args.command == "bind-speakers":
            from .speaker_face_binding import bind_speakers_to_faces

            data = bind_speakers_to_faces(
                args.visual_json,
                args.speech_json,
                output_path=args.output,
                timeline_offset_ms=args.timeline_offset_ms,
                min_evidence_ms=args.min_evidence_ms,
                min_speaker_coverage=args.min_speaker_coverage,
                min_margin=args.min_margin,
            )
            errors = validate_output(data)
            if errors:
                raise RuntimeError("output validation failed: " + "; ".join(errors))
            output = args.output.expanduser().resolve() if args.output else args.visual_json.expanduser().resolve()
            print(f"[done] {output}")
            for association in data["speaker_face_associations"]:
                print(
                    f"[binding] {association['speaker_label']} -> "
                    f"{association['face_id'] or 'offscreen'} "
                    f"({association['status']}, confidence={association['confidence']:.3f}, "
                    f"evidence={association['evidence_duration_ms'] / 1000:.2f}s)"
                )
            return 0
        if args.command == "classify-visibility":
            from .visibility import classify_visibility

            data = classify_visibility(
                args.visual_json,
                output_path=args.output,
                shot_threshold=args.shot_threshold,
                shot_sample_fps=args.shot_sample_fps,
                occlusion_max_ms=args.occlusion_max_ms,
                boundary_tolerance_ms=args.boundary_tolerance_ms,
            )
            errors = validate_output(data)
            if errors:
                raise RuntimeError("output validation failed: " + "; ".join(errors))
            output = args.output.expanduser().resolve() if args.output else args.visual_json.expanduser().resolve()
            print(f"[done] {output}")
            counts = data["statistics"]["speaker_visibility_state_counts"]
            print(
                "[visibility] "
                + ", ".join(
                    f"{state}={counts[state]}"
                    for state in ("visible", "offscreen", "occluded", "unknown")
                )
            )
            return 0
        if args.command == "analyze-scenes":
            from .scene_context import analyze_scenes

            data = analyze_scenes(
                args.visual_json,
                output_path=args.output,
                adaptive_threshold=args.adaptive_threshold,
                min_content_val=args.min_content_val,
                min_shot_seconds=args.min_shot_seconds,
                horizontal_threshold=args.horizontal_threshold,
                vertical_threshold=args.vertical_threshold,
                near_threshold=args.near_threshold,
                overlap_iou_threshold=args.overlap_iou_threshold,
            )
            output = (
                args.output.expanduser().resolve()
                if args.output
                else args.visual_json.expanduser().resolve().parent / "scene_context.json"
            )
            stats = data["statistics"]
            print(f"[done] {output}")
            print(
                f"[scenes] {stats['shots']} shots, {stats['keyframes']} keyframes, "
                f"{stats['person_states']} person states, "
                f"{stats['spatial_relations']} spatial relations"
            )
            return 0
        if args.command == "validate-scenes":
            from .scene_context import validate_scene_context

            with args.json_path.open(encoding="utf-8") as handle:
                errors = validate_scene_context(json.load(handle))
            if errors:
                for error in errors:
                    print(f"ERROR: {error}", file=sys.stderr)
                return 1
            print(f"PASS: {args.json_path}")
            return 0
        if args.command == "analyze-scene-semantics":
            from .scene_semantics import analyze_scene_semantics

            data = analyze_scene_semantics(
                args.scene_json,
                output_path=args.output,
                model=args.model,
                timeout=args.timeout,
                proxy=args.groq_proxy,
                minimum_interval=args.minimum_interval,
                batch_size=args.batch_size,
                progress=lambda message: print(f"[semantics] {message}", flush=True),
            )
            output = (
                args.output.expanduser().resolve()
                if args.output
                else args.scene_json.expanduser().resolve().parent / "scene_semantics.json"
            )
            stats = data["statistics"]
            print(f"[done] {output}")
            print(
                f"[semantics] status={data['status']}, "
                f"keyframes={stats['analyzed_keyframes']}/{stats['keyframes']}, "
                f"objects={stats['semantic_objects']}, interactions={stats['interactions']}"
            )
            for warning in data["warnings"]:
                print(f"[warning] {warning}")
            return 0 if data["status"] == "complete" else 1
        if args.command == "validate-scene-semantics":
            from .scene_semantics import validate_scene_semantics

            with args.json_path.open(encoding="utf-8") as handle:
                errors = validate_scene_semantics(json.load(handle))
            if errors:
                for error in errors:
                    print(f"ERROR: {error}", file=sys.stderr)
                return 1
            print(f"PASS: {args.json_path}")
            return 0
        if args.command == "build-scene-graph":
            from .scene_graph import project_scene_graph

            data = project_scene_graph(
                args.semantics_json,
                participant_context_path=args.participant_context,
                output_path=args.output,
                similarity_threshold=args.scene_similarity_threshold,
            )
            output = (
                args.output.expanduser().resolve()
                if args.output
                else args.semantics_json.expanduser().resolve().parent
                / "multimodal_scene_graph.json"
            )
            stats = data["statistics"]
            print(f"[done] {output}")
            print(
                f"[graph] persons={stats['persons']}, scenes={stats['scenes']}, "
                f"objects={stats['objects']}, interactions={stats['interactions']}, "
                f"edges={stats['edges']}"
            )
            return 0
        if args.command == "validate-scene-graph":
            from .scene_graph import validate_scene_graph

            with args.json_path.open(encoding="utf-8") as handle:
                errors = validate_scene_graph(json.load(handle))
            if errors:
                for error in errors:
                    print(f"ERROR: {error}", file=sys.stderr)
                return 1
            print(f"PASS: {args.json_path}")
            return 0
        if args.command == "build-conversation-graph":
            from .conversation_graph import project_conversation_graph

            data = project_conversation_graph(
                args.scene_graph_json,
                args.session_memory_json,
                output_path=args.output,
            )
            output = (
                args.output.expanduser().resolve()
                if args.output
                else args.scene_graph_json.expanduser().resolve().parent
                / "multimodal_conversation_graph.json"
            )
            stats = data["statistics"]
            print(f"[done] {output}")
            print(
                f"[conversation] persons={stats['persons']}, topics={stats['topics']}, "
                f"opinions={stats['opinions']}, intents={stats['intents']}, "
                f"scene-linked-turns={stats['temporally_linked_turns']}/{stats['opinions']}"
            )
            for warning in data["warnings"]:
                print(f"[warning] {warning}")
            return 0
        if args.command == "validate-conversation-graph":
            from .conversation_graph import validate_conversation_graph

            with args.json_path.open(encoding="utf-8") as handle:
                errors = validate_conversation_graph(json.load(handle))
            if errors:
                for error in errors:
                    print(f"ERROR: {error}", file=sys.stderr)
                return 1
            print(f"PASS: {args.json_path}")
            return 0
        if args.command in {"visualize-scene-graph", "visualize-conversation-graph"}:
            from .scene_graph_server import serve_scene_graph

            return serve_scene_graph(
                args.graph_json,
                host=args.host,
                port=args.port,
                open_browser=not args.no_open,
            )
        if args.command == "project-participants":
            from .multimodal_context import project_multimodal_context

            data = project_multimodal_context(
                args.visual_json,
                args.participant_context_json,
                output_path=args.output,
            )
            output = (
                args.output.expanduser().resolve()
                if args.output
                else args.visual_json.expanduser().resolve().parent
                / "multimodal_participant_context.json"
            )
            print(f"[done] {output}")
            print(
                f"[participants] {data['stats']['participants']} participants, "
                f"{data['stats']['visual_associations']} visual associations "
                f"({data['stats']['confirmed_visual_associations']} confirmed, "
                f"{data['stats']['disputed_visual_associations']} disputed)"
            )
            return 0
        if args.command == "validate-participants":
            from .multimodal_context import validate_multimodal_context

            with args.json_path.open(encoding="utf-8") as handle:
                errors = validate_multimodal_context(json.load(handle))
            if errors:
                for error in errors:
                    print(f"ERROR: {error}", file=sys.stderr)
                return 1
            print(f"PASS: {args.json_path}")
            return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
