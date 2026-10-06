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
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
