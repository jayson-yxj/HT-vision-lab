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
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
