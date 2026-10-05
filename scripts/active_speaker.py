from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import python_speech_features
import torch
from scipy import signal

from .lr_asd_model import LRASDInference
from .models import manifest, model_path, require_models


ASD_FPS = 25.0
ASD_FRAME_MS = 40


@dataclass
class TrackSequence:
    tracklet_id: str
    face_id: str
    times_ms: np.ndarray
    source_frames: np.ndarray
    boxes: np.ndarray
    crops: Optional[np.ndarray] = None
    raw_scores: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    speaking: Optional[np.ndarray] = None


def _choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    return torch.device(requested)


def _load_model(model_name: str, device: torch.device) -> LRASDInference:
    manifest_name = "lr_asd_talkset" if model_name == "talkset" else "lr_asd_ava"
    require_models([manifest_name])
    model = LRASDInference()
    state = torch.load(str(model_path(manifest_name)), map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def _model_record(model_name: str) -> dict:
    manifest_name = "lr_asd_talkset" if model_name == "talkset" else "lr_asd_ava"
    data = manifest()
    spec = data["models"][manifest_name]
    return {
        "name": manifest_name,
        "filename": spec["filename"],
        "sha256": spec["sha256"],
        "repository_revision": "1b6dcd2d8fc2895683de6508ec6294ec47d388ca",
        "license": spec["license"],
    }


def _median(values: np.ndarray, kernel: int = 13) -> np.ndarray:
    if len(values) < 3:
        return values
    size = min(kernel, len(values) if len(values) % 2 else len(values) - 1)
    return signal.medfilt(values, kernel_size=max(3, size))


def _build_sequences(data: dict) -> List[TrackSequence]:
    fps = float(data["source"]["fps"])
    observations = {item["observation_id"]: item for item in data["observations"]}
    sequences = []
    for tracklet in data["tracklets"]:
        if tracklet["face_id"] is None:
            continue
        items = sorted(
            (observations[item] for item in tracklet["observation_ids"]), key=lambda item: item["timestamp_ms"]
        )
        if len(items) < 2:
            continue
        source_times = np.asarray([item["timestamp_ms"] for item in items], dtype=np.float64)
        source_boxes = np.asarray([item["bbox_px"] for item in items], dtype=np.float64)
        count = max(1, int(math.floor((source_times[-1] - source_times[0]) / ASD_FRAME_MS)) + 1)
        times_ms = source_times[0] + np.arange(count, dtype=np.float64) * ASD_FRAME_MS
        boxes = np.column_stack(
            [np.interp(times_ms, source_times, source_boxes[:, column]) for column in range(4)]
        )
        centers_x = _median(boxes[:, 0] + boxes[:, 2] / 2)
        centers_y = _median(boxes[:, 1] + boxes[:, 3] / 2)
        scales = _median(np.maximum(boxes[:, 2], boxes[:, 3]) / 2)
        boxes[:, 0] = centers_x - scales
        boxes[:, 1] = centers_y - scales
        boxes[:, 2] = scales * 2
        boxes[:, 3] = scales * 2
        source_frames = np.rint(times_ms * fps / 1000).astype(np.int64)
        sequences.append(
            TrackSequence(
                tracklet_id=tracklet["tracklet_id"],
                face_id=tracklet["face_id"],
                times_ms=np.rint(times_ms).astype(np.int64),
                source_frames=source_frames,
                boxes=boxes,
            )
        )
    return sequences


def _crop_lr_asd(frame: np.ndarray, box: Sequence[float], crop_scale: float) -> np.ndarray:
    x, y, width, height = box
    size = max(width, height) / 2
    center_x = x + width / 2
    center_y = y + height / 2
    padding = max(2, int(math.ceil(size * (1 + 2 * crop_scale))))
    padded = cv2.copyMakeBorder(
        frame, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(110, 110, 110)
    )
    center_x += padding
    center_y += padding
    x1 = int(round(center_x - size * (1 + crop_scale)))
    x2 = int(round(center_x + size * (1 + crop_scale)))
    y1 = int(round(center_y - size))
    y2 = int(round(center_y + size * (1 + 2 * crop_scale)))
    crop = padded[max(0, y1) : max(y1 + 1, y2), max(0, x1) : max(x1 + 1, x2)]
    if crop.size == 0:
        crop = np.full((224, 224, 3), 110, dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return gray[56:168, 56:168]


def _extract_track_crops(video_path: Path, sequence: TrackSequence, crop_scale: float) -> np.ndarray:
    crops = np.zeros((len(sequence.times_ms), 112, 112), dtype=np.uint8)
    captured = np.zeros(len(sequence.times_ms), dtype=np.bool_)
    requests: Dict[int, List[int]] = {}
    for index, frame_index in enumerate(sequence.source_frames.tolist()):
        requests.setdefault(frame_index, []).append(index)
    first_frame, last_frame = int(sequence.source_frames[0]), int(sequence.source_frames[-1])
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    frame_index = first_frame
    try:
        while frame_index <= last_frame:
            ok, frame = capture.read()
            if not ok:
                break
            for index in requests.get(frame_index, []):
                crops[index] = _crop_lr_asd(frame, sequence.boxes[index], crop_scale)
                captured[index] = True
            frame_index += 1
    finally:
        capture.release()
    valid = np.flatnonzero(captured)
    if not len(valid):
        raise RuntimeError(f"could not extract face crops for {sequence.tracklet_id}")
    for index in np.flatnonzero(~captured):
        nearest = valid[np.argmin(np.abs(valid - index))]
        crops[index] = crops[nearest]
    return crops


def _decode_audio(video_path: Path) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for LR-ASD audio extraction")
    command = [
        ffmpeg,
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "pipe:1",
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise RuntimeError("ffmpeg audio extraction failed: " + completed.stderr.decode("utf-8", "replace"))
    audio = np.frombuffer(completed.stdout, dtype=np.int16)
    if not len(audio):
        raise RuntimeError("video contains no decodable audio")
    return audio


def _audio_features(audio: np.ndarray, start_ms: int, visual_frames: int) -> np.ndarray:
    start = max(0, int(round(start_ms * 16)))
    samples = int(math.ceil(visual_frames / ASD_FPS * 16000)) + 400
    segment = audio[start : start + samples]
    if not len(segment):
        segment = np.zeros(samples, dtype=np.int16)
    features = python_speech_features.mfcc(
        segment, 16000, numcep=13, winlen=0.025, winstep=0.010
    ).astype(np.float32)
    expected = visual_frames * 4
    if len(features) < expected:
        if len(features):
            features = np.pad(features, ((0, expected - len(features)), (0, 0)), mode="wrap")
        else:
            features = np.zeros((expected, 13), dtype=np.float32)
    return features[:expected]


def _infer_scores(
    model: LRASDInference,
    device: torch.device,
    audio_features: np.ndarray,
    visual_features: np.ndarray,
    durations: Sequence[int],
) -> np.ndarray:
    frame_count = len(visual_features)
    variants = []
    with torch.inference_mode():
        for duration in durations:
            chunk_frames = duration * int(ASD_FPS)
            scores = []
            for start in range(0, frame_count, chunk_frames):
                end = min(frame_count, start + chunk_frames)
                visual = torch.from_numpy(visual_features[start:end].astype(np.float32)).unsqueeze(0).to(device)
                audio = torch.from_numpy(audio_features[start * 4 : end * 4]).unsqueeze(0).to(device)
                logits = model.logits(audio, visual)
                scores.append(logits[:, 1].detach().cpu().numpy())
            variants.append(np.concatenate(scores)[:frame_count])
    return np.mean(np.stack(variants), axis=0)


def _smooth_scores(values: np.ndarray, radius: int = 2) -> np.ndarray:
    if not len(values):
        return values
    result = np.empty_like(values, dtype=np.float32)
    for index in range(len(values)):
        result[index] = float(np.mean(values[max(0, index - radius) : min(len(values), index + radius + 1)]))
    return result


def _postprocess_activity(values: np.ndarray, threshold: float, min_frames: int, bridge_frames: int) -> np.ndarray:
    active = values >= threshold
    index = 0
    while index < len(active):
        if active[index]:
            index += 1
            continue
        start = index
        while index < len(active) and not active[index]:
            index += 1
        if start > 0 and index < len(active) and index - start <= bridge_frames:
            active[start:index] = True
    index = 0
    while index < len(active):
        if not active[index]:
            index += 1
            continue
        start = index
        while index < len(active) and active[index]:
            index += 1
        if index - start < min_frames:
            active[start:index] = False
    return active


def _assign_activity(
    sequences: Sequence[TrackSequence],
    threshold: float,
    min_frames: int,
    bridge_frames: int,
    exclusive_speaker: bool,
) -> None:
    for sequence in sequences:
        sequence.speaking = _postprocess_activity(
            sequence.raw_scores, threshold, min_frames, bridge_frames
        )
    if not exclusive_speaker:
        return

    # The first product scope assumes that one person speaks at a time. LR-ASD
    # can give weak positive scores to nearby listeners, so retain only the
    # strongest visible candidate in each 40 ms time bucket.
    buckets: Dict[int, List[tuple]] = {}
    for sequence in sequences:
        for index in np.flatnonzero(sequence.speaking).tolist():
            bucket = int(round(int(sequence.times_ms[index]) / ASD_FRAME_MS))
            buckets.setdefault(bucket, []).append((float(sequence.raw_scores[index]), sequence, index))
    for candidates in buckets.values():
        candidates.sort(key=lambda item: (-item[0], item[1].face_id, item[1].tracklet_id))
        for _, sequence, index in candidates[1:]:
            sequence.speaking[index] = False

    # Exclusivity may leave tiny islands on a losing face. Remove those without
    # bridging gaps, which could otherwise recreate simultaneous positives.
    for sequence in sequences:
        sequence.speaking = _postprocess_activity(
            np.where(sequence.speaking, sequence.raw_scores, -np.inf),
            threshold,
            min_frames,
            0,
        )


def _score_records(sequences: Sequence[TrackSequence], source_fps: float) -> List[dict]:
    records = []
    for sequence in sequences:
        for index, timestamp_ms in enumerate(sequence.times_ms.tolist()):
            raw_score = float(sequence.raw_scores[index])
            records.append(
                {
                    "score_id": "",
                    "face_id": sequence.face_id,
                    "tracklet_id": sequence.tracklet_id,
                    "frame_index": int(round(timestamp_ms * source_fps / 1000)),
                    "timestamp_ms": int(timestamp_ms),
                    "raw_score": round(raw_score, 6),
                    "score": round(float(sequence.probabilities[index]), 6),
                    "is_speaking": bool(sequence.speaking[index]),
                }
            )
    records.sort(key=lambda item: (item["timestamp_ms"], item["face_id"], item["tracklet_id"]))
    for index, record in enumerate(records, start=1):
        record["score_id"] = f"asd-score-{index:07d}"
    return records


def _segments(records: Sequence[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = {}
    for record in records:
        if record["is_speaking"]:
            grouped.setdefault(record["face_id"], []).append(record)
    pending = []
    for face_id, items in grouped.items():
        items.sort(key=lambda item: item["timestamp_ms"])
        current = []
        for item in items:
            if current and item["timestamp_ms"] - current[-1]["timestamp_ms"] > ASD_FRAME_MS * 2:
                pending.append((face_id, current))
                current = []
            current.append(item)
        if current:
            pending.append((face_id, current))
    pending.sort(key=lambda item: item[1][0]["timestamp_ms"])
    segments = []
    for index, (face_id, items) in enumerate(pending, start=1):
        segments.append(
            {
                "segment_id": f"active-speaker-{index:05d}",
                "face_id": face_id,
                "start_ms": items[0]["timestamp_ms"],
                "end_ms": items[-1]["timestamp_ms"] + ASD_FRAME_MS,
                "score": round(float(np.mean([item["score"] for item in items])), 6),
                "raw_score_mean": round(float(np.mean([item["raw_score"] for item in items])), 6),
                "tracklet_ids": sorted({item["tracklet_id"] for item in items}),
            }
        )
    return segments


def _draw_active(frame: np.ndarray, box: Sequence[float], face_id: str, probability: float, speaking: bool) -> None:
    x, y, width, height = [int(round(value)) for value in box]
    color = (50, 210, 70) if speaking else (70, 80, 220)
    label = f"{face_id} {'SPEAK' if speaking else 'silent'} {probability:.2f}"
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 3, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.58, 2
    (text_width, text_height), _ = cv2.getTextSize(label, font, scale, thickness)
    top = max(0, y - text_height - 10)
    cv2.rectangle(frame, (x, top), (x + text_width + 10, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 5), font, scale, (15, 15, 15), thickness, cv2.LINE_AA)


def render_active_speaker_video(
    video_path: Path, output_dir: Path, sequences: Sequence[TrackSequence], source: dict
) -> Path:
    silent_path = output_dir / "active_speaker.silent.mp4"
    final_path = output_dir / "active_speaker.mp4"
    silent_path.unlink(missing_ok=True)
    final_path.unlink(missing_ok=True)
    fps = float(source["fps"])
    capture = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(
        str(silent_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(source["width"]), int(source["height"])),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not open the active-speaker video writer")
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = frame_index * 1000 / fps
            for sequence in sequences:
                if timestamp_ms < sequence.times_ms[0] - 20 or timestamp_ms > sequence.times_ms[-1] + 20:
                    continue
                index = int(round((timestamp_ms - sequence.times_ms[0]) / ASD_FRAME_MS))
                if 0 <= index < len(sequence.times_ms):
                    _draw_active(
                        frame,
                        sequence.boxes[index],
                        sequence.face_id,
                        float(sequence.probabilities[index]),
                        bool(sequence.speaking[index]),
                    )
            writer.write(frame)
            frame_index += 1
            if frame_index % 1000 == 0:
                print(f"[asd-render] {frame_index}/{source['frame_count']} frames")
    finally:
        capture.release()
        writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(silent_path),
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(final_path),
        ]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode == 0:
            silent_path.unlink()
            return final_path
    silent_path.replace(final_path)
    return final_path


def analyze_active_speakers(
    json_path: Path,
    output_path: Optional[Path] = None,
    model_name: str = "talkset",
    device_name: str = "auto",
    threshold: float = 0.0,
    min_segment_ms: int = 200,
    bridge_gap_ms: int = 160,
    crop_scale: float = 0.40,
    exclusive_speaker: bool = True,
    render: bool = True,
) -> dict:
    if model_name not in {"ava", "talkset"}:
        raise ValueError("model_name must be ava or talkset")
    if min_segment_ms < 0 or bridge_gap_ms < 0:
        raise ValueError("segment and gap durations cannot be negative")
    json_path = json_path.expanduser().resolve()
    with json_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    video_path = Path(data["source"]["path"])
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    output_path = output_path.expanduser().resolve() if output_path else json_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = _choose_device(device_name)
    print(f"[asd] device={device}, model={model_name}")
    model = _load_model(model_name, device)
    audio = _decode_audio(video_path)
    sequences = _build_sequences(data)
    if not sequences:
        raise RuntimeError("visual_tracks.json contains no retained face tracklets")
    durations = (1, 2, 3, 4, 5, 6)
    started = time.monotonic()
    for index, sequence in enumerate(sequences, start=1):
        print(
            f"[asd] {index}/{len(sequences)} {sequence.face_id} {sequence.tracklet_id} "
            f"{sequence.times_ms[0] / 1000:.2f}-{sequence.times_ms[-1] / 1000:.2f}s"
        )
        sequence.crops = _extract_track_crops(video_path, sequence, crop_scale)
        features = _audio_features(audio, int(sequence.times_ms[0]), len(sequence.crops))
        raw = _infer_scores(model, device, features, sequence.crops, durations)
        sequence.raw_scores = _smooth_scores(raw)
        sequence.probabilities = 1.0 / (1.0 + np.exp(-sequence.raw_scores))
        sequence.crops = None
    _assign_activity(
        sequences,
        threshold,
        max(1, int(math.ceil(min_segment_ms / ASD_FRAME_MS))),
        int(math.floor(bridge_gap_ms / ASD_FRAME_MS)),
        exclusive_speaker,
    )
    elapsed = time.monotonic() - started

    records = _score_records(sequences, float(data["source"]["fps"]))
    segments = _segments(records)
    data["active_speaker_scores"] = records
    data["active_speaker_segments"] = segments
    data["processing"]["active_speaker"] = _model_record(model_name)
    data["processing"]["active_speaker_parameters"] = {
        "device": str(device),
        "model_variant": model_name,
        "fps": ASD_FPS,
        "durations_seconds": list(durations),
        "threshold": threshold,
        "min_segment_ms": min_segment_ms,
        "bridge_gap_ms": bridge_gap_ms,
        "crop_scale": crop_scale,
        "exclusive_speaker": exclusive_speaker,
        "elapsed_seconds": round(elapsed, 3),
    }
    data["statistics"]["active_speaker_tracklets"] = len(sequences)
    data["statistics"]["active_speaker_score_count"] = len(records)
    data["statistics"]["active_speaker_segment_count"] = len(segments)
    data["warnings"] = [item for item in data["warnings"] if not item.startswith("LR-ASD")]
    if not segments:
        data["warnings"].append("LR-ASD found no active visible-speaker segments.")

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output_path)
    if render:
        render_active_speaker_video(video_path, output_path.parent, sequences, data["source"])
    print(
        f"[asd] {len(records)} scores, {len(segments)} speaking segments, "
        f"{elapsed:.2f}s inference preparation/runtime"
    )
    return data
