from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .speaker_face_binding import (
    _interval_duration,
    _load_speech_timeline,
    _maximum_overlap_assignment,
    _overlap_evidence,
)


@dataclass(frozen=True)
class ActiveSpeakerChunk:
    chunk_id: str
    start_ms: int
    publish_start_ms: int
    end_ms: int
    observations: Dict[str, Tuple[dict, ...]]


class FaceSpeakingStateMachine:
    """Turn frame activity into independent start/continue/end events per Face-ID."""

    def __init__(self, *, start_confirm_ms: int = 200, end_silence_ms: int = 1000) -> None:
        if start_confirm_ms < 40 or end_silence_ms < 40:
            raise ValueError("speaking confirmation durations must be at least 40 ms")
        self.start_confirm_ms = start_confirm_ms
        self.end_silence_ms = end_silence_ms
        self.states: Dict[str, dict] = {}

    @staticmethod
    def _deduplicate(records: Sequence[dict]) -> Dict[str, List[dict]]:
        grouped: Dict[str, Dict[int, dict]] = {}
        for item in records:
            face_id = item.get("face_id")
            if not face_id:
                continue
            timestamp_ms = int(item["timestamp_ms"])
            current = grouped.setdefault(face_id, {}).get(timestamp_ms)
            if current is None or bool(item.get("is_speaking")):
                grouped[face_id][timestamp_ms] = item
        return {
            face_id: [items[timestamp] for timestamp in sorted(items)]
            for face_id, items in grouped.items()
        }

    def update(
        self,
        records: Sequence[dict],
        *,
        publish_start_ms: int,
        end_ms: int,
    ) -> List[dict]:
        grouped = self._deduplicate(
            [
                item
                for item in records
                if publish_start_ms <= int(item["timestamp_ms"]) < end_ms
            ]
        )
        events = []
        for face_id in sorted(set(self.states) | set(grouped)):
            state = self.states.setdefault(
                face_id,
                {
                    "status": "silent",
                    "positive_since_ms": None,
                    "negative_since_ms": None,
                },
            )
            samples = list(grouped.get(face_id, ()))
            if state["status"] == "silent" and not samples:
                state["positive_since_ms"] = None
            if state["status"] == "speaking":
                tail_start = (
                    max(publish_start_ms, int(samples[-1]["timestamp_ms"]) + 40)
                    if samples
                    else publish_start_ms
                )
                if tail_start < end_ms:
                    samples.append(
                        {
                            "face_id": face_id,
                            "timestamp_ms": tail_start,
                            "is_speaking": False,
                        }
                    )

            started = False
            ended = False
            positive_seen = False
            for item in samples:
                timestamp_ms = int(item["timestamp_ms"])
                sample_end_ms = min(end_ms, timestamp_ms + 40)
                if bool(item.get("is_speaking")):
                    positive_seen = True
                    state["negative_since_ms"] = None
                    if state["status"] == "silent":
                        if state["positive_since_ms"] is None:
                            state["positive_since_ms"] = timestamp_ms
                        if sample_end_ms - state["positive_since_ms"] >= self.start_confirm_ms:
                            state["status"] = "speaking"
                            started = True
                            events.append(
                                {
                                    "face_id": face_id,
                                    "phase": "started",
                                    "timestamp_ms": state["positive_since_ms"],
                                }
                            )
                    continue

                state["positive_since_ms"] = None
                if state["status"] != "speaking":
                    continue
                if state["negative_since_ms"] is None:
                    state["negative_since_ms"] = timestamp_ms
                if end_ms - state["negative_since_ms"] >= self.end_silence_ms:
                    state["status"] = "silent"
                    ended = True
                    events.append(
                        {
                            "face_id": face_id,
                            "phase": "ended",
                            "timestamp_ms": state["negative_since_ms"],
                        }
                    )
                    state["negative_since_ms"] = None

            if state["status"] == "speaking" and positive_seen and not started and not ended:
                first_positive = next(
                    int(item["timestamp_ms"])
                    for item in samples
                    if bool(item.get("is_speaking"))
                )
                events.append(
                    {
                        "face_id": face_id,
                        "phase": "continued",
                        "timestamp_ms": first_positive,
                    }
                )
        order = {"ended": 0, "started": 1, "continued": 2}
        return sorted(events, key=lambda item: (item["timestamp_ms"], order[item["phase"]]))


def _sequence_specs(
    chunk: ActiveSpeakerChunk,
    source_fps: float,
    *,
    max_observation_gap_ms: int = 500,
) -> List[dict]:
    """Interpolate 5 FPS face boxes onto LR-ASD's 25 FPS time grid."""
    specs = []
    for face_id, raw in sorted(chunk.observations.items()):
        observations = sorted(raw, key=lambda item: item["timestamp_ms"])
        runs: List[List[dict]] = []
        for item in observations:
            if runs and item["timestamp_ms"] - runs[-1][-1]["timestamp_ms"] <= max_observation_gap_ms:
                runs[-1].append(item)
            else:
                runs.append([item])
        for run_index, run in enumerate(runs, start=1):
            if len(run) < 2:
                continue
            source_times = np.asarray([item["timestamp_ms"] for item in run], dtype=np.float64)
            if source_times[-1] - source_times[0] < 160:
                continue
            source_boxes = np.asarray([item["bbox_px"] for item in run], dtype=np.float64)
            first = int(math.ceil(source_times[0] / 40) * 40)
            last = int(math.floor(source_times[-1] / 40) * 40)
            if last < first:
                continue
            times_ms = np.arange(first, last + 1, 40, dtype=np.int64)
            boxes = np.column_stack(
                [np.interp(times_ms, source_times, source_boxes[:, column]) for column in range(4)]
            )
            specs.append(
                {
                    "tracklet_id": f"{chunk.chunk_id}:{face_id}:{run_index:02d}",
                    "face_id": face_id,
                    "times_ms": times_ms,
                    "source_frames": np.rint(times_ms * source_fps / 1000).astype(np.int64),
                    "boxes": boxes,
                }
            )
    return specs


class OnlineSpeakerBinder:
    """Recompute conservative A/B/C/D-to-Face bindings from accumulated segments."""

    def __init__(
        self,
        timeline_path: Path,
        duration_ms: int,
        *,
        timeline_offset_ms: int = 0,
        min_evidence_ms: int = 3000,
        min_speaker_coverage: float = 0.50,
        min_margin: float = 0.40,
    ) -> None:
        if min_evidence_ms < 0:
            raise ValueError("binding evidence must not be negative")
        if not 0 <= min_speaker_coverage <= 1 or not 0 <= min_margin <= 1:
            raise ValueError("binding coverage and margin must be between zero and one")
        self.timeline_path = timeline_path.expanduser().resolve()
        if not self.timeline_path.is_file():
            raise FileNotFoundError(self.timeline_path)
        self.duration_ms = duration_ms
        self.timeline_offset_ms = timeline_offset_ms
        self.min_evidence_ms = min_evidence_ms
        self.min_speaker_coverage = min_speaker_coverage
        self.min_margin = min_margin

    def associations(self, active_segments: Sequence[dict]) -> List[dict]:
        speech = _load_speech_timeline(
            self.timeline_path,
            self.duration_ms,
            self.timeline_offset_ms,
        )
        active_by_face: Dict[str, List[dict]] = {}
        for segment in active_segments:
            face_id = segment.get("face_id")
            if face_id:
                active_by_face.setdefault(face_id, []).append(segment)
        speakers, faces = sorted(speech), sorted(active_by_face)
        overlaps: Dict[Tuple[str, str], int] = {}
        evidence: Dict[Tuple[str, str], List[dict]] = {}
        for speaker in speakers:
            for face_id in faces:
                duration, items = _overlap_evidence(speech[speaker], active_by_face[face_id])
                overlaps[(speaker, face_id)] = duration
                evidence[(speaker, face_id)] = items
        assignment = _maximum_overlap_assignment(speakers, faces, overlaps)
        results = []
        for speaker in speakers:
            speech_duration = _interval_duration(
                [(item.start_ms, item.end_ms) for item in speech[speaker]]
            )
            face_id = assignment.get(speaker)
            overlap = overlaps.get((speaker, face_id), 0) if face_id else 0
            alternatives = sorted(
                (
                    (duration, candidate)
                    for (label, candidate), duration in overlaps.items()
                    if label == speaker and candidate != face_id
                ),
                reverse=True,
            )
            runner_up = alternatives[0][0] if alternatives else 0
            coverage = overlap / speech_duration if speech_duration else 0.0
            margin = (overlap - runner_up) / speech_duration if speech_duration else 0.0
            if face_id is None:
                status = "offscreen"
            elif overlap < self.min_evidence_ms or coverage < self.min_speaker_coverage:
                status = "candidate"
            elif margin < self.min_margin:
                status = "ambiguous"
            else:
                status = "confirmed"
            results.append(
                {
                    "speaker_label": speaker,
                    "face_id": face_id,
                    "status": status,
                    "evidence_duration_ms": overlap,
                    "speaker_coverage": round(coverage, 6),
                    "margin": round(margin, 6),
                    "evidence": evidence.get((speaker, face_id), []) if face_id else [],
                }
            )
        return results


class LiveActiveSpeakerWorker:
    """Bounded LR-ASD chunk worker; model and audio decoding stay off the frame thread."""

    def __init__(
        self,
        video_path: Path,
        source_fps: float,
        *,
        on_result: Callable[[ActiveSpeakerChunk, object, float], None],
        on_error: Callable[[Optional[ActiveSpeakerChunk], Exception], None],
        on_drop: Callable[[ActiveSpeakerChunk], None],
        on_ready: Callable[[str], None],
        on_finished: Callable[[], None],
        model_name: str = "talkset",
        device_name: str = "auto",
        threshold: float = 0.0,
        min_segment_ms: int = 200,
        bridge_gap_ms: int = 160,
        crop_scale: float = 0.40,
        queue_size: int = 2,
        processor: Optional[Callable[[ActiveSpeakerChunk], object]] = None,
    ) -> None:
        if model_name not in {"ava", "talkset"}:
            raise ValueError("active-speaker model must be ava or talkset")
        if queue_size < 1 or min_segment_ms < 0 or bridge_gap_ms < 0:
            raise ValueError("active-speaker queue and durations are invalid")
        self.video_path = video_path
        self.source_fps = source_fps
        self.on_result = on_result
        self.on_error = on_error
        self.on_drop = on_drop
        self.on_ready = on_ready
        self.on_finished = on_finished
        self.model_name = model_name
        self.device_name = device_name
        self.threshold = threshold
        self.min_segment_ms = min_segment_ms
        self.bridge_gap_ms = bridge_gap_ms
        self.crop_scale = crop_scale
        self.processor = processor
        self.tasks: queue.Queue[ActiveSpeakerChunk] = queue.Queue(maxsize=queue_size)
        self._closing = threading.Event()
        self._stopping = threading.Event()
        self._finished = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def pending(self) -> int:
        return self.tasks.qsize() + int(self._running)

    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="live-active-speaker", daemon=True)
        self._thread.start()

    def submit(self, chunk: ActiveSpeakerChunk) -> None:
        try:
            self.tasks.put_nowait(chunk)
            return
        except queue.Full:
            pass
        try:
            dropped = self.tasks.get_nowait()
            self.tasks.task_done()
            self.on_drop(dropped)
        except queue.Empty:
            pass
        self.tasks.put_nowait(chunk)

    def close(self) -> None:
        self._closing.set()

    def stop(self) -> None:
        self._stopping.set()
        self._closing.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=5)

    def _discard_pending(self) -> None:
        while True:
            try:
                chunk = self.tasks.get_nowait()
            except queue.Empty:
                return
            self.tasks.task_done()
            self.on_drop(chunk)

    def _default_processor(self):
        from .active_speaker import (
            ASD_FRAME_MS,
            TrackSequence,
            _assign_activity,
            _audio_features,
            _choose_device,
            _decode_audio,
            _extract_track_crops,
            _infer_scores,
            _load_model,
            _score_records,
            _segments,
            _smooth_scores,
        )

        device = _choose_device(self.device_name)
        model = _load_model(self.model_name, device)
        audio = _decode_audio(self.video_path)
        self.on_ready(str(device))

        def process(chunk: ActiveSpeakerChunk) -> dict:
            sequences = []
            for spec in _sequence_specs(chunk, self.source_fps):
                sequence = TrackSequence(**spec)
                sequence.crops = _extract_track_crops(
                    self.video_path,
                    sequence,
                    self.crop_scale,
                )
                features = _audio_features(audio, int(sequence.times_ms[0]), len(sequence.crops))
                raw = _infer_scores(model, device, features, sequence.crops, (1, 2, 3))
                sequence.raw_scores = _smooth_scores(raw)
                sequence.probabilities = 1.0 / (1.0 + np.exp(-sequence.raw_scores))
                sequence.crops = None
                sequences.append(sequence)
            if not sequences:
                return {"segments": [], "activity": []}
            _assign_activity(
                sequences,
                self.threshold,
                max(1, int(math.ceil(self.min_segment_ms / ASD_FRAME_MS))),
                int(math.floor(self.bridge_gap_ms / ASD_FRAME_MS)),
                True,
            )
            records = _score_records(sequences, self.source_fps)
            return {"segments": _segments(records), "activity": records}

        return process

    def _run(self) -> None:
        try:
            processor = self.processor
            if processor is None:
                processor = self._default_processor()
            else:
                self.on_ready("test")
            while not self._stopping.is_set():
                try:
                    chunk = self.tasks.get(timeout=0.1)
                except queue.Empty:
                    if self._closing.is_set():
                        break
                    continue
                self._running = True
                started = time.monotonic()
                try:
                    result = processor(chunk)
                    if not isinstance(result, dict):
                        result = {"segments": list(result), "activity": []}
                    self.on_result(chunk, result, time.monotonic() - started)
                except Exception as error:
                    self.on_error(chunk, error)
                    self.tasks.task_done()
                    self._running = False
                    self._discard_pending()
                    break
                self.tasks.task_done()
                self._running = False
        except Exception as error:
            self.on_error(None, error)
            self._discard_pending()
        finally:
            self._finished.set()
            self.on_finished()
