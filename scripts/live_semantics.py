from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from .scene_semantics import GroqSceneAnalyzer, analyze_scene_batch


@dataclass(frozen=True)
class SemanticTask:
    keyframe_id: str
    scene_id: str
    timestamp_ms: int
    metadata: dict
    image_path: Path


class LiveSemanticWorker:
    """Bounded background consumer for live keyframe semantics."""

    def __init__(
        self,
        analyzer: GroqSceneAnalyzer,
        cache_dir: Path,
        *,
        on_result: Callable[[SemanticTask, dict, str, bool], None],
        on_error: Callable[[SemanticTask, Exception], None],
        on_drop: Callable[[SemanticTask], None],
        on_finished: Callable[[], None],
        queue_size: int = 8,
    ) -> None:
        if queue_size < 1:
            raise ValueError("semantic queue size must be positive")
        self.analyzer = analyzer
        self.cache_dir = cache_dir
        self.on_result = on_result
        self.on_error = on_error
        self.on_drop = on_drop
        self.on_finished = on_finished
        self.tasks: queue.Queue[SemanticTask] = queue.Queue(maxsize=queue_size)
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
        self._thread = threading.Thread(
            target=self._run,
            name="live-scene-semantics",
            daemon=True,
        )
        self._thread.start()

    def submit(self, task: SemanticTask) -> None:
        try:
            self.tasks.put_nowait(task)
            return
        except queue.Full:
            pass
        try:
            dropped = self.tasks.get_nowait()
            self.tasks.task_done()
            self.on_drop(dropped)
        except queue.Empty:
            pass
        self.tasks.put_nowait(task)

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
                task = self.tasks.get_nowait()
            except queue.Empty:
                return
            self.tasks.task_done()
            self.on_drop(task)

    def _run(self) -> None:
        provider_failed = False
        while not self._stopping.is_set():
            try:
                task = self.tasks.get(timeout=0.1)
            except queue.Empty:
                if self._closing.is_set():
                    break
                continue
            self._running = True
            try:
                raw, request_hash, cache_hit = analyze_scene_batch(
                    self.analyzer,
                    [task.metadata],
                    [task.image_path],
                    self.cache_dir,
                )
                if len(raw) != 1:
                    raise ValueError("live scene analysis must return exactly one keyframe")
                self.on_result(task, raw[0], request_hash, cache_hit)
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.on_error(task, error)
                provider_failed = isinstance(error, OSError)
            finally:
                self._running = False
                self.tasks.task_done()
            if provider_failed:
                self._discard_pending()
                break
        self._finished.set()
        self.on_finished()
