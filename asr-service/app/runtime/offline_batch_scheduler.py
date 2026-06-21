from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkJob:
    task_id: str
    index: int
    path: str
    offset_sec: float
    duration_sec: float
    language: str | None
    split_after: bool


@dataclass(frozen=True)
class ChunkResult:
    task_id: str
    index: int
    results: list | None
    error: str | None = None
    cancelled: bool = False


@dataclass
class _PendingJob:
    job: ChunkJob
    group_id: str
    done: threading.Event = field(default_factory=threading.Event)
    result: ChunkResult | None = None


@dataclass
class BatchStats:
    batches: int = 0
    chunks: int = 0
    batch_sizes: list[int] = field(default_factory=list)


class ASRBatchScheduler:
    """全局离线 ASR 合批调度核心。

    本类只负责把多个任务的 chunk 组成安全 batch 并调用单个 ASR 引擎入口；
    不拥有任务状态、文件清理或后处理职责。
    """

    def __init__(
        self,
        asr,
        *,
        batch_size: int,
        batch_wait_ms: int,
        is_cancelled: Callable[[str], bool] | None = None,
    ):
        self.asr = asr
        self.batch_size = max(1, int(batch_size))
        self.batch_wait_ms = max(0, int(batch_wait_ms))
        self.is_cancelled = is_cancelled or (lambda _task_id: False)
        self.stats = BatchStats()
        self._pending: list[_PendingJob] = []
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._run,
            name="offline-asr-batch",
            daemon=True,
        )
        self._worker.start()

    def submit(self, job: ChunkJob, timeout: float | None = None) -> ChunkResult:
        """提交一个 chunk 并阻塞等待 ASR 结果。"""
        pending = _PendingJob(job=job, group_id=uuid.uuid4().hex)
        with self._wake:
            if self._stop_event.is_set():
                return ChunkResult(job.task_id, job.index, None, "ASR 调度器已停止", True)
            self._pending.append(pending)
            self._wake.notify()

        if not pending.done.wait(timeout):
            return ChunkResult(job.task_id, job.index, None, "ASR 调度等待超时")
        return (
            pending.result
            or ChunkResult(job.task_id, job.index, None, "ASR 调度无结果")
        )

    def submit_many(
        self,
        jobs: list[ChunkJob],
        timeout: float | None = None,
    ) -> list[ChunkResult]:
        """提交同一调用方的一组 chunk，并尽量保持它们在同一 ASR batch 内。"""
        if not jobs:
            return []

        group_id = uuid.uuid4().hex
        pending_jobs = [_PendingJob(job=job, group_id=group_id) for job in jobs]
        with self._wake:
            if self._stop_event.is_set():
                return [
                    ChunkResult(job.task_id, job.index, None, "ASR 调度器已停止", True)
                    for job in jobs
                ]
            self._pending.extend(pending_jobs)
            self._wake.notify()

        deadline = None if timeout is None else time.monotonic() + timeout
        results = []
        for pending in pending_jobs:
            wait_timeout = (
                None if deadline is None else max(0, deadline - time.monotonic())
            )
            if not pending.done.wait(wait_timeout):
                results.append(
                    ChunkResult(
                        pending.job.task_id,
                        pending.job.index,
                        None,
                        "ASR 调度等待超时",
                    )
                )
                continue
            results.append(
                pending.result
                or ChunkResult(
                    pending.job.task_id,
                    pending.job.index,
                    None,
                    "ASR 调度无结果",
                )
            )
        return results

    def shutdown(self):
        """停止后台调度线程，并唤醒所有仍在等待的调用方。"""
        self._stop_event.set()
        with self._wake:
            for pending in self._pending:
                pending.result = ChunkResult(
                    pending.job.task_id,
                    pending.job.index,
                    None,
                    "ASR 调度器已停止",
                    True,
                )
                pending.done.set()
            self._pending.clear()
            self._wake.notify_all()
        if self._worker.is_alive():
            self._worker.join(timeout=2)

    @staticmethod
    def result_text(result: ChunkResult) -> str:
        if not result.results:
            return ""
        item = result.results[0]
        if isinstance(item, dict):
            return item.get("text", "")
        return getattr(item, "text", "")

    def transcribe_ready(self, jobs: list[ChunkJob]) -> list[ChunkResult]:
        """转写一组已准备好的 chunk。

        准确度优先：不同 language 的 chunk 不混批，避免改变上游 ASR 的语言语义。
        """
        results: list[ChunkResult] = []
        groups: dict[str | None, list[ChunkJob]] = {}
        for job in jobs:
            if self.is_cancelled(job.task_id):
                results.append(
                    ChunkResult(job.task_id, job.index, None, "ASR 任务已取消", True)
                )
                continue
            groups.setdefault(job.language, []).append(job)

        for group in groups.values():
            for start in range(0, len(group), self.batch_size):
                batch = group[start:start + self.batch_size]
                results.extend(self._transcribe_batch(batch))
        return results

    def _transcribe_batch(self, jobs: list[ChunkJob]) -> list[ChunkResult]:
        if not jobs:
            return []

        audio_paths = [job.path for job in jobs]
        language = jobs[0].language
        try:
            batch_results = self.asr.batch_transcribe(
                audio_paths=audio_paths,
                language=language,
            )
        except Exception as exc:
            logger.exception("离线 ASR 合批推理失败")
            error = f"批次推理失败: {exc}"
            return [ChunkResult(job.task_id, job.index, None, error) for job in jobs]

        if len(batch_results) != len(jobs):
            error = f"批次结果数不匹配: 期望 {len(jobs)}, 得到 {len(batch_results)}"
            return [ChunkResult(job.task_id, job.index, None, error) for job in jobs]

        self.stats.batches += 1
        self.stats.chunks += len(jobs)
        self.stats.batch_sizes.append(len(jobs))
        logger.info(
            "离线 ASR 合批完成: batch_size=%s, language=%s, total_batches=%s",
            len(jobs),
            language or "auto",
            self.stats.batches,
        )
        return [
            ChunkResult(job.task_id, job.index, [result], None)
            for job, result in zip(jobs, batch_results)
        ]

    def _run(self):
        while not self._stop_event.is_set():
            pending_batch = self._take_next_batch()
            if not pending_batch:
                continue

            jobs = [pending.job for pending in pending_batch]
            results = self.transcribe_ready(jobs)
            by_key = {
                (result.task_id, result.index): result
                for result in results
            }

            for pending in pending_batch:
                key = (pending.job.task_id, pending.job.index)
                pending.result = by_key.get(key) or ChunkResult(
                    pending.job.task_id,
                    pending.job.index,
                    None,
                    "ASR 任务已取消",
                    True,
                )
                pending.done.set()

    def _take_next_batch(self) -> list[_PendingJob]:
        deadline = None
        with self._wake:
            while not self._stop_event.is_set() and not self._pending:
                self._wake.wait()
            if self._stop_event.is_set():
                return []

            deadline = time.monotonic() + self.batch_wait_ms / 1000.0
            while (
                len(self._pending) < self.batch_size
                and self.batch_wait_ms > 0
                and not self._stop_event.is_set()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._wake.wait(timeout=remaining)

            batch = self._pop_next_batch_locked()
            return batch

    def _pop_next_batch_locked(self) -> list[_PendingJob]:
        first = self._pending[0]
        # Keep chunks submitted by the same caller adjacent before filling spare slots.
        group = [item for item in self._pending if item.group_id == first.group_id]
        batch = group[:self.batch_size]
        selected = {id(item) for item in batch}
        remaining_slots = self.batch_size - len(batch)
        if remaining_slots > 0:
            for item in self._pending:
                if id(item) in selected:
                    continue
                batch.append(item)
                selected.add(id(item))
                remaining_slots -= 1
                if remaining_slots <= 0:
                    break

        self._pending = [item for item in self._pending if id(item) not in selected]
        return batch
