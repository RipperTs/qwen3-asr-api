import types
from concurrent.futures import ThreadPoolExecutor

from app.runtime.offline_batch_scheduler import (
    ASRBatchScheduler,
    ChunkJob,
    ChunkResult,
)


def test_scheduler_batches_chunks_from_multiple_tasks():
    class ASR:
        align_enabled = True

        def __init__(self):
            self.calls = []

        def batch_transcribe(self, audio_paths, language=None):
            self.calls.append((list(audio_paths), language))
            return [types.SimpleNamespace(text=f"txt:{path}") for path in audio_paths]

    asr = ASR()
    scheduler = ASRBatchScheduler(asr, batch_size=3, batch_wait_ms=0)
    jobs = [
        ChunkJob("task-a", 0, "a0.wav", 0.0, 1.0, "zh", False),
        ChunkJob("task-b", 0, "b0.wav", 1.0, 1.0, "zh", False),
        ChunkJob("task-a", 1, "a1.wav", 2.0, 1.0, "zh", True),
    ]

    results = scheduler.transcribe_ready(jobs)

    assert asr.calls == [(["a0.wav", "b0.wav", "a1.wav"], "zh")]
    assert results == [
        ChunkResult("task-a", 0, [types.SimpleNamespace(text="txt:a0.wav")], None),
        ChunkResult("task-b", 0, [types.SimpleNamespace(text="txt:b0.wav")], None),
        ChunkResult("task-a", 1, [types.SimpleNamespace(text="txt:a1.wav")], None),
    ]


def test_scheduler_groups_by_language_to_preserve_asr_language_semantics():
    class ASR:
        def __init__(self):
            self.calls = []

        def batch_transcribe(self, audio_paths, language=None):
            self.calls.append((list(audio_paths), language))
            return [types.SimpleNamespace(text=f"{language}:{path}") for path in audio_paths]

    asr = ASR()
    scheduler = ASRBatchScheduler(asr, batch_size=8, batch_wait_ms=0)
    jobs = [
        ChunkJob("task-a", 0, "a.wav", 0.0, 1.0, "zh", False),
        ChunkJob("task-b", 0, "b.wav", 0.0, 1.0, "en", False),
        ChunkJob("task-c", 0, "c.wav", 0.0, 1.0, "zh", False),
    ]

    results = scheduler.transcribe_ready(jobs)

    assert asr.calls == [
        (["a.wav", "c.wav"], "zh"),
        (["b.wav"], "en"),
    ]
    assert [(r.task_id, r.index, r.error) for r in results] == [
        ("task-a", 0, None),
        ("task-c", 0, None),
        ("task-b", 0, None),
    ]


def test_scheduler_skips_cancelled_jobs_before_asr_call():
    class ASR:
        def __init__(self):
            self.calls = []

        def batch_transcribe(self, audio_paths, language=None):
            self.calls.append(list(audio_paths))
            return [types.SimpleNamespace(text=path) for path in audio_paths]

    cancelled = {"task-b"}
    scheduler = ASRBatchScheduler(
        ASR(),
        batch_size=4,
        batch_wait_ms=0,
        is_cancelled=lambda task_id: task_id in cancelled,
    )
    jobs = [
        ChunkJob("task-a", 0, "a.wav", 0.0, 1.0, None, False),
        ChunkJob("task-b", 0, "b.wav", 0.0, 1.0, None, False),
    ]

    results = scheduler.transcribe_ready(jobs)

    assert [(r.task_id, r.cancelled, r.error) for r in results] == [
        ("task-b", True, "ASR 任务已取消"),
        ("task-a", False, None),
    ]
    assert scheduler.asr.calls == [["a.wav"]]


def test_scheduler_returns_chunk_errors_when_batch_result_count_mismatches():
    class ASR:
        def batch_transcribe(self, audio_paths, language=None):
            return [types.SimpleNamespace(text="only-one")]

    scheduler = ASRBatchScheduler(ASR(), batch_size=4, batch_wait_ms=0)
    jobs = [
        ChunkJob("task-a", 0, "a.wav", 0.0, 1.0, None, False),
        ChunkJob("task-b", 0, "b.wav", 0.0, 1.0, None, False),
    ]

    results = scheduler.transcribe_ready(jobs)

    assert results == [
        ChunkResult("task-a", 0, None, "批次结果数不匹配: 期望 2, 得到 1"),
        ChunkResult("task-b", 0, None, "批次结果数不匹配: 期望 2, 得到 1"),
    ]


def test_scheduler_returns_chunk_errors_when_asr_batch_raises():
    class ASR:
        def batch_transcribe(self, audio_paths, language=None):
            raise RuntimeError("boom")

    scheduler = ASRBatchScheduler(ASR(), batch_size=4, batch_wait_ms=0)
    jobs = [
        ChunkJob("task-a", 0, "a.wav", 0.0, 1.0, None, False),
        ChunkJob("task-b", 0, "b.wav", 0.0, 1.0, None, False),
    ]

    results = scheduler.transcribe_ready(jobs)

    assert results == [
        ChunkResult("task-a", 0, None, "批次推理失败: boom"),
        ChunkResult("task-b", 0, None, "批次推理失败: boom"),
    ]


def test_scheduler_submit_waits_briefly_and_batches_concurrent_callers():
    class ASR:
        def __init__(self):
            self.calls = []

        def batch_transcribe(self, audio_paths, language=None):
            self.calls.append(list(audio_paths))
            return [types.SimpleNamespace(text=path) for path in audio_paths]

    scheduler = ASRBatchScheduler(ASR(), batch_size=2, batch_wait_ms=100)
    job_a = ChunkJob("task-a", 0, "a.wav", 0.0, 1.0, None, False)
    job_b = ChunkJob("task-b", 0, "b.wav", 0.0, 1.0, None, False)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(scheduler.submit, job_a, timeout=2)
            future_b = executor.submit(scheduler.submit, job_b, timeout=2)
            result_a = future_a.result(timeout=3)
            result_b = future_b.result(timeout=3)
    finally:
        scheduler.shutdown()

    assert ASRBatchScheduler.result_text(result_a) == "a.wav"
    assert ASRBatchScheduler.result_text(result_b) == "b.wav"
    assert len(scheduler.asr.calls) == 1
    assert set(scheduler.asr.calls[0]) == {"a.wav", "b.wav"}


def test_scheduler_submit_many_preserves_single_task_batching():
    class ASR:
        def __init__(self):
            self.calls = []

        def batch_transcribe(self, audio_paths, language=None):
            self.calls.append((list(audio_paths), language))
            return [types.SimpleNamespace(text=path) for path in audio_paths]

    scheduler = ASRBatchScheduler(ASR(), batch_size=8, batch_wait_ms=100)
    jobs = [
        ChunkJob("task-a", 0, "a0.wav", 0.0, 1.0, "zh", False),
        ChunkJob("task-a", 1, "a1.wav", 1.0, 1.0, "zh", False),
    ]

    try:
        results = scheduler.submit_many(jobs, timeout=2)
    finally:
        scheduler.shutdown()

    assert [ASRBatchScheduler.result_text(result) for result in results] == [
        "a0.wav",
        "a1.wav",
    ]
    assert scheduler.asr.calls == [(["a0.wav", "a1.wav"], "zh")]
