"""app/runtime/stream_session.py 测试（mock svad/asr/punc，真实 executor + Semaphore）。

异步用例依赖 pytest-asyncio（asyncio_mode=auto）。验证按句 final 产出、seg_id 递增、
时间戳偏移、对齐 words、标点、长句兜底、flush 末句，以及 backend 并发准入与 AudioBuffer。
"""
import asyncio
import types
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.engines.vad_engine import VADEngine
from app.runtime.stream_session import StreamSession, VadOfflineBackend, AudioBuffer


class FakeSVAD:
    """按 feed 调用序号返回脚本化事件；is_final 调用返回 final_events。"""

    def __init__(self, events_by_call=None, final_events=None):
        self.events_by_call = events_by_call or {}
        self.final_events = final_events or []
        self.calls = 0

    def new_cache(self):
        return {}

    def process_chunk(self, arr, cache, is_final):
        if is_final:
            return list(self.final_events)
        ev = self.events_by_call.get(self.calls, [])
        self.calls += 1
        return list(ev)


def _pcm_ms(ms, sr=16000):
    """生成 ms 毫秒的非零 PCM16 字节（int16）。"""
    n = int(ms * sr / 1000)
    return (np.ones(n, dtype="<i2") * 1000).tobytes()


def _make_session(svad, *, enable_words=False, punc=None, max_segment_sec=30,
                  asr_result=None):
    asr = MagicMock()
    asr.transcribe_array.return_value = asr_result or [types.SimpleNamespace(text="hi")]
    executor = ThreadPoolExecutor(max_workers=2)
    sem = asyncio.Semaphore(1)
    s = StreamSession("sid", svad, asr, punc, executor, sem,
                      enable_words=enable_words, max_segment_sec=max_segment_sec)
    s.configure({"audio_fs": 16000})
    return s, asr, executor


async def _collect(agen):
    return [m async for m in agen]


# ─── StreamSession ───

async def test_complete_event_emits_final():
    svad = FakeSVAD(events_by_call={0: [{"type": "complete", "start": 0, "end": 1000}]})
    s, asr, ex = _make_session(svad)
    try:
        msgs = await _collect(s.feed_audio(_pcm_ms(1000)))
        assert len(msgs) == 1
        m = msgs[0]
        assert m["type"] == "final" and m["seg_id"] == 0
        assert m["text"] == "hi"
        assert m["start"] == 0 and m["end"] == 1000
        assert "words" not in m
    finally:
        ex.shutdown(wait=False)


async def test_start_then_end_emits_final():
    svad = FakeSVAD(events_by_call={
        0: [{"type": "start", "start": 0, "end": None}],
        1: [{"type": "end", "start": None, "end": 1000}],
    })
    s, asr, ex = _make_session(svad)
    try:
        assert await _collect(s.feed_audio(_pcm_ms(500))) == []     # 仅 start，无输出
        msgs = await _collect(s.feed_audio(_pcm_ms(500)))           # end → final
        assert len(msgs) == 1
        assert msgs[0]["start"] == 0 and msgs[0]["end"] == 1000
    finally:
        ex.shutdown(wait=False)


async def test_seg_id_increments_across_finals():
    svad = FakeSVAD(events_by_call={
        0: [{"type": "complete", "start": 0, "end": 500}],
        1: [{"type": "complete", "start": 500, "end": 1000}],
    })
    s, asr, ex = _make_session(svad)
    try:
        m0 = await _collect(s.feed_audio(_pcm_ms(500)))
        m1 = await _collect(s.feed_audio(_pcm_ms(500)))
        assert m0[0]["seg_id"] == 0
        assert m1[0]["seg_id"] == 1
    finally:
        ex.shutdown(wait=False)


async def test_flush_emits_pending_segment():
    # 收到 start 未收到 end，flush 时冲刷剩余缓冲
    svad = FakeSVAD(events_by_call={0: [{"type": "start", "start": 0, "end": None}]})
    s, asr, ex = _make_session(svad)
    try:
        await _collect(s.feed_audio(_pcm_ms(800)))
        flushed = await _collect(s.flush())
        assert len(flushed) == 1
        assert flushed[0]["type"] == "final"
        assert flushed[0]["start"] == 0
    finally:
        ex.shutdown(wait=False)


async def test_words_attached_when_enabled():
    word = types.SimpleNamespace(text="a", start_time=0.1, end_time=0.5)
    item = types.SimpleNamespace(text="a", time_stamps=types.SimpleNamespace(items=[word]))
    svad = FakeSVAD(events_by_call={0: [{"type": "complete", "start": 1000, "end": 2000}]})
    s, asr, ex = _make_session(svad, enable_words=True, asr_result=[item])
    try:
        msgs = await _collect(s.feed_audio(_pcm_ms(2000)))
        assert "words" in msgs[0]
        # 偏移叠加：start_ms=1000 -> +1.0s
        assert msgs[0]["words"][0] == {"text": "a", "start": 1.1, "end": 1.5}
    finally:
        ex.shutdown(wait=False)


async def test_punctuation_applied():
    punc = MagicMock()
    punc.restore.return_value = "hi。"
    svad = FakeSVAD(events_by_call={0: [{"type": "complete", "start": 0, "end": 1000}]})
    s, asr, ex = _make_session(svad, punc=punc)
    try:
        msgs = await _collect(s.feed_audio(_pcm_ms(1000)))
        assert msgs[0]["text"] == "hi。"
        punc.restore.assert_called_once()
    finally:
        ex.shutdown(wait=False)


async def test_long_segment_fallback_split():
    # start 后长时间无 end，超过 max_segment_sec 强制切分
    svad = FakeSVAD(events_by_call={0: [{"type": "start", "start": 0, "end": None}]})
    s, asr, ex = _make_session(svad, max_segment_sec=1)  # 1s 阈值
    try:
        msgs = await _collect(s.feed_audio(_pcm_ms(1500)))  # 1.5s > 1s
        assert len(msgs) == 1
        assert msgs[0]["type"] == "final"
        assert msgs[0]["start"] == 0
    finally:
        ex.shutdown(wait=False)


# ─── VadOfflineBackend ───

async def test_backend_acquire_limit_and_release():
    vad = VADEngine()
    vad._model = MagicMock()
    asr = MagicMock()
    asr.align_enabled = False
    backend = VadOfflineBackend(asr, vad, None, max_sessions=2, asr_concurrency=1)
    try:
        assert await backend.acquire() is True
        assert await backend.acquire() is True
        assert await backend.acquire() is False      # 超额
        backend.release(backend.create_session("x"))  # 释放一个
        assert await backend.acquire() is True
    finally:
        backend.shutdown()


def test_backend_capabilities_reflect_align():
    vad = VADEngine()
    vad._model = MagicMock()
    asr_aligned = MagicMock()
    asr_aligned.align_enabled = True
    b1 = VadOfflineBackend(asr_aligned, vad, None)
    assert b1.capabilities["word_timestamps"] is True
    b1.shutdown()

    asr_plain = MagicMock()
    asr_plain.align_enabled = False
    b2 = VadOfflineBackend(asr_plain, vad, None)
    assert b2.capabilities["word_timestamps"] is False
    b2.shutdown()


# ─── 输入校验（audio_fs）───

@pytest.mark.parametrize("bad_fs", [0, -1, 1, 7999, 96001, "abc", {}])
def test_configure_rejects_invalid_audio_fs(bad_fs):
    s = StreamSession("sid", FakeSVAD(), MagicMock(), None,
                      ThreadPoolExecutor(max_workers=1), asyncio.Semaphore(1))
    with pytest.raises(ValueError):
        s.configure({"audio_fs": bad_fs})


@pytest.mark.parametrize("ok_fs", [8000, 16000, 48000, 96000])
def test_configure_accepts_valid_audio_fs(ok_fs):
    s = StreamSession("sid", FakeSVAD(), MagicMock(), None,
                      ThreadPoolExecutor(max_workers=1), asyncio.Semaphore(1))
    s.configure({"audio_fs": ok_fs})
    assert s.audio_fs == ok_fs


# ─── 空帧 / 长静音 / flush 容错 ───

async def test_empty_frame_skips_vad():
    svad = FakeSVAD()
    s, asr, ex = _make_session(svad)
    try:
        assert await _collect(s.feed_audio(b"")) == []
        assert svad.calls == 0                      # 空帧不喂 VAD
    finally:
        ex.shutdown(wait=False)


async def test_idle_buffer_trimmed_during_silence():
    # 无 VAD 事件（长静音）时缓冲应被裁剪，防止无界增长
    svad = FakeSVAD()                               # 永不产出事件
    s, asr, ex = _make_session(svad)
    try:
        for _ in range(10):
            await _collect(s.feed_audio(_pcm_ms(1000)))   # 共 10s 静音
        assert s.buffer.end_ms == 10000
        assert s.buffer.end_ms - s.buffer.base_ms <= 5000  # 仅保留回溯余量
    finally:
        ex.shutdown(wait=False)


async def test_flush_survives_vad_final_failure():
    # VAD final 冲刷抛异常时，仍应冲刷未闭合句的剩余缓冲
    class RaisingFinalSVAD(FakeSVAD):
        def process_chunk(self, arr, cache, is_final):
            if is_final:
                raise RuntimeError("vad boom")
            return super().process_chunk(arr, cache, is_final)

    svad = RaisingFinalSVAD(events_by_call={0: [{"type": "start", "start": 0, "end": None}]})
    s, asr, ex = _make_session(svad)
    try:
        await _collect(s.feed_audio(_pcm_ms(800)))
        flushed = await _collect(s.flush())
        assert len(flushed) == 1
        assert flushed[0]["type"] == "final" and flushed[0]["start"] == 0
    finally:
        ex.shutdown(wait=False)


# ─── AudioBuffer ───

def test_audio_buffer_slice_and_drop():
    buf = AudioBuffer(16000)
    buf.append(np.arange(16000, dtype=np.float32))   # 0..1000ms
    buf.append(np.arange(16000, dtype=np.float32))   # 1000..2000ms
    assert buf.end_ms == 2000
    assert buf.slice_ms(0, 1000).shape[0] == 16000
    assert buf.slice_ms(1000, 2000).shape[0] == 16000

    buf.drop_until_ms(1000)
    assert buf.base_ms == 1000
    assert buf.slice_ms(1000, 2000).shape[0] == 16000


def test_audio_buffer_chunked_append_and_cross_chunk_slice():
    # 分块存储：多次 append 后跨块切片/裁剪结果与单块语义一致
    buf = AudioBuffer(16000)
    for _ in range(10):
        buf.append(np.ones(1600, dtype=np.float32))  # 10 × 100ms
    assert buf.end_ms == 1000
    assert buf.slice_ms(0, 1000).shape[0] == 16000
    assert buf.slice_ms(250, 750).shape[0] == 8000   # 跨块切片

    buf.drop_until_ms(500)
    assert buf.base_ms == 500
    buf.append(np.zeros(1600, dtype=np.float32))     # drop 后继续追加
    assert buf.end_ms == 1100
    assert buf.slice_ms(500, 1100).shape[0] == 9600


def test_audio_buffer_drop_all_then_append():
    buf = AudioBuffer(16000)
    buf.append(np.ones(1600, dtype=np.float32))
    buf.drop_until_ms(100)                           # 全量释放
    assert buf.base_ms == 100 and buf.end_ms == 100
    assert buf.slice_ms(0, 100).shape[0] == 0
    buf.append(np.ones(1600, dtype=np.float32))
    assert buf.end_ms == 200
    assert buf.slice_ms(100, 200).shape[0] == 1600
