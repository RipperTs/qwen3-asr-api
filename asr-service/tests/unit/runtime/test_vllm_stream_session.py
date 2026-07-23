"""vLLM 流式会话/后端单元测试（不依赖 vLLM / GPU）。

EnergyEndpointer 能量端点事件、VllmStreamSession 信封序列（mock 引擎）、
VllmStreamBackend 准入/释放/能力。在 standard venv 即可运行（模块不 import vllm）。
"""
import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from app.runtime.vllm_stream_session import (
    EnergyEndpointer, VllmStreamSession, VllmStreamBackend,
)

SR = 16000


def _pcm16_bytes(amp, ms):
    n = int(SR * ms / 1000)
    return (np.full(n, amp, dtype=np.float32) * 32767).astype("<i2").tobytes()


def _voice(ms=200):
    return _pcm16_bytes(0.2, ms)      # rms≈-14 dBFS ≥ -45 → 语音


def _silence(ms=200):
    return _pcm16_bytes(0.0, ms)      # -120 dBFS → 静音


async def _collect(agen):
    return [m async for m in agen]


# ─── EnergyEndpointer ───

def test_endpointer_start_then_end():
    ep = EnergyEndpointer(energy_floor_dbfs=-45.0, end_silence_ms=800)
    v = np.full(3200, 0.2, dtype=np.float32)      # 200ms 语音
    s = np.zeros(3200, dtype=np.float32)          # 200ms 静音

    assert ep.process(s, 200) == []               # 静音不起句
    ev = ep.process(v, 200)
    assert ev == [{"type": "start", "start": 200}]
    assert ep.in_speech is True
    assert ep.process(v, 200) == []               # 句内无事件
    # 尾静音累计：800ms（4×200）才判停
    assert ep.process(s, 200) == []
    assert ep.process(s, 200) == []
    assert ep.process(s, 200) == []
    ev = ep.process(s, 200)
    assert ev == [{"type": "end", "end": 1400}]
    assert ep.in_speech is False


def test_endpointer_reset():
    ep = EnergyEndpointer()
    ep.process(np.full(3200, 0.2, dtype=np.float32), 200)
    assert ep.in_speech is True
    ep.reset()
    assert ep.in_speech is False


# ─── mock 引擎 ───

class _MockEngine:
    """累积式 mock：每次 feed 追加一个字符，finish 补句号。"""

    def __init__(self):
        self.feeds = 0
        self.feed_sizes = []
        self.finishes = 0
        self.new_states = 0

    def new_state(self, language=None, chunk_size_sec=None):
        self.new_states += 1
        return SimpleNamespace(text="", language=language or "Chinese",
                               _acc="", chunk_size_sec=chunk_size_sec)

    def feed(self, arr, state):
        self.feeds += 1
        self.feed_sizes.append(int(arr.size))
        state._acc += "字"
        state.text = state._acc
        return state.text, state.language

    def finish(self, state):
        self.finishes += 1
        state.text = state._acc + "。"
        return state.text, state.language


class _SilenceAwareMockEngine(_MockEngine):
    """静音 feed 不新增文本，用来覆盖尾静音收尾边界。"""

    def feed(self, arr, state):
        self.feeds += 1
        self.feed_sizes.append(int(arr.size))
        if arr.size and float(np.max(np.abs(arr))) > 0:
            state._acc += "字"
        state.text = state._acc
        return state.text, state.language


class _FakeSpeakerEngine:
    """按调用顺序返回正交 embedding，并记录实际收到的 PCM。"""

    def __init__(self, vec_indices=None):
        self.vec_indices = list(vec_indices or [])
        self.calls = 0
        self.audio_sizes = []

    def embed_segment(self, audio):
        self.audio_sizes.append(int(audio.size))
        idx = self.vec_indices[self.calls] if self.calls < len(self.vec_indices) else 0
        self.calls += 1
        embedding = np.zeros(192, dtype=np.float32)
        embedding[idx] = 1.0
        return embedding


class _BoomSpeakerEngine:
    def embed_segment(self, audio):
        raise RuntimeError("speaker boom")


class _FakeSpeakerStore:
    def __init__(self):
        self.cache_version = 0


class _FakeSpeakerService:
    """按调用顺序返回名字；None 表示本次未命中。"""

    def __init__(self, names=("张三",)):
        self.names = list(names)
        self.calls = 0
        self.store = _FakeSpeakerStore()

    def map_clusters(self, clusters, *, id_threshold=None, id_margin=None):
        index = min(self.calls, len(self.names) - 1)
        name = self.names[index]
        self.calls += 1
        return [{
            "label": clusters[0]["label"],
            "speaker_id": "x" * 32 if name is not None else None,
            "name": name,
            "score": 0.8 if name is not None else None,
        }]


def _make_session(engine=None, **bk):
    eng = engine or _MockEngine()
    backend = VllmStreamBackend(eng, **bk)
    return backend, backend.create_session("sid-test-0001")


async def _feed_sentence(session, *, voice_ms=2000, silence_ms=800):
    messages = await _collect(session.feed_audio(_voice(voice_ms)))
    messages += await _collect(session.feed_audio(_silence(silence_ms)))
    return messages


# ─── VllmStreamSession.configure ───

def test_configure_warns_unsupported_params():
    _, sess = _make_session()
    warns = sess.configure({"audio_fs": 16000, "with_words": True, "diarize": True,
                            "with_punc": True, "speaker_threshold": 0.5})
    assert set(warns) == {"with_words", "diarize", "with_punc", "speaker_threshold"}


def test_configure_invalid_audio_fs_raises():
    _, sess = _make_session()
    with pytest.raises(ValueError):
        sess.configure({"audio_fs": 100})        # < 8000 下限


def test_configure_chunk_size_override_and_range():
    _, sess = _make_session()
    assert sess.configure({"chunk_size_sec": 1.5}) == []
    assert sess._chunk_size_sec == 1.5
    with pytest.raises(ValueError):
        sess.configure({"chunk_size_sec": 10})    # > 5.0 上限


def test_configure_accepts_speaker_options_when_enabled():
    backend, sess = _make_session(
        speaker=_FakeSpeakerEngine(),
        speaker_service=_FakeSpeakerService(),
    )
    try:
        warnings = sess.configure({
            "speaker_threshold": 0.6,
            "speaker_min_seg_ms": 800,
            "speaker_max": 5,
            "speaker_id_threshold": 0.55,
            "speaker_id_margin": 0.2,
            "identify_speakers": True,
        })
        assert warnings == []
        assert sess._spk_cluster is not None
        assert sess._spk_threshold == 0.6
        assert sess._spk_min_seg_ms == 800
        assert sess._spk_max == 5
        assert sess._identify is True
    finally:
        backend.shutdown()


@pytest.mark.parametrize("options", [
    {"speaker_threshold": 0.1},
    {"speaker_min_seg_ms": 10001},
    {"speaker_max": 0},
    {"speaker_id_threshold": 1.1},
    {"speaker_id_margin": -0.1},
    {"diarize": "yes"},
    {"identify_speakers": "yes"},
])
def test_configure_rejects_invalid_speaker_options(options):
    backend, sess = _make_session(speaker=_FakeSpeakerEngine())
    try:
        with pytest.raises(ValueError):
            sess.configure(options)
    finally:
        backend.shutdown()


def test_configure_diarize_false_disables_cluster_without_warning():
    backend, sess = _make_session(speaker=_FakeSpeakerEngine())
    try:
        assert sess.configure({"diarize": False}) == []
        assert sess._spk_cluster is None
        sess.configure({})
        assert sess._spk_cluster is not None
    finally:
        backend.shutdown()


def test_configure_identify_without_ready_service_warns():
    backend, sess = _make_session(speaker=_FakeSpeakerEngine())
    try:
        warnings = sess.configure({
            "identify_speakers": True,
            "speaker_id_threshold": 0.5,
        })
        assert set(warnings) == {"identify_speakers", "speaker_id_threshold"}
    finally:
        backend.shutdown()


# ─── VllmStreamSession.feed_audio / flush ───

def test_feed_audio_partial_then_final():
    eng = _MockEngine()
    backend, sess = _make_session(eng, end_silence_ms=800)
    sess.configure({"audio_fs": 16000})

    async def run():
        msgs = []
        for _ in range(3):                        # 语音 → 起句 + partial
            msgs += await _collect(sess.feed_audio(_voice()))
        for _ in range(4):                        # 800ms 静音 → 判停 final
            msgs += await _collect(sess.feed_audio(_silence()))
        return msgs

    msgs = asyncio.run(run())
    partials = [m for m in msgs if m["type"] == "partial"]
    finals = [m for m in msgs if m["type"] == "final"]

    assert len(partials) >= 3
    assert all(m["seg_id"] == 0 and m["text"] for m in partials)
    assert len(finals) == 1
    f = finals[0]
    assert f["seg_id"] == 0 and f["text"].endswith("。")
    assert f["start"] == 0 and f["end"] == 1400
    assert sess.state is None                     # 句尾已 reset
    assert eng.new_states == 1                    # 仅起了一句


def test_feed_audio_marks_realtime_priority_section():
    class Gate:
        def __init__(self):
            self.entries = 0

        def realtime_section(self):
            gate = self

            class _Ctx:
                def __enter__(self):
                    gate.entries += 1

                def __exit__(self, exc_type, exc, tb):
                    return False

            return _Ctx()

    gate = Gate()
    eng = _MockEngine()
    backend, sess = _make_session(eng, priority_gate=gate)
    sess.configure({"audio_fs": 16000})

    async def run():
        return await _collect(sess.feed_audio(_voice()))

    msgs = asyncio.run(run())
    assert msgs and msgs[0]["type"] == "partial"
    assert gate.entries == 1


def test_feed_audio_marks_realtime_priority_while_waiting_for_sem():
    class Gate:
        def __init__(self):
            self.entries = 0
            self.exits = 0

        def realtime_section(self):
            gate = self

            class _Ctx:
                def __enter__(self):
                    gate.entries += 1

                def __exit__(self, exc_type, exc, tb):
                    gate.exits += 1
                    return False

            return _Ctx()

    gate = Gate()
    eng = _MockEngine()
    sem = asyncio.Semaphore(1)
    backend = VllmStreamBackend(eng, priority_gate=gate)
    sess = VllmStreamSession(
        "sid-test-0001", eng, EnergyEndpointer(),
        backend._executor, sem, priority_gate=gate)
    sess.configure({"audio_fs": 16000})

    async def run():
        await sem.acquire()
        task = asyncio.create_task(_collect(sess.feed_audio(_voice())))
        try:
            await asyncio.sleep(0.01)
            assert gate.entries == 1
            assert gate.exits == 0
            sem.release()
            msgs = await task
            assert msgs and msgs[0]["type"] == "partial"
            assert gate.exits == 1
        finally:
            if not task.done():
                task.cancel()
            backend.shutdown()

    asyncio.run(run())


def test_flush_emits_final_for_open_segment():
    eng = _MockEngine()
    _, sess = _make_session(eng)
    sess.configure({"audio_fs": 16000})

    async def run():
        msgs = []
        for _ in range(2):                        # 起句但不静音收尾
            msgs += await _collect(sess.feed_audio(_voice()))
        msgs += await _collect(sess.flush())      # stop → 冲刷末句
        return msgs

    msgs = asyncio.run(run())
    finals = [m for m in msgs if m["type"] == "final"]
    assert len(finals) == 1
    assert finals[0]["text"].endswith("。")
    assert sess.state is None


def test_utterance_cut_keeps_sdk_state_and_starts_next_ui_segment():
    eng = _MockEngine()
    _, sess = _make_session(eng, max_utterance_sec=1, max_state_sec=300)
    sess.configure({"audio_fs": 16000})

    async def run():
        msgs = []
        for _ in range(6):                        # 1.2s 连续语音；1.0s 触发 UI 分段
            msgs += await _collect(sess.feed_audio(_voice()))
        return msgs

    msgs = asyncio.run(run())
    finals = [m for m in msgs if m["type"] == "final"]
    partials = [m for m in msgs if m["type"] == "partial"]

    assert len(finals) == 1
    assert finals[0]["seg_id"] == 0
    assert finals[0]["text"] == "字字字字字"
    assert partials[-1]["seg_id"] == 1
    assert partials[-1]["text"] == "字"
    assert sess.state is not None
    assert eng.new_states == 1                    # 20s/UI 分段不重建 SDK state
    assert eng.finishes == 0                      # 未到自然句尾/stop/state 上限，不 finish


def test_state_cut_finishes_and_restarts_sdk_state_before_more_audio():
    eng = _MockEngine()
    _, sess = _make_session(eng, max_utterance_sec=10, max_state_sec=1)
    sess.configure({"audio_fs": 16000})

    async def run():
        msgs = []
        for _ in range(6):                        # 1.2s 连续语音；1.0s 触发 SDK state 重置
            msgs += await _collect(sess.feed_audio(_voice()))
        return msgs

    msgs = asyncio.run(run())
    finals = [m for m in msgs if m["type"] == "final"]
    partials = [m for m in msgs if m["type"] == "partial"]

    assert len(finals) == 1
    assert finals[0]["text"] == "字字字字字。"
    assert partials[-1]["seg_id"] == 1
    assert partials[-1]["text"] == "字"
    assert sess.state is not None
    assert eng.new_states == 2
    assert eng.finishes == 1


def test_large_frame_is_split_before_state_budget_is_exceeded():
    eng = _MockEngine()
    _, sess = _make_session(eng, max_utterance_sec=10, max_state_sec=1)
    sess.configure({"audio_fs": 16000})

    async def run():
        return await _collect(sess.feed_audio(_voice(ms=1200)))

    asyncio.run(run())
    assert max(eng.feed_sizes) <= SR              # 不把超过 state 剩余预算的音频喂进旧 state


def test_flush_after_exact_state_cut_does_not_emit_empty_segment():
    eng = _MockEngine()
    _, sess = _make_session(eng, max_utterance_sec=10, max_state_sec=1)
    sess.configure({"audio_fs": 16000})

    async def run():
        msgs = []
        msgs += await _collect(sess.feed_audio(_voice(ms=1000)))
        msgs += await _collect(sess.flush())
        return msgs

    msgs = asyncio.run(run())
    finals = [m for m in msgs if m["type"] == "final"]
    assert len(finals) == 1
    assert finals[0]["text"]


def test_flush_after_utterance_cut_uses_finish_text():
    eng = _MockEngine()
    _, sess = _make_session(eng, max_utterance_sec=1, max_state_sec=300)
    sess.configure({"audio_fs": 16000})

    async def run():
        msgs = []
        cut_msgs = await _collect(sess.feed_audio(_voice(ms=1000)))
        assert [m for m in cut_msgs if m["type"] == "final"] == []
        msgs += cut_msgs
        msgs += await _collect(sess.flush())
        return msgs

    msgs = asyncio.run(run())
    finals = [m for m in msgs if m["type"] == "final"]

    assert len(finals) == 1
    assert finals[0]["seg_id"] == 0
    assert finals[0]["text"] == "字。"
    assert sess.state is None
    assert eng.new_states == 1
    assert eng.finishes == 1


def test_utterance_cut_then_trailing_silence_does_not_emit_empty_segment():
    eng = _SilenceAwareMockEngine()
    _, sess = _make_session(eng, max_utterance_sec=1, max_state_sec=300, end_silence_ms=800)
    sess.configure({"audio_fs": 16000})

    async def run():
        msgs = []
        cut_msgs = await _collect(sess.feed_audio(_voice(ms=1000)))
        assert [m for m in cut_msgs if m["type"] == "final"] == []
        msgs += cut_msgs
        for _ in range(4):
            msgs += await _collect(sess.feed_audio(_silence()))   # 仅尾静音，不应新建 UI 段
        return msgs

    msgs = asyncio.run(run())
    finals = [m for m in msgs if m["type"] == "final"]

    assert len(finals) == 1
    assert finals[0]["seg_id"] == 0
    assert finals[0]["text"] == "字。"
    assert sess.state is None
    assert eng.new_states == 1
    assert eng.finishes == 1


def test_feed_audio_silence_only_no_segment():
    eng = _MockEngine()
    _, sess = _make_session(eng)
    sess.configure({"audio_fs": 16000})

    async def run():
        msgs = []
        for _ in range(5):
            msgs += await _collect(sess.feed_audio(_silence()))
        return msgs

    msgs = asyncio.run(run())
    assert msgs == []                             # 全静音不起句、不解码
    assert eng.new_states == 0 and eng.feeds == 0


# ─── 实时说话人分离 / 声纹识别 ───

def test_speaker_labels_different_and_same_speakers():
    speaker = _FakeSpeakerEngine(vec_indices=[0, 1, 0])
    backend, sess = _make_session(
        _SilenceAwareMockEngine(), speaker=speaker, end_silence_ms=800)
    sess.configure({})

    async def run():
        finals = []
        for _ in range(3):
            finals += [
                msg for msg in await _feed_sentence(sess)
                if msg["type"] == "final"
            ]
        return finals

    try:
        finals = asyncio.run(run())
        assert [msg["speaker"] for msg in finals] == ["A", "B", "A"]
    finally:
        backend.shutdown()


def test_speaker_short_segment_remains_unlabeled():
    speaker = _FakeSpeakerEngine()
    backend, sess = _make_session(
        _SilenceAwareMockEngine(), speaker=speaker, end_silence_ms=800)
    sess.configure({})
    try:
        messages = asyncio.run(_feed_sentence(sess, voice_ms=1000))
        final = next(msg for msg in messages if msg["type"] == "final")
        assert "speaker" not in final
        assert sess._spk_cluster.centroids == []
    finally:
        backend.shutdown()


def test_speaker_failure_does_not_drop_text_final():
    backend, sess = _make_session(
        _SilenceAwareMockEngine(),
        speaker=_BoomSpeakerEngine(),
        end_silence_ms=800,
    )
    sess.configure({})
    try:
        messages = asyncio.run(_feed_sentence(sess))
        final = next(msg for msg in messages if msg["type"] == "final")
        assert final["text"]
        assert "speaker" not in final
    finally:
        backend.shutdown()


def test_stale_generation_drops_final_before_cluster_mutation():
    backend, sess = _make_session(
        _SilenceAwareMockEngine(), speaker=_FakeSpeakerEngine())
    sess.configure({})
    old_generation = sess._generation
    sess.close()

    try:
        result = asyncio.run(sess._annotate_final(
            {"type": "final", "seg_id": 0, "text": "旧结果"},
            (np.array([1.0, 0.0], dtype=np.float32), 2000),
            old_generation,
        ))
        assert result is None
    finally:
        backend.shutdown()


def test_diarize_false_skips_audio_buffer_and_embedding():
    speaker = _FakeSpeakerEngine()
    backend, sess = _make_session(
        _SilenceAwareMockEngine(), speaker=speaker, end_silence_ms=800)
    sess.configure({"diarize": False})
    try:
        messages = asyncio.run(_feed_sentence(sess))
        final = next(msg for msg in messages if msg["type"] == "final")
        assert "speaker" not in final
        assert speaker.calls == 0
        assert sess._segment_audio is None
    finally:
        backend.shutdown()


def test_speaker_audio_trims_endpoint_tail_silence():
    speaker = _FakeSpeakerEngine()
    backend, sess = _make_session(
        _SilenceAwareMockEngine(), speaker=speaker, end_silence_ms=800)
    sess.configure({})
    try:
        asyncio.run(_feed_sentence(sess, voice_ms=2000, silence_ms=800))
        assert speaker.audio_sizes == [2 * SR]
    finally:
        backend.shutdown()


def test_pending_ui_final_keeps_exact_audio_snapshot():
    speaker = _FakeSpeakerEngine()
    backend, sess = _make_session(
        _SilenceAwareMockEngine(),
        speaker=speaker,
        max_utterance_sec=1,
        max_state_sec=300,
        end_silence_ms=800,
    )
    sess.configure({"speaker_min_seg_ms": 0})

    async def run():
        messages = await _collect(sess.feed_audio(_voice(1000)))
        assert not [msg for msg in messages if msg["type"] == "final"]
        messages += await _collect(sess.feed_audio(_voice(200)))
        messages += await _collect(sess.flush())
        return [msg for msg in messages if msg["type"] == "final"]

    try:
        finals = asyncio.run(run())
        assert len(finals) == 2
        assert [msg["speaker"] for msg in finals] == ["A", "A"]
        assert speaker.audio_sizes == [SR, int(0.2 * SR)]
    finally:
        backend.shutdown()


def test_final_carries_speaker_name_and_uses_cluster_cache():
    speaker = _FakeSpeakerEngine()
    service = _FakeSpeakerService()
    backend, sess = _make_session(
        _SilenceAwareMockEngine(),
        speaker=speaker,
        speaker_service=service,
        end_silence_ms=800,
    )
    sess.configure({"identify_speakers": True})

    async def run():
        finals = []
        for _ in range(3):
            finals += [
                msg for msg in await _feed_sentence(sess)
                if msg["type"] == "final"
            ]
        return finals

    try:
        finals = asyncio.run(run())
        assert [msg["speaker_name"] for msg in finals] == ["张三", "张三", "张三"]
        assert service.calls == 2              # count=1、2 查询；count=3 命中缓存
    finally:
        backend.shutdown()


def test_unknown_speaker_can_match_after_centroid_stabilizes():
    service = _FakeSpeakerService(names=(None, "张三"))
    backend, sess = _make_session(
        _SilenceAwareMockEngine(),
        speaker=_FakeSpeakerEngine(),
        speaker_service=service,
        end_silence_ms=800,
    )
    sess.configure({"identify_speakers": True})

    async def run():
        first = await _feed_sentence(sess)
        second = await _feed_sentence(sess)
        return (
            next(msg for msg in first if msg["type"] == "final"),
            next(msg for msg in second if msg["type"] == "final"),
        )

    try:
        first, second = asyncio.run(run())
        assert first["speaker"] == "A" and "speaker_name" not in first
        assert second["speaker_name"] == "张三"
    finally:
        backend.shutdown()


def test_speaker_name_cache_refreshes_after_store_change():
    service = _FakeSpeakerService(names=(None, None, "张三"))
    backend, sess = _make_session(
        _SilenceAwareMockEngine(),
        speaker=_FakeSpeakerEngine(),
        speaker_service=service,
        end_silence_ms=800,
    )
    sess.configure({"identify_speakers": True})

    async def run():
        await _feed_sentence(sess)             # count=1，查询
        await _feed_sentence(sess)             # count=2，翻倍重查
        service.store.cache_version += 1
        return await _feed_sentence(sess)      # count=3，本应缓存；版本变化触发重查

    try:
        messages = asyncio.run(run())
        final = next(msg for msg in messages if msg["type"] == "final")
        assert service.calls == 3
        assert final["speaker_name"] == "张三"
    finally:
        backend.shutdown()


def test_speaker_service_failure_keeps_anonymous_label():
    class BrokenStore:
        @property
        def cache_version(self):
            raise RuntimeError("store closed")

    class BrokenService:
        store = BrokenStore()

    backend, sess = _make_session(
        _SilenceAwareMockEngine(),
        speaker=_FakeSpeakerEngine(),
        speaker_service=BrokenService(),
        end_silence_ms=800,
    )
    sess.configure({"identify_speakers": True})
    try:
        messages = asyncio.run(_feed_sentence(sess))
        final = next(msg for msg in messages if msg["type"] == "final")
        assert final["speaker"] == "A"
        assert "speaker_name" not in final
    finally:
        backend.shutdown()


def test_release_invalidates_inflight_speaker_result():
    started = threading.Event()
    proceed = threading.Event()

    class BlockingSpeaker(_FakeSpeakerEngine):
        def embed_segment(self, audio):
            started.set()
            assert proceed.wait(timeout=2)
            return super().embed_segment(audio)

    backend, sess = _make_session(
        _SilenceAwareMockEngine(),
        speaker=BlockingSpeaker(),
        end_silence_ms=800,
    )
    sess.configure({})

    async def run():
        await _collect(sess.feed_audio(_voice(2000)))
        task = asyncio.create_task(_collect(sess.feed_audio(_silence(800))))
        assert await asyncio.to_thread(started.wait, 2)
        backend.release(sess)
        proceed.set()
        return await task

    try:
        assert asyncio.run(run()) == []
        assert sess._spk_cluster is None
        assert sess._pending_ui_final is None
    finally:
        proceed.set()
        backend.shutdown()


# ─── VllmStreamBackend ───

def test_backend_capabilities():
    backend, _ = _make_session()
    assert backend.mode == "vllm" and backend.backend == "vllm-native"
    assert backend.capabilities["partial_results"] is True
    assert backend.capabilities["word_timestamps"] is False
    assert backend.capabilities["speaker_labels"] is False


def test_backend_speaker_capabilities_and_release_cleanup():
    speaker = _FakeSpeakerEngine()
    service = _FakeSpeakerService()
    backend = VllmStreamBackend(
        _MockEngine(), speaker=speaker, speaker_service=service)
    assert backend.capabilities["speaker_labels"] is True
    assert backend.capabilities["speaker_identification"] is True
    assert backend.capabilities["speaker_tunable"] is True

    session = backend.create_session("speaker-session")
    session.configure({})
    session._begin_ui_segment(0)
    assert session._spk_cluster is not None
    assert session._segment_audio is not None
    backend.release(session)
    assert session._spk_cluster is None
    assert session._segment_audio is None
    backend.shutdown()


def test_backend_acquire_release_limits():
    backend = VllmStreamBackend(_MockEngine(), max_sessions=2)

    async def run():
        a = await backend.acquire()
        b = await backend.acquire()
        c = await backend.acquire()               # 超额
        return a, b, c

    a, b, c = asyncio.run(run())
    assert (a, b, c) == (True, True, False)
    backend.release(backend.create_session("x"))  # 释放一个名额
    assert asyncio.run(backend.acquire()) is True
    backend.shutdown()
