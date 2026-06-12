"""vLLM 离线处理器单元测试（mock engine，不依赖 vLLM/GPU/ffmpeg）。

覆盖：词间隙分段 / 整文兜底 / max_segment 二次切 / warnings 生成 / run_vllm_offline
端到端（result schema 同 standard、progress、cancelled 早退、with_words 透传）。
dependency-neutral：standard venv 即可运行。
"""
from types import SimpleNamespace

import pytest

from app.runtime import vllm_offline as vo
from app import config as cfg


class _Engine:
    """最小 mock 引擎：align_enabled 属性 + transcribe 记录调用并回放预置结果。"""

    def __init__(self, align=True, result=None):
        self._align = align
        self._result = result or []
        self.transcribe_calls = []

    @property
    def align_enabled(self):
        return self._align

    def transcribe(self, audio_path, language=None, with_words=False):
        self.transcribe_calls.append((audio_path, language, with_words))
        return self._result


def _trans(text, items=None):
    """构造 ASRTranscription 形态：.text + .time_stamps.items[].{text,start_time,end_time}。"""
    ts = None
    if items:
        ts = SimpleNamespace(items=[
            SimpleNamespace(text=t, start_time=s, end_time=e) for t, s, e in items])
    return SimpleNamespace(text=text, time_stamps=ts, language="Chinese")


# ── _segment ──────────────────────────────────────────────
def test_segment_by_sentence():
    """标点优先（主路径）：按句末标点 。！？ 切句，段文本含标点、精确平铺。"""
    full = "哎呦。王处？辛苦！"
    words = [
        {"text": "哎", "start": 0.0, "end": 0.2}, {"text": "呦", "start": 0.2, "end": 0.4},
        {"text": "王", "start": 1.0, "end": 1.2}, {"text": "处", "start": 1.2, "end": 1.4},
        {"text": "辛", "start": 2.0, "end": 2.2}, {"text": "苦", "start": 2.2, "end": 2.4},
    ]
    segs = vo._segment(full, words, 3.0, None)
    assert [s["text"] for s in segs] == ["哎呦。", "王处？", "辛苦！"]
    assert "".join(s["text"] for s in segs) == full
    assert segs[0]["start"] == 0.0 and segs[0]["end"] == 0.4 and len(segs[0]["words"]) == 2


def test_segment_word_gap_fallback():
    """无句末标点（罕见）→ 退化为词间隙分段（间隙 0.8s > 0.5 断段）。"""
    words = [
        {"text": "你", "start": 0.0, "end": 0.2},
        {"text": "好", "start": 0.25, "end": 0.4},   # 间隙 0.05s → 同段
        {"text": "世", "start": 1.2, "end": 1.4},     # 间隙 0.8s > 0.5 → 新段
        {"text": "界", "start": 1.45, "end": 1.6},
    ]
    segs = vo._segment("你好世界", words, 2.0, None)
    assert [s["text"] for s in segs] == ["你好", "世界"]
    assert segs[0]["start"] == 0.0 and segs[0]["end"] == 0.4 and len(segs[0]["words"]) == 2


def test_segment_whole_text_fallback():
    assert vo._segment("整段", None, 3.0, None) == [{"start": 0.0, "end": 3.0, "text": "整段"}]


def test_segment_empty_text():
    assert vo._segment("", None, 1.0, None) == []


def test_segment_max_segment_cap(monkeypatch):
    monkeypatch.setattr(cfg, "VLLM_SEGMENT_GAP_MS", 2000)   # 间隙阈值很大 → 不靠间隙断
    words = [{"text": str(i), "start": i * 0.3, "end": i * 0.3 + 0.2} for i in range(10)]  # 跨度 ~2.9s
    segs = vo._segment("0123456789", words, 3.0, 1.0)        # max_segment=1s 强制二次切
    assert len(segs) >= 2
    assert all((s["end"] - s["start"]) <= 1.0 + 0.3 for s in segs)


def test_segment_preserves_punctuation():
    """段文本取自 full_text 切片 → 保留模型原生标点；逗号不单独成段（句级），精确平铺。"""
    full = "你好，世界。再见！"
    words = [
        {"text": "你", "start": 0.0, "end": 0.2},
        {"text": "好", "start": 0.25, "end": 0.4},
        {"text": "世", "start": 1.2, "end": 1.4},
        {"text": "界", "start": 1.45, "end": 1.6},
        {"text": "再", "start": 2.5, "end": 2.7},
        {"text": "见", "start": 2.75, "end": 2.9},
    ]
    segs = vo._segment(full, words, 3.0, None)
    assert "".join(s["text"] for s in segs) == full          # 精确平铺
    assert [s["text"] for s in segs] == ["你好，世界。", "再见！"]   # 句级：逗号不断


def test_segment_long_sentence_comma_subsplit():
    """超 max_segment 的长句在逗号处二次切。"""
    full = "甲，乙，丙。"
    words = [
        {"text": "甲", "start": 0.0, "end": 0.2},
        {"text": "乙", "start": 3.0, "end": 3.2},
        {"text": "丙", "start": 6.0, "end": 6.2},     # 整句跨度 6.2s > max_segment=5 → 按逗号切
    ]
    segs = vo._segment(full, words, 7.0, 5)
    assert "".join(s["text"] for s in segs) == full
    assert [s["text"] for s in segs] == ["甲，", "乙，", "丙。"]


def test_segment_clamps_corrupt_duration():
    """对齐器时间戳回退/过摊致段跨度异常 → end 钳制、无负时长、文本仍完整。"""
    full = "前后。"                                    # 单句，含句末标点
    words = [
        {"text": "前", "start": 100.0, "end": 100.2},
        {"text": "后", "start": 60.0, "end": 60.2},    # 回退：min/max 跨度 40.2s
    ]
    segs = vo._segment(full, words, 200.0, 5)
    assert len(segs) == 1
    assert segs[0]["text"] == "前后。"
    assert segs[0]["end"] >= segs[0]["start"]            # 无负时长
    assert segs[0]["end"] - segs[0]["start"] <= 5 + 0.01  # 钳制到 max_segment


# ── _collect_warnings ─────────────────────────────────────
def test_warnings_all():
    w = vo._collect_warnings(
        _Engine(align=False),
        {"with_punc": False, "with_words": True, "diarize": True, "speaker_id_threshold": 0.5},
        identify_speakers=True)
    assert set(w) == {"with_punc", "with_words", "diarize",
                      "identify_speakers", "speaker_id_threshold/margin"}


def test_warnings_clean_when_align_on():
    # 对齐器开 + 仅请求 words → 无 warning
    assert vo._collect_warnings(_Engine(align=True), {"with_words": True}, False) == []


# ── run_vllm_offline 端到端 ────────────────────────────────
@pytest.fixture
def patched(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "UPLOADS_DIR", str(tmp_path))
    monkeypatch.setattr(vo, "convert_to_wav", lambda i, o: open(o, "wb").close())
    monkeypatch.setattr(vo, "get_audio_duration", lambda p: 5.0)
    return monkeypatch


def test_run_with_words(patched):
    eng = _Engine(align=True, result=[_trans(
        "你好。世界。", [("你", 0.0, 0.2), ("好", 0.25, 0.4), ("世", 1.2, 1.4), ("界", 1.45, 1.6)])])
    prog = []
    task = {"task_id": "t1", "file_path": "/x.wav", "language": "zh",
            "options": {"with_words": True}}
    r = vo.run_vllm_offline(eng, task, progress_callback=prog.append)

    assert r["full_text"] == "你好。世界。"
    assert r["align_enabled"] is True and r["punc_enabled"] is True
    assert r["language"] == "zh"
    assert len(r["segments"]) == 2 and r["segments"][0]["words"]    # 按句切：你好。| 世界。
    assert "warnings" not in r
    assert prog[-1] == 1.0
    assert eng.transcribe_calls[0][2] is True          # with_words 透传


def test_run_no_align_fallback(patched):
    eng = _Engine(align=False, result=[_trans("整段文本。")])
    task = {"task_id": "t2", "file_path": "/x.wav", "options": {"with_words": True}}
    r = vo.run_vllm_offline(eng, task)

    assert r["align_enabled"] is False
    assert len(r["segments"]) == 1 and "words" not in r["segments"][0]
    assert r["segments"][0]["text"] == "整段文本。"
    assert "with_words" in r["warnings"]               # 请求 words 但无对齐器
    assert eng.transcribe_calls[0][2] is False         # align off → 不透传 with_words


def test_run_cancelled_before_transcribe(patched):
    eng = _Engine(align=True, result=[_trans("不应产生")])
    task = {"task_id": "t3", "file_path": "/x.wav", "options": {}}
    r = vo.run_vllm_offline(eng, task, cancelled=lambda: True)

    assert r["segments"] == [] and r["full_text"] == ""
    assert eng.transcribe_calls == []                  # 取消 → 未触发推理
