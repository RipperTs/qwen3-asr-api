"""vLLM 模式离线转写处理器（Phase 1）。

供 TaskManager 的 process_fn 调用：上传音频经 ffmpeg 转 16k → 一次性 vLLM 批量
transcribe → 按词间隙分段 → 组装成与 standard /v2/asr 同形的 result（segments /
full_text / words / warnings）。

设计要点（见 docs/plan/features/20260612_vllm_offline_asr/）：
- 不依赖 funasr：分段用词级时间戳的「词间隙」（对齐器开时）/ 整文兜底；标点用模型原生。
- 顶层不 import vllm/qwen_asr（仅经传入的 engine 间接调用），依赖中性，standard venv 可单测。
- transcribe 为单次阻塞调用、不可中断：仅在开始前检查取消以免空耗。
"""
import logging
import os
import re

from app import config as cfg
from app.pipeline.audio_preprocessor import convert_to_wav, get_audio_duration
from app.utils.result_parser import extract_text, extract_words

logger = logging.getLogger(__name__)


def run_vllm_offline(engine, task, *, progress_callback=None, cancelled=None) -> dict:
    """执行一次离线转写，返回与 standard ASRPipeline.run 同形的 result dict。"""
    task_id = task["task_id"]
    file_path = task["file_path"]
    language = task.get("language")
    opts = task.get("options") or {}
    identify_speakers = task.get("identify_speakers", False)

    with_words = opts.get("with_words", True)
    max_segment = opts.get("max_segment")        # 秒；None → cfg.MAX_SEGMENT_DURATION

    warnings = _collect_warnings(engine, opts, identify_speakers)

    wav_path = None
    try:
        if progress_callback:
            progress_callback(0.05)
        os.makedirs(cfg.UPLOADS_DIR, exist_ok=True)
        wav_path = os.path.join(cfg.UPLOADS_DIR, f"{task_id}.wav")
        convert_to_wav(file_path, wav_path)

        duration = get_audio_duration(wav_path)
        if duration < cfg.MIN_AUDIO_DURATION:
            raise ValueError(f"音频过短（{duration:.1f}s），最短要求 {cfg.MIN_AUDIO_DURATION}s")
        if duration > cfg.MAX_AUDIO_DURATION:
            raise ValueError(f"音频过长（{duration:.0f}s），最大支持 {cfg.MAX_AUDIO_DURATION}s")

        # transcribe 单次阻塞、不可中断：仅开始前检查取消（worker 据 cancel_event 定终态）
        if cancelled and cancelled():
            return _result([], "", language, engine, warnings)

        if progress_callback:
            progress_callback(0.1)
        want_words = with_words and engine.align_enabled
        results = engine.transcribe(wav_path, language=language, with_words=want_words)

        if progress_callback:
            progress_callback(0.9)
        full_text = extract_text(results).strip()
        words = extract_words(results, 0.0) if want_words else None
        segments = _segment(full_text, words, duration, max_segment)

        if progress_callback:
            progress_callback(1.0)
        return _result(segments, full_text, language, engine, warnings)
    finally:
        _cleanup(file_path, wav_path)


def _collect_warnings(engine, opts: dict, identify_speakers: bool) -> list:
    """请求了但本模式不支持/无法生效的项 → 软提示（随 result 返回，不报错）。"""
    w = []
    if opts.get("with_punc") is False:
        w.append("with_punc")            # vLLM 标点由模型原生提供，无法单独关闭
    if opts.get("with_words") is True and not engine.align_enabled:
        w.append("with_words")           # 对齐器未加载
    if opts.get("diarize") is True:
        w.append("diarize")              # Phase 1 无说话人分离
    if identify_speakers:
        w.append("identify_speakers")
    if opts.get("speaker_id_threshold") is not None or opts.get("speaker_id_margin") is not None:
        w.append("speaker_id_threshold/margin")
    return w


_SENTENCE_PUNCT = r"[。！？；!?;]"      # 句末标点（中英）→ 主切点
_CLAUSE_PUNCT = r"[，,、]"              # 子句标点 → 超长句二次切
_DURATION_CLAMP_FACTOR = 2.0           # 段时长超 max_seg×此值视为对齐器跨块损坏 → 钳制


def _segment(full_text: str, words, duration: float, max_segment) -> list:
    """标点优先分段：用 full_text 原生句末标点切句（超长句按逗号细切），词时间戳仅用于
    定位 start/end（min/max + 钳制，免疫对齐器伪间隙/时间戳回退）。

    段文本取自 full_text 切片（非词拼接）→ 保留模型原生标点、concat(segments)==full_text。
    无词级时间戳则整文单段；无句末标点（罕见短句/非中文）退化为词间隙分段。
    """
    if not full_text:
        return []
    if not words:
        return [{"start": 0.0, "end": round(float(duration), 3), "text": full_text}]

    positions = _word_positions(full_text, words)        # 每词在 full_text 的起始下标（同序）
    max_seg = float(max_segment) if max_segment else float(cfg.MAX_SEGMENT_DURATION)

    sentence_cuts = [m.end() for m in re.finditer(_SENTENCE_PUNCT, full_text)]
    if not sentence_cuts:
        return _segment_by_word_gap(full_text, words, positions, max_seg)

    # 句级切片，超 max_seg 的句子在逗号处细切
    final = []
    for c0, c1 in _spans(0, len(full_text), sentence_cuts):
        if _span_seconds(c0, c1, positions, words) > max_seg:
            sub = [c0 + m.end() for m in re.finditer(_CLAUSE_PUNCT, full_text[c0:c1])]
            final.extend(_spans(c0, c1, sub))
        else:
            final.append((c0, c1))

    segments = []
    for c0, c1 in final:
        text = full_text[c0:c1]
        sw = [w for i, w in enumerate(words) if c0 <= positions[i] < c1]
        if not sw:                                       # 纯标点/空片段 → 文本并入前段
            if segments:
                segments[-1]["text"] += text
            continue
        start = min(w["start"] for w in sw)
        end = max(w["end"] for w in sw)                  # min/max 保证 end>=start
        if end - start > max_seg * _DURATION_CLAMP_FACTOR:
            end = start + max_seg                        # 跨块时间戳损坏 → 钳制为近似时长
        segments.append({"start": round(start, 3), "end": round(end, 3),
                         "text": text, "words": list(sw)})
    return segments or [{"start": 0.0, "end": round(float(duration), 3), "text": full_text}]


def _spans(lo: int, hi: int, cut_ends: list) -> list:
    """按升序切点（段结束位）把 [lo, hi) 切成平铺片段 [(s,e), ...]。"""
    spans, s = [], lo
    for c in cut_ends:
        if s < c <= hi:
            spans.append((s, c))
            s = c
    if s < hi:
        spans.append((s, hi))
    return spans


def _span_seconds(c0: int, c1: int, positions: list, words: list) -> float:
    sw = [w for i, w in enumerate(words) if c0 <= positions[i] < c1]
    return (max(w["end"] for w in sw) - min(w["start"] for w in sw)) if sw else 0.0


def _segment_by_word_gap(full_text: str, words: list, positions: list, max_seg: float) -> list:
    """退化路径（full_text 无句末标点，罕见）：按词间隙/回退/段长分段，段文本取 full_text 切片。"""
    gap = cfg.VLLM_SEGMENT_GAP_MS / 1000.0
    groups, cur = [], []
    for w in words:
        if cur:
            prev = cur[-1]
            if (w["start"] - prev["end"]) > gap or w["start"] < prev["end"] \
                    or (w["end"] - cur[0]["start"]) > max_seg:
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    first_idx, gi = [], 0
    for g in groups:
        first_idx.append(gi)
        gi += len(g)
    segments = []
    for k, g in enumerate(groups):
        t0 = 0 if k == 0 else positions[first_idx[k]]
        t1 = positions[first_idx[k + 1]] if k + 1 < len(groups) else len(full_text)
        end = max(g[0]["start"], g[-1]["end"])
        segments.append({"start": round(g[0]["start"], 3), "end": round(end, 3),
                         "text": full_text[t0:t1], "words": list(g)})
    return segments


def _word_positions(full_text: str, words: list) -> list:
    """每词在 full_text 中的起始下标（贪心游标推进）；匹配不到时以游标兜底，不抛错。"""
    positions, cursor = [], 0
    for w in words:
        t = w.get("text", "")
        idx = full_text.find(t, cursor) if t else -1
        if idx < 0:
            idx = cursor                                 # 对齐文本与模型文本不符 → 兜底
        positions.append(idx)
        cursor = idx + len(t)
    return positions


def _result(segments, full_text, language, engine, warnings) -> dict:
    result = {
        "segments": segments,
        "full_text": full_text,
        "language": language,
        "align_enabled": engine.align_enabled,
        # vLLM 标点由模型原生提供（恒有，故 True）；非 CT-Transformer 且不可单独关闭，
        # with_punc=false 时进 warnings 表达"无法关闭"。与 standard 的 bool 类型对齐。
        "punc_enabled": True,
    }
    if warnings:
        result["warnings"] = warnings
    return result


def _cleanup(*paths):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError as e:
                logger.warning(f"临时文件清理失败 {p}: {e}")
