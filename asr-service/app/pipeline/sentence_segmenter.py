"""句子级分句（accurate sentence segmentation）。

evolution.md §二.4 的落地：把"处理用的 ASR 切块（受 MAX_SEGMENT_DURATION 约束的音频块）"
重组为"真正的句子"。切句依据组合多种信号，而不是只按时长硬切：

  - 标点：句末标点（。！？!?;； 及句末英文句点 .）为强切；子句标点（，、,）为弱切
  - 停顿：词/块间静音 >= long_pause 为强切，>= short_pause 为弱切
  - 说话人切换：强切
  - 最大句长：仅当显式给定 max_segment 时，作为输出上限触发弱切→硬切兜底
  - 保护：小数（3.14）、点开头 token（.env）、单字母缩写（e.g.）不被英文句点误切

关键设计——处理切块时长与句子边界解耦：
落在"处理切块边界"上的句末标点，只有在伴随停顿或说话人切换时才算真句末；否则视为模型
按块产生的伪标点（软边界，不切），避免把固定的处理切块边界（如 5s）变成句子边界。

输入 chunks（按时间顺序），每个为 dict：
    {"start": float秒, "end": float秒, "text": str,
     "words": [{"text","start","end"}, ...] | None,   # 可选，词级时间戳
     "speaker": str | None}                            # 可选，块级说话人

输出句子级 segments（同形），其中 start/end 为绝对秒，words/speaker 视有无透传。
"""
from app import config as cfg

_SENTENCE_PUNCT = "。！？!?;；"   # 句末标点（中英）
_CLAUSE_PUNCT = "，,、"          # 子句标点（超长句弱切点）
_FAIL_MARK = "[识别失败]"


def segment_sentences(chunks, *, max_segment=None,
                      long_pause_ms=None, short_pause_ms=None, dedupe=True):
    """把 ASR 处理块重组为句子级 segments。max_segment 为 None/0 时不按时长切。

    dedupe=True 时先去除"处理块被拦腰切断"导致的边界重复识别（仅作用于紧邻块边界，
    见 dedupe_contiguous_boundaries）。
    """
    chunks = [c for c in chunks if (c.get("text") or "").strip()]
    if not chunks:
        return []
    if dedupe:
        chunks = dedupe_contiguous_boundaries(chunks)
        if not chunks:
            return []
    long_pause = (cfg.SENTENCE_LONG_PAUSE_MS if long_pause_ms is None else long_pause_ms) / 1000.0
    short_pause = (cfg.SENTENCE_SHORT_PAUSE_MS if short_pause_ms is None else short_pause_ms) / 1000.0

    sentences = []
    buf = None   # 当前累积句：{text, words(list|None), start, end, speaker}

    def flush():
        nonlocal buf
        if buf is not None:
            sentences.append(buf)
            buf = None

    def append_piece(piece, hard_after):
        nonlocal buf
        if buf is None:
            buf = {
                "text": piece["text"],
                "words": list(piece["words"]) if piece["words"] else None,
                "start": piece["start"],
                "end": piece["end"],
                "speaker": piece.get("speaker"),
            }
        else:
            buf["text"] += piece["text"]
            if piece["words"]:
                if buf["words"] is None:
                    buf["words"] = []
                buf["words"].extend(piece["words"])
            buf["end"] = max(buf["end"], piece["end"])
        if hard_after:
            flush()

    prev = None
    for ci, chunk in enumerate(chunks):
        text = chunk["text"]
        speaker = chunk.get("speaker")

        # 失败标记块：独立成句，两侧强切，不并入相邻句
        if text.strip() == _FAIL_MARK:
            flush()
            sentences.append({"text": text, "words": None,
                              "start": float(chunk["start"]), "end": float(chunk["end"]),
                              "speaker": speaker})
            prev = chunk
            continue

        # 进入本块前：跨块长停顿 / 说话人切换 → 先把已累积句强切
        if prev is not None and buf is not None:
            gap = float(chunk["start"]) - float(prev["end"])
            if gap >= long_pause or _speaker_changed(prev.get("speaker"), speaker):
                flush()

        pieces = _split_chunk_pieces(chunk, long_pause)
        nxt = chunks[ci + 1] if ci + 1 < len(chunks) else None
        for k, piece in enumerate(pieces):
            if k < len(pieces) - 1:
                # 块内片段：以内部句末标点 / 长停顿结尾 → 强切
                append_piece(piece, hard_after=True)
            else:
                append_piece(piece, hard_after=_chunk_end_is_hard(
                    piece, chunk, nxt, speaker, long_pause, short_pause))
        prev = chunk
    flush()

    if max_segment:
        sentences = _apply_max_segment(sentences, float(max_segment), short_pause)

    out = []
    for s in sentences:
        seg = {"start": round(float(s["start"]), 3),
               "end": round(float(max(s["end"], s["start"])), 3),
               "text": s["text"]}
        if s.get("words"):
            seg["words"] = s["words"]
        if s.get("speaker") is not None:
            seg["speaker"] = s["speaker"]
        out.append(seg)
    return out


# ─── 边界重复去重（处理块被拦腰切断的产物）────────────────────────────

def dedupe_contiguous_boundaries(chunks, *, gap_eps=0.05, min_overlap=2, max_overlap=20):
    """去除"长语音被时长切块拦腰切断"造成的边界重复识别。

    长于 MAX_SEGMENT_DURATION 的连续语音会被按时长切成多个"紧邻块"（块间间隙≈0）；
    切点落在词中时，边界词常被两侧各识别一次（如 "…面前。" + "面前，…"）。
    本函数仅在紧邻块（gap<=gap_eps）边界上，把前一块尾部与后一块头部重复的"内容字串"
    （>=min_overlap 个 isalnum 字符，精确匹配）从前一块尾部连同其后随标点一并删除。

    只作用于紧邻（时长切块）边界——真正连续重复的口语（"对对对"）通常在同一块内，或跨越
    VAD 静音间隙（gap>0），不会被误删。
    """
    out = [dict(c) for c in chunks]
    for i in range(len(out) - 1):
        a, b = out[i], out[i + 1]
        if not (a.get("text") and b.get("text")):
            continue
        if float(b["start"]) - float(a["end"]) > gap_eps:
            continue                                   # 仅紧邻（时长切块）边界
        overlap = _boundary_overlap(a["text"], b["text"], min_overlap, max_overlap)
        if overlap:
            _trim_tail_content(a, overlap)             # 删前块尾重复 + 其后随标点
    return [c for c in out if (c.get("text") or "").strip()]


def _boundary_overlap(a_text, b_text, min_overlap, max_overlap):
    """前块尾内容字串 == 后块头内容字串 的最长长度（内容字符数）；不足 min_overlap 返回 0。"""
    a_core = [ch for ch in a_text if ch.isalnum()]
    b_core = [ch for ch in b_text if ch.isalnum()]
    hi = min(len(a_core), len(b_core), max_overlap)
    for L in range(hi, min_overlap - 1, -1):
        if a_core[len(a_core) - L:] == b_core[:L]:
            return L
    return 0


def _trim_tail_content(chunk, n_content):
    """从块尾删除最后 n_content 个内容字符及其后随标点；词级时间戳同步裁剪。"""
    text = chunk["text"]
    idx = [k for k, ch in enumerate(text) if ch.isalnum()]
    keep_upto = 0 if n_content >= len(idx) else idx[len(idx) - n_content]
    chunk["text"] = text[:keep_upto].rstrip()
    words = chunk.get("words")
    if words:
        kept, removed = list(words), 0
        while kept and removed < n_content:
            w = kept.pop()
            removed += sum(1 for ch in w.get("text", "") if ch.isalnum()) or 1
        chunk["words"] = kept or None
        if kept:
            chunk["end"] = max(w["end"] for w in kept)


def _chunk_end_is_hard(piece, chunk, nxt, speaker, long_pause, short_pause):
    """块末片段是否强切：最后一块→是；否则需句末标点+短停顿/说话人切换，或长停顿/说话人切换。"""
    if nxt is None:
        return True
    gap = float(nxt["start"]) - float(chunk["end"])
    spk_change = _speaker_changed(speaker, nxt.get("speaker"))
    if _ends_with_sentence_punct(piece["text"]) and (gap >= short_pause or spk_change):
        return True              # 块末标点 + 真实停顿/换人 → 真句末
    return gap >= long_pause or spk_change   # 无标点也可被长停顿/换人切开


def _speaker_changed(a, b) -> bool:
    return a is not None and b is not None and a != b


# ─── 块内切片（内部强切点）─────────────────────────────────────────────

def _split_chunk_pieces(chunk, long_pause):
    """把一个块的文本切成"内部强切片段"：内部句末标点 / 长词间隙之后切开。

    块末标点不在此切（由 _chunk_end_is_hard 决定），从而实现"处理块边界 ≠ 句子边界"。
    """
    text = chunk["text"]
    words = chunk.get("words") or None
    speaker = chunk.get("speaker")
    cs, ce = float(chunk["start"]), float(chunk["end"])
    n = len(text)

    positions = _word_positions(text, words) if words else None

    cuts = set()
    for i in range(n):
        if _is_sentence_end_at(text, i):
            cuts.add(i + 1)                          # 在标点之后切
    if words and positions:
        for wi in range(len(words) - 1):
            if (words[wi + 1]["start"] - words[wi]["end"]) >= long_pause:
                cuts.add(positions[wi + 1])          # 长词间隙：在后一词起始处切
    cuts = sorted(c for c in cuts if 0 < c < n)      # 排除块末切点（块末单独处理）

    spans = _spans(0, n, cuts)
    pieces = _pieces(text, words, positions, cs, ce, spans, speaker)
    return pieces or [{"text": text, "words": words, "start": cs, "end": ce, "speaker": speaker}]


# ─── max_segment 上限（仅显式给定时）──────────────────────────────────

def _apply_max_segment(sentences, max_seg, short_pause):
    """超过 max_seg 的句子：先按子句标点细切，仍超长且无标点的片段按时间硬切。"""
    out = []
    for s in sentences:
        if (s["end"] - s["start"]) <= max_seg:
            out.append(s)
            continue
        out.extend(_subsplit(s, max_seg))
    return out


def _subsplit(s, max_seg):
    text = s["text"]
    words = s.get("words") or None
    speaker = s.get("speaker")
    cs, ce = float(s["start"]), float(s["end"])
    n = len(text)
    positions = _word_positions(text, words) if words else None

    cuts = sorted(i + 1 for i, ch in enumerate(text) if ch in _CLAUSE_PUNCT and 0 < i + 1 < n)
    spans = _spans(0, n, cuts)
    raw = _pieces(text, words, positions, cs, ce, spans, speaker)

    final = []
    for p in raw:
        if (p["end"] - p["start"]) > max_seg * 1.5:
            final.extend(_time_slice(p, max_seg))
        else:
            final.append(p)
    return final


def _time_slice(p, max_seg):
    """无标点超长片段：按等时长切若干段，文本按字符比例分摊，词按归属落段。"""
    dur = p["end"] - p["start"]
    k = max(1, int(-(-dur // max_seg)))   # ceil(dur / max_seg)
    if k <= 1:
        return [p]
    text = p["text"]
    n = len(text)
    words = p.get("words") or None
    positions = _word_positions(text, words) if words else None
    out = []
    for j in range(k):
        c0 = round(n * j / k)
        c1 = n if j == k - 1 else round(n * (j + 1) / k)
        st = p["start"] + dur * j / k
        en = p["start"] + dur * (j + 1) / k
        pw = [w for wi, w in enumerate(words) if c0 <= positions[wi] < c1] if words else None
        out.append({"text": text[c0:c1], "words": pw or None,
                    "start": st, "end": en, "speaker": p.get("speaker")})
    return out


# ─── 文本/标点工具 ────────────────────────────────────────────────────

def _pieces(text, words, positions, cs, ce, spans, speaker):
    """按字符区间切片，词级时间戳优先定位 start/end，无词时按字符比例估时。"""
    n = len(text)
    out = []
    for c0, c1 in spans:
        pw = [w for wi, w in enumerate(words) if c0 <= positions[wi] < c1] if words else None
        if pw:
            start = min(w["start"] for w in pw)
            end = max(w["end"] for w in pw)
        else:
            start = cs + (ce - cs) * (c0 / n) if n else cs
            end = cs + (ce - cs) * (c1 / n) if n else ce
            pw = None
        out.append({"text": text[c0:c1], "words": pw, "start": start, "end": end,
                    "speaker": speaker})
    return out


def _spans(lo, hi, cut_ends):
    """按升序切点（段结束位）把 [lo, hi) 切成平铺片段 [(s,e), ...]。"""
    spans, s = [], lo
    for c in cut_ends:
        if s < c <= hi:
            spans.append((s, c))
            s = c
    if s < hi:
        spans.append((s, hi))
    return spans


def _word_positions(full_text, words):
    """每词在 full_text 中的起始下标（贪心游标推进）；匹配不到以游标兜底，不抛错。"""
    positions, cursor = [], 0
    for w in words:
        t = w.get("text", "")
        idx = full_text.find(t, cursor) if t else -1
        if idx < 0:
            idx = cursor
        positions.append(idx)
        cursor = idx + len(t)
    return positions


def _ends_with_sentence_punct(text):
    t = text.rstrip()
    return bool(t) and _is_sentence_end_at(t, len(t) - 1)


def _is_sentence_end_at(text, i):
    """text[i] 是否构成句末标点（含英文句点的保护判定）。"""
    ch = text[i]
    if ch in _SENTENCE_PUNCT:
        return True
    if ch == ".":
        return _is_english_period_end(text, i)
    return False


def _is_english_period_end(text, i):
    """英文句点 . 是否为句末：排除小数、点开头 token、单字母缩写。"""
    if i == 0:
        return False
    prev = text[i - 1]
    if not prev.isalnum():
        return False                       # .env / 连续标点 / 句点前是空白
    nxt = text[i + 1] if i + 1 < len(text) else ""
    if prev.isdigit() and nxt.isdigit():
        return False                       # 小数 3.14
    # 单字母缩写保护（e.g. / i.e.）：句点前紧邻字母串长度为 1
    j = i - 1
    while j >= 0 and text[j].isalnum():
        j -= 1
    if (i - 1 - j) < 2 and prev.isascii() and prev.isalpha():
        return False
    if nxt == "" or nxt.isspace() or nxt.isupper() or _is_cjk(nxt):
        return True                        # 句末 / 空白 / 大写起句 / 中文起句（含 back.In）
    return False


def _is_cjk(ch):
    return "一" <= ch <= "鿿"
