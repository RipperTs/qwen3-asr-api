"""qwen_asr transcribe 结果解析共享工具（离线管线与实时会话共用）。

从 ASRTranscription 结果中提取纯文本与单词级时间戳，
统一 asr_pipeline 与 stream_session 两处的解析逻辑，避免拷贝发散。
"""


def extract_text(results) -> str:
    """从 qwen_asr transcribe 结果中提取纯文本"""
    if not results:
        return ""
    if isinstance(results, str):
        return results
    if isinstance(results, list):
        texts = []
        for item in results:
            if hasattr(item, "text"):
                texts.append(item.text)
            elif isinstance(item, dict):
                texts.append(item.get("text", ""))
            elif isinstance(item, str):
                texts.append(item)
        return "".join(texts)
    if hasattr(results, "text"):
        return results.text
    return str(results)


def extract_words(results, offset_sec: float) -> list[dict] | None:
    """从 qwen_asr 结果中提取单词级时间戳（带偏移修正）"""
    if not results or not isinstance(results, list):
        return None

    words = []
    for item in results:
        # ASRTranscription.time_stamps -> ForcedAlignResult.items -> [ForcedAlignItem]
        ts = getattr(item, "time_stamps", None)
        if ts is None:
            continue
        for w in getattr(ts, "items", []):
            words.append({
                "text": w.text,
                "start": round(w.start_time + offset_sec, 3),
                "end": round(w.end_time + offset_sec, 3),
            })
    return words if words else None
