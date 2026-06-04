"""app/utils/result_parser.py 测试（离线管线与实时会话共享的解析工具）。

asr_pipeline._extract_text/_extract_words 已委托至此模块，
test_asr_pipeline_pure.py 经委托路径覆盖；此处直接覆盖模块函数。
"""
import types

from app.utils.result_parser import extract_text, extract_words


# ─── extract_text ───

def test_extract_text_empty_and_none():
    assert extract_text(None) == ""
    assert extract_text([]) == ""
    assert extract_text("") == ""


def test_extract_text_str_passthrough():
    assert extract_text("hello") == "hello"


def test_extract_text_mixed_list():
    items = [
        types.SimpleNamespace(text="你"),
        {"text": "好"},
        "！",
    ]
    assert extract_text(items) == "你好！"


def test_extract_text_single_object():
    assert extract_text(types.SimpleNamespace(text="hi")) == "hi"


# ─── extract_words ───

def _aligned_item(text, start, end):
    w = types.SimpleNamespace(text=text, start_time=start, end_time=end)
    return types.SimpleNamespace(text=text, time_stamps=types.SimpleNamespace(items=[w]))


def test_extract_words_with_offset():
    words = extract_words([_aligned_item("a", 0.1, 0.5)], offset_sec=1.0)
    assert words == [{"text": "a", "start": 1.1, "end": 1.5}]


def test_extract_words_none_when_no_timestamps():
    assert extract_words([types.SimpleNamespace(text="x", time_stamps=None)], 0.0) is None
    assert extract_words(None, 0.0) is None
    assert extract_words("not-a-list", 0.0) is None
