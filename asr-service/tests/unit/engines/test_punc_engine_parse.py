"""app/engines/punc_engine.py::restore 结果解析测试（mock self._model，不加载模型）。

行为依源码确认（punc_engine.py:44）。
"""
from unittest.mock import MagicMock

import pytest

from app.engines.punc_engine import PuncEngine


def test_restore_requires_loaded():
    eng = PuncEngine()
    with pytest.raises(RuntimeError):
        eng.restore("hello")


def test_restore_empty_or_blank_returns_same_without_calling_model():
    eng = PuncEngine()
    eng._model = MagicMock()
    assert eng.restore("") == ""
    assert eng.restore("   ") == "   "
    eng._model.generate.assert_not_called()


def test_restore_adds_punctuation():
    eng = PuncEngine()
    eng._model = MagicMock()
    eng._model.generate.return_value = [{"text": "你好。"}]
    assert eng.restore("你好") == "你好。"


def test_restore_empty_result_returns_original():
    eng = PuncEngine()
    eng._model = MagicMock()
    eng._model.generate.return_value = []
    assert eng.restore("abc") == "abc"


def test_restore_missing_text_key_returns_original():
    eng = PuncEngine()
    eng._model = MagicMock()
    eng._model.generate.return_value = [{}]
    assert eng.restore("abc") == "abc"
