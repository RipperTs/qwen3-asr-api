"""app/engines/vad_engine.py::detect 结果解析测试（mock self._model，不加载模型）。

行为依源码确认（vad_engine.py:31）：解析 res[0]["value"] 中的 [start,end] 对。
"""
from unittest.mock import MagicMock

import pytest

from app.engines.vad_engine import VADEngine


def test_detect_requires_loaded():
    eng = VADEngine()
    with pytest.raises(RuntimeError):
        eng.detect("x.wav")


def test_detect_parses_pairs():
    eng = VADEngine()
    eng._model = MagicMock()
    eng._model.generate.return_value = [{"value": [[100, 500], [600, 1200]]}]
    assert eng.detect("x.wav") == [(100, 500), (600, 1200)]


def test_detect_passes_max_end_silence_to_reset_leak():
    # 离线 detect 须显式传 max_end_silence_time：流式会话经 init_cache 写入的共享 vad_opts
    # 不会自动复位，离线不传则沿用上一个流式会话的遗留值导致段边界漂移
    import app.config as cfg
    eng = VADEngine()
    eng._model = MagicMock()
    eng._model.generate.return_value = [{"value": []}]
    eng.detect("x.wav")
    assert eng._model.generate.call_args.kwargs["max_end_silence_time"] == cfg.VAD_MAX_SILENCE


def test_detect_ignores_non_pair_entries():
    eng = VADEngine()
    eng._model = MagicMock()
    eng._model.generate.return_value = [{"value": [[1, 2, 3], [10, 20]]}]
    assert eng.detect("x.wav") == [(10, 20)]


def test_detect_casts_float_to_int():
    eng = VADEngine()
    eng._model = MagicMock()
    eng._model.generate.return_value = [{"value": [[100.7, 500.2]]}]
    assert eng.detect("x.wav") == [(100, 500)]


def test_detect_empty_results():
    eng = VADEngine()
    eng._model = MagicMock()
    eng._model.generate.return_value = []
    assert eng.detect("x.wav") == []


def test_detect_falsy_first_element():
    eng = VADEngine()
    eng._model = MagicMock()
    eng._model.generate.return_value = [None]
    assert eng.detect("x.wav") == []


def test_detect_empty_dict_first_element():
    eng = VADEngine()
    eng._model = MagicMock()
    eng._model.generate.return_value = [{}]
    assert eng.detect("x.wav") == []
