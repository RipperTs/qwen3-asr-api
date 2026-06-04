"""app/utils/audio_resampler.py 测试（纯逻辑：PCM 解析 + 重采样）。

torchaudio/librosa 真实 API 已对照已装包核实（resample 收 Tensor / librosa 关键字参数）。
"""
import numpy as np
import pytest

from app.utils.audio_resampler import pcm_bytes_to_array, resample_to_16k


# ─── pcm_bytes_to_array ───

def test_pcm_empty():
    assert pcm_bytes_to_array(b"").shape == (0,)


def test_pcm_values_normalized():
    raw = np.array([0, 16384, -16384, 32767], dtype="<i2").tobytes()
    out = pcm_bytes_to_array(raw)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.0, 0.5, -0.5, 32767 / 32768], rtol=1e-6)


def test_pcm_odd_trailing_byte_dropped():
    raw = np.array([100, 200], dtype="<i2").tobytes() + b"\x01"  # 5 字节，末字节不足一采样
    out = pcm_bytes_to_array(raw)
    assert out.shape == (2,)


def test_pcm_bad_sample_width():
    with pytest.raises(ValueError):
        pcm_bytes_to_array(b"\x00\x00", sample_width=4)


# ─── resample_to_16k ───

def test_resample_16k_fast_path_returns_same():
    a = np.ones(100, dtype=np.float32)
    assert resample_to_16k(a, 16000) is a


def test_resample_empty():
    assert resample_to_16k(np.zeros(0, dtype=np.float32), 8000).size == 0


@pytest.mark.parametrize("src_sr,n_samples,expected_16k", [
    (8000, 8000, 16000),     # 1s @8k -> 16000
    (48000, 48000, 16000),   # 1s @48k -> 16000
    (44100, 44100, 16000),   # 1s @44.1k -> ~16000
])
def test_resample_length_ratio(src_sr, n_samples, expected_16k):
    out = resample_to_16k(np.zeros(n_samples, dtype=np.float32), src_sr)
    assert abs(out.shape[0] - expected_16k) <= 5  # 容忍重采样滤波边界少量误差
    assert out.dtype == np.float32
