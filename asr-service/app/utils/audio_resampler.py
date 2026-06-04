"""内存音频工具：裸 PCM 解析与重采样（实时流式用，不落盘、不调 ffmpeg）。

依赖均已在 requirements：numpy、torchaudio、librosa。
重采样优先 torchaudio.functional.resample（收/返 Tensor），异常时回退 librosa.resample。
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)

TARGET_SR = 16000


def pcm_bytes_to_array(pcm: bytes, sample_width: int = 2) -> np.ndarray:
    """裸 PCM16(小端) 单声道 → float32 [-1, 1)。

    参数:
        pcm: 原始 PCM 字节
        sample_width: 采样字节宽，目前仅支持 2（16-bit）

    边界:
        空 bytes → 空数组；末尾不足一个采样的字节被丢弃。
    """
    if sample_width != 2:
        raise ValueError(f"仅支持 16-bit PCM (sample_width=2)，收到 {sample_width}")
    if not pcm:
        return np.zeros(0, dtype=np.float32)

    usable = len(pcm) - (len(pcm) % sample_width)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)

    ints = np.frombuffer(pcm[:usable], dtype="<i2")
    return ints.astype(np.float32) / 32768.0


def resample_to_16k(audio: np.ndarray, src_sr: int) -> np.ndarray:
    """将单声道 float 数组从 src_sr 重采样到 16kHz。

    src_sr == 16000 或空数组走快速路径直接返回。优先 torchaudio，失败回退 librosa。
    """
    audio = np.asarray(audio, dtype=np.float32)
    if src_sr == TARGET_SR or audio.size == 0:
        return audio

    try:
        import torch
        import torchaudio.functional as AF
        tensor = torch.from_numpy(np.ascontiguousarray(audio))
        out = AF.resample(tensor, orig_freq=src_sr, new_freq=TARGET_SR)
        return out.numpy().astype(np.float32, copy=False)
    except Exception as e:
        logger.warning(f"torchaudio 重采样失败，回退 librosa: {e}")
        import librosa
        out = librosa.resample(audio, orig_sr=src_sr, target_sr=TARGET_SR)
        return out.astype(np.float32, copy=False)
