from contextlib import nullcontext
import logging
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi

from app.utils.model_manager import ensure_model_modelscope
from app.config import MODEL_LOCAL_MAP, MODELSCOPE_ONLY_REPO_MAP

logger = logging.getLogger(__name__)

# 3D-Speaker diarization 生产默认滑窗（spike 验证）
WIN_SEC = 1.5
STEP_SEC = 0.75


def make_windows(start_sec: float, end_sec: float,
                 win: float = WIN_SEC, step: float = STEP_SEC) -> list[tuple[float, float]]:
    """滑窗切分（秒），逻辑对齐 3D-Speaker chunk()：窗长落在 (win-step, win]。

    补丁：整段 ≤ win-step 时上游产生 0 窗，此处返回整段作 1 窗，避免短段永远无标签。
    """
    out = []
    t = start_sec
    while t + win < end_sec + step:
        out.append((t, min(t + win, end_sec)))
        t += step
    if not out and end_sec > start_sec:
        out.append((start_sec, end_sec))
    return out


class SpeakerEmbeddingEngine:
    """CAM++ 声纹 embedding 引擎（CPU）。窗级批量提取，输出 L2 归一化的 [N,192]。

    纯 torch 加载（vendored app/engines/campplus + torch.load 权重），
    FBank 用 torchaudio.compliance.kaldi（80mel + CMN），不依赖 modelscope pipeline / funasr。
    """

    BACKEND = "pytorch"
    # 声纹库模板兼容性标识（V 系列衔接面）：权重/特征/归一化任一变更须升版本
    MODEL_TAG = "campplus_cn_common@v1"
    EMB_DIM = 192
    SAMPLE_RATE = 16000
    _BATCH = 64

    def __init__(self, *, priority_gate=None):
        self._model_key = "campplus"
        self._model = None
        # 共享实例跨会话/跨任务调用，模型 forward 必须串行。
        self._infer_lock = threading.Lock()
        # vLLM 模式注入独立门控：实时段优先，离线在 batch 边界让路。
        self._priority_gate = priority_gate

    def load(self):
        local_dir = MODEL_LOCAL_MAP[self._model_key]
        repo_id = MODELSCOPE_ONLY_REPO_MAP[self._model_key]
        ensure_model_modelscope(repo_id, local_dir)

        weight_path = os.path.join(local_dir, "campplus_cn_common.bin")
        size = os.path.getsize(weight_path) if os.path.exists(weight_path) else 0
        if size < 20 * 1024 * 1024:
            raise RuntimeError(
                f"CAM++ 权重异常（{size} B，疑似 LFS 指针或下载不完整）: {weight_path}"
            )

        from app.engines.campplus import CAMPPlus
        model = CAMPPlus(feat_dim=80, embedding_size=self.EMB_DIM)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        self._model = model
        logger.info(f"说话人 embedding 模型已加载 (PyTorch/CPU): {local_dir}")

    @staticmethod
    def _fbank(wav: torch.Tensor) -> torch.Tensor:
        """[T] float32 → [frames, 80]；与 3D-Speaker FBank 等价：kaldi fbank + CMN。"""
        feat = Kaldi.fbank(wav.unsqueeze(0), num_mel_bins=80,
                           sample_frequency=SpeakerEmbeddingEngine.SAMPLE_RATE, dither=0)
        return feat - feat.mean(0, keepdim=True)

    @staticmethod
    def _circle_pad(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """循环补齐至 target_len（对齐 3D-Speaker circle_pad）。"""
        if x.shape[0] >= target_len:
            return x[:target_len]
        n = -(-target_len // x.shape[0])
        return torch.cat([x] * n)[:target_len]

    def _embed_windows(self, wav: np.ndarray, windows: list[tuple[float, float]],
                       *, realtime: bool) -> np.ndarray:
        """按窗口批量提取；有优先门控时，离线在每个 batch 前为实时任务让路。

        wav 须为 16k 单声道 float32（实时侧 final 段、离线侧 soundfile 读出均天然满足）。
        """
        if self._model is None:
            raise RuntimeError("说话人 embedding 模型未加载，请先调用 load()")
        if not windows:
            return np.zeros((0, self.EMB_DIM), dtype=np.float32)

        gate = self._priority_gate
        section = gate.realtime_section() if realtime and gate is not None else nullcontext()
        with section:
            sr = self.SAMPLE_RATE
            clips = []
            for st, ed in windows:
                clip = wav[max(int(st * sr), 0):int(ed * sr)]
                if len(clip) == 0:
                    # 防御：窗落在音频界外（理论不应发生），以零样本占位保持与窗表对齐
                    clip = np.zeros(1, dtype=np.float32)
                clips.append(torch.from_numpy(np.ascontiguousarray(clip)).float())
            # fbank 帧长 25ms（400 样本），全调用统一补齐长度，保持既有 embedding 语义。
            max_len = max(max(c.shape[0] for c in clips), 400)

            if gate is None:
                feats = torch.stack([
                    self._fbank(self._circle_pad(c, max_len)) for c in clips
                ])
                outs = []
                with self._infer_lock, torch.no_grad():
                    for i in range(0, len(feats), self._BATCH):
                        outs.append(self._model(feats[i:i + self._BATCH]))
            else:
                # 特征和 forward 都按 batch 推进；离线不会再跨全部窗口长期占用模型锁。
                outs = []
                with torch.no_grad():
                    for i in range(0, len(clips), self._BATCH):
                        if not realtime:
                            gate.wait_realtime_clear()
                        batch = clips[i:i + self._BATCH]
                        feats = torch.stack([
                            self._fbank(self._circle_pad(c, max_len)) for c in batch
                        ])
                        if not realtime:
                            gate.wait_realtime_clear()
                        with self._infer_lock:
                            outs.append(self._model(feats))
        return F.normalize(torch.cat(outs), dim=1).numpy()

    def embed_windows(self, wav: np.ndarray,
                      windows: list[tuple[float, float]]) -> np.ndarray:
        """按窗口（秒）批量提取。短窗循环补齐至调用内最长。返回 L2 归一化 [N,192]。"""
        return self._embed_windows(wav, windows, realtime=False)

    def _embed_segment(self, wav: np.ndarray, *, realtime: bool) -> np.ndarray:
        embs = self._embed_windows(
            wav,
            make_windows(0.0, len(wav) / self.SAMPLE_RATE),
            realtime=realtime,
        )
        mean = embs.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 0 else mean

    def embed_segment(self, wav: np.ndarray) -> np.ndarray:
        """整段提取：滑窗 embedding 均值 + 重归一化。返回 [192]。"""
        return self._embed_segment(wav, realtime=False)

    def embed_realtime_segment(self, wav: np.ndarray) -> np.ndarray:
        """实时整段提取；有优先门控时，阻止离线任务继续抢占后续 batch。"""
        return self._embed_segment(wav, realtime=True)

    def unload(self):
        self._model = None
        logger.info("说话人 embedding 模型已卸载")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
