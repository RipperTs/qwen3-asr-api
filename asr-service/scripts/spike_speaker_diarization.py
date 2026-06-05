"""CAM++ 说话人分离 spike（S0）—— 阈值标定与可行性实测。

实验内容（对应实施方案 §6.2）：
  1. 权重完整性校验（>20MB 防 LFS 指针）+ 纯 torch 加载
  2. 官方样例 1:1 验证：同人/异人整句相似度 vs 官方阈 0.31
  3. 窗级相似度分布（同人窗内 / 同人跨句 / 异人）→ 推导聚类阈值
  4. 离线 AHC 阈值扫描 + 谱聚类分支（合成对话，标签有真值）
  5. 在线增量聚类 τ 扫描 + 短段可靠性（标定 min_seg_ms）
  6. funasr AutoModel 加载验证（仅记录结论）
  7. CPU 性能（ms/窗、RTF）

用法：venv/bin/python scripts/spike_speaker_diarization.py
依赖：torch / torchaudio / numpy / scipy / scikit-learn / soundfile / modelscope（venv 已含）

说明：
- 模型定义 import 自 app.engines.campplus（vendored，纯 torch，无 app 配置依赖）；
- 权重与样例经 modelscope 下载到 models/speaker/campplus，本脚本不读取 docs/plan 下任何文件；
- 样例仅 2 人 3 句（speaker1_a/b、speaker2_a），多人真实对话的复核留待端到端手测。
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio.compliance.kaldi as Kaldi
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment

ASR_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASR_SERVICE_ROOT))

from app.engines.campplus import CAMPPlus  # noqa: E402  纯 torch 模型定义，无运行时依赖

MODEL_DIR = ASR_SERVICE_ROOT / "models" / "speaker" / "campplus"
WEIGHT_FILE = MODEL_DIR / "campplus_cn_common.bin"
EXAMPLES = {
    "sp1_a": MODEL_DIR / "examples" / "speaker1_a_cn_16k.wav",
    "sp1_b": MODEL_DIR / "examples" / "speaker1_b_cn_16k.wav",
    "sp2_a": MODEL_DIR / "examples" / "speaker2_a_cn_16k.wav",
}
SR = 16000
WIN_SEC, STEP_SEC = 1.5, 0.75
BATCH = 64

torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------- 基础组件

def ensure_weights():
    if not WEIGHT_FILE.exists():
        from modelscope import snapshot_download
        snapshot_download(model_id="iic/speech_campplus_sv_zh-cn_16k-common",
                          local_dir=str(MODEL_DIR))
    size = WEIGHT_FILE.stat().st_size
    assert size > 20 * 1024 * 1024, f"权重疑似 LFS 指针（{size}B），下载失败"
    return size


def load_model():
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    state = torch.load(WEIGHT_FILE, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


def fbank(wav: torch.Tensor) -> torch.Tensor:
    """[T] float32 → [frames, 80]，kaldi fbank + CMN（对齐 3D-Speaker FBank）。"""
    feat = Kaldi.fbank(wav.unsqueeze(0), num_mel_bins=80,
                       sample_frequency=SR, dither=0)
    return feat - feat.mean(0, keepdim=True)


def make_windows(st: float, ed: float, win=WIN_SEC, step=STEP_SEC):
    """对齐 3D-Speaker chunk()：窗长落在 (0.75, 1.5]s；补丁：整段 ≤0.75s 时返回整段 1 窗。"""
    out = []
    t = st
    while t + win < ed + step:
        out.append((t, min(t + win, ed)))
        t += step
    if not out:
        out.append((st, ed))
    return out


def circle_pad(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.shape[0] >= target_len:
        return x[:target_len]
    n = int(np.ceil(target_len / x.shape[0]))
    return torch.cat([x] * n)[:target_len]


def embed_windows(model, wav: np.ndarray, windows) -> np.ndarray:
    """按窗批量提取，L2 归一化 [N,192]。"""
    clips = [torch.from_numpy(wav[int(st * SR):int(ed * SR)]).float() for st, ed in windows]
    max_len = max(c.shape[0] for c in clips)
    feats = torch.stack([fbank(circle_pad(c, max_len)) for c in clips])
    embs = []
    with torch.no_grad():
        for i in range(0, len(feats), BATCH):
            embs.append(model(feats[i:i + BATCH]))
    return F.normalize(torch.cat(embs), dim=1).numpy()


def embed_segment(model, wav: np.ndarray) -> np.ndarray:
    """整段：滑窗均值 + 重归一化（实时侧 embed_segment 的原型）。"""
    embs = embed_windows(model, wav, make_windows(0, len(wav) / SR))
    mean = embs.mean(0)
    return mean / np.linalg.norm(mean)


def stats(arr) -> str:
    a = np.asarray(arr)
    return (f"n={len(a)} mean={a.mean():.3f} std={a.std():.3f} "
            f"min={a.min():.3f} p5={np.percentile(a, 5):.3f} "
            f"p50={np.percentile(a, 50):.3f} p95={np.percentile(a, 95):.3f} max={a.max():.3f}")


def label_accuracy(pred, truth) -> float:
    """聚类标签与真值的最优映射准确率（匈牙利算法）。"""
    pred, truth = np.asarray(pred), np.asarray(truth)
    pl, tl = np.unique(pred), np.unique(truth)
    cm = np.zeros((len(pl), len(tl)))
    for i, p in enumerate(pl):
        for j, t in enumerate(tl):
            cm[i, j] = np.sum((pred == p) & (truth == t))
    ri, ci = linear_sum_assignment(-cm)
    return cm[ri, ci].sum() / len(pred)


# ---------------------------------------------------------------- 聚类原型

def ahc_cluster(embs: np.ndarray, sim_thr: float) -> np.ndarray:
    """AHC average linkage；sim_thr 为余弦相似度阈（≥ 则并簇），等价距离切点 1-sim_thr。"""
    z = linkage(embs, method="average", metric="cosine")
    return fcluster(z, t=1.0 - sim_thr, criterion="distance") - 1


def spectral_cluster(embs: np.ndarray, pval=0.012, min_pnum=6,
                     min_spks=1, max_spks=15) -> np.ndarray:
    """改写自 3D-Speaker SpectralCluster（未归一化拉普拉斯 + eigen-gap + k-means）。"""
    from sklearn.cluster._kmeans import k_means
    sim = embs @ embs.T  # 已 L2 归一化
    n = sim.shape[0]
    n_prune = min(int((1 - pval) * n), n - min_pnum)
    pruned = sim.copy()
    for i in range(n):
        pruned[i, np.argsort(pruned[i])[:n_prune]] = 0
    sym = 0.5 * (pruned + pruned.T)
    np.fill_diagonal(sym, 0)
    lap = -sym
    lap[np.diag_indices(n)] = np.abs(sym).sum(1)
    import scipy.sparse.linalg as sla
    k = min(max_spks + 1, n)
    lambdas, vecs = sla.eigsh(lap, k=k, which="SM")
    gaps = np.diff(lambdas[min_spks - 1:max_spks + 1])
    num_spks = int(np.argmax(gaps)) + min_spks
    _, labels, _ = k_means(vecs[:, :num_spks], num_spks, n_init=10, random_state=42)
    return labels


def post_process(labels: np.ndarray, embs: np.ndarray,
                 min_cluster_size=4, mer_cos=0.8) -> np.ndarray:
    """小簇并入最近大簇 + 质心相似度 > mer_cos 的簇合并（对齐 CommonClustering）。"""
    labels = labels.copy()
    uniq, counts = np.unique(labels, return_counts=True)
    major = uniq[counts > min_cluster_size]
    if len(major) and len(major) < len(uniq):
        centers = np.stack([embs[labels == c].mean(0) for c in major])
        for i in np.where(~np.isin(labels, major))[0]:
            labels[i] = major[np.argmax(centers @ embs[i])]
    while True:
        uniq = np.unique(labels)
        if len(uniq) == 1:
            break
        centers = np.stack([embs[labels == c].mean(0) for c in uniq])
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        aff = np.triu(centers @ centers.T, 1)
        i, j = np.unravel_index(np.argmax(aff), aff.shape)
        if aff[i, j] < mer_cos:
            break
        labels[labels == uniq[j]] = uniq[i]
    return labels


class OnlineClusterer:
    """在线增量聚类原型：质心 + 余弦阈值，计数加权更新。"""

    def __init__(self, threshold: float, min_seg_ms=800, max_speakers=8):
        self.thr, self.min_seg_ms, self.max = threshold, min_seg_ms, max_speakers
        self.centroids, self.counts = [], []

    def assign(self, emb: np.ndarray, dur_ms: int):
        sims = [c @ emb for c in self.centroids]
        best = max(sims, default=-1.0)
        idx = int(np.argmax(sims)) if sims else -1
        if dur_ms < self.min_seg_ms:
            return idx if best >= self.thr else None
        if best >= self.thr:
            c = self.centroids[idx] * self.counts[idx] + emb
            self.centroids[idx] = c / np.linalg.norm(c)
            self.counts[idx] += 1
            return idx
        if len(self.centroids) >= self.max:
            return idx
        self.centroids.append(emb.copy())
        self.counts.append(1)
        return len(self.centroids) - 1


# ---------------------------------------------------------------- 实验

def build_conversation(wavs, order, gap_sec=0.3):
    """按 order 拼接合成对话；返回 (wav, vad_segments[(st,ed)], seg_speaker[真值])。"""
    gap = np.zeros(int(gap_sec * SR), dtype=np.float32)
    pieces, segs, speakers = [], [], []
    cur = 0.0
    for key in order:
        w = wavs[key]
        st, ed = cur, cur + len(w) / SR
        segs.append((st, ed))
        speakers.append(key[:3])  # sp1 / sp2
        pieces += [w, gap]
        cur = ed + gap_sec
    return np.concatenate(pieces), segs, speakers


def main():
    print("=" * 70)
    print("EXP1 权重与加载")
    size = ensure_weights()
    t0 = time.perf_counter()
    model, n_params = load_model()
    print(f"  权重 {size:,} B（>20MB 校验通过）；加载 {time.perf_counter() - t0:.2f}s；"
          f"参数量 {n_params / 1e6:.2f}M")

    wavs = {k: sf.read(p, dtype="float32")[0] for k, p in EXAMPLES.items()}

    print("=" * 70)
    print("EXP2 官方样例 1:1 验证（整句 embedding 余弦，官方判决阈 0.31）")
    utt = {k: embed_segment(model, w) for k, w in wavs.items()}
    same = utt["sp1_a"] @ utt["sp1_b"]
    diff1 = utt["sp1_a"] @ utt["sp2_a"]
    diff2 = utt["sp1_b"] @ utt["sp2_a"]
    print(f"  同人 sp1_a×sp1_b = {same:.4f}（应 > 0.31）")
    print(f"  异人 sp1_a×sp2_a = {diff1:.4f}，sp1_b×sp2_a = {diff2:.4f}（应 < 0.31）")

    print("=" * 70)
    print("EXP3 窗级相似度分布（1.5s/0.75s 滑窗）")
    win_embs = {k: embed_windows(model, w, make_windows(0, len(w) / SR))
                for k, w in wavs.items()}
    for k, e in win_embs.items():
        print(f"  {k}: {len(e)} 窗")
    intra = [e[i] @ e[j] for e in win_embs.values()
             for i in range(len(e)) for j in range(i + 1, len(e))]
    cross_same = [a @ b for a in win_embs["sp1_a"] for b in win_embs["sp1_b"]]
    inter = [a @ b for k in ("sp1_a", "sp1_b") for a in win_embs[k]
             for b in win_embs["sp2_a"]]
    print(f"  同人句内: {stats(intra)}")
    print(f"  同人跨句: {stats(cross_same)}")
    print(f"  异人    : {stats(inter)}")
    gap_lo, gap_hi = np.percentile(inter, 95), np.percentile(cross_same, 5)
    print(f"  → 异人 p95={gap_lo:.3f} ↔ 同人跨句 p5={gap_hi:.3f}，"
          f"可分隔带 {'存在' if gap_lo < gap_hi else '不存在'}，中点 {((gap_lo + gap_hi) / 2):.3f}")

    print("=" * 70)
    print("EXP4 离线 AHC 阈值扫描（合成对话 sp1_a→sp2_a→sp1_b，真值 2 人）")
    conv, segs, seg_spk = build_conversation(wavs, ["sp1_a", "sp2_a", "sp1_b"])
    windows, win_truth = [], []
    for (st, ed), spk in zip(segs, seg_spk):
        ws = make_windows(st, ed)
        windows += ws
        win_truth += [spk] * len(ws)
    embs = embed_windows(model, conv, windows)
    print(f"  共 {len(windows)} 窗（<40 → 生产路径走 AHC 分支）")
    for sim_thr in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        raw = ahc_cluster(embs, sim_thr)
        post = post_process(raw, embs)
        acc = label_accuracy(post, win_truth)
        n_raw, n_post = len(np.unique(raw)), len(np.unique(post))
        print(f"  sim_thr={sim_thr:.2f}: 原始 {n_raw} 簇 → 后处理 {n_post} 簇，"
              f"窗准确率 {acc:.1%}")
    print("  （上游 AHCluster 语义 = 相似度 ≥0.4 并簇，等价本表 sim_thr=0.40）")

    print("=" * 70)
    print("EXP5 谱聚类分支（对话平铺 ×3 → ≥40 窗，pval=0.012）")
    conv3, segs3, seg_spk3 = build_conversation(
        wavs, ["sp1_a", "sp2_a", "sp1_b", "sp2_a", "sp1_a", "sp2_a", "sp1_b"])
    windows3, truth3 = [], []
    for (st, ed), spk in zip(segs3, seg_spk3):
        ws = make_windows(st, ed)
        windows3 += ws
        truth3 += [spk] * len(ws)
    embs3 = embed_windows(model, conv3, windows3)
    raw3 = spectral_cluster(embs3)
    post3 = post_process(raw3, embs3)
    acc3 = label_accuracy(post3, truth3)
    print(f"  {len(windows3)} 窗：谱聚类 {len(np.unique(raw3))} 簇 → "
          f"后处理 {len(np.unique(post3))} 簇，窗准确率 {acc3:.1%}（真值 2 人）")
    ahc3 = post_process(ahc_cluster(embs3, 0.40), embs3)
    print(f"  对照：同 {len(windows3)} 窗 AHC(sim_thr=0.40) → "
          f"{len(np.unique(ahc3))} 簇，窗准确率 {label_accuracy(ahc3, truth3):.1%}"
          f"（cluster_line 取值的依据）")

    print("=" * 70)
    print("EXP6 在线增量聚类 τ 扫描（按 final 段顺序喂入，真值 A,B,A,B,A,B,A）")
    seg_means = []
    for (st, ed) in segs3:
        seg_wav = conv3[int(st * SR):int(ed * SR)]
        seg_means.append(embed_segment(model, seg_wav))
    truth_seq = [0 if s == "sp1" else 1 for s in seg_spk3]
    for tau in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        oc = OnlineClusterer(threshold=tau)
        pred = [oc.assign(e, 3000) for e in seg_means]
        n_spk = len(oc.centroids)
        acc = label_accuracy(pred, truth_seq)
        print(f"  τ={tau:.2f}: 判定 {n_spk} 人，序列 {pred}，准确率 {acc:.1%}")

    print("=" * 70)
    print("EXP7 短段可靠性（标定 min_seg_ms：短切片 embedding vs 本人整句质心）")
    for dur in (0.3, 0.5, 0.8, 1.0, 1.5):
        n = int(dur * SR)
        same_sims, diff_sims = [], []
        # 仅取跨句对（切片与参照不同句），避免自包含膨胀；sp2 只有单句故不参与同人侧
        for key, ref in (("sp1_b", "sp1_a"), ("sp1_a", "sp1_b")):
            w = wavs[key]
            for off in range(0, len(w) - n, n):
                clip = w[off:off + n]
                e = embed_windows(model, clip, [(0, dur)])[0]
                same_sims.append(e @ utt[ref])
                diff_sims.append(e @ utt["sp2_a"])
        print(f"  {int(dur * 1000)}ms: 同人 {stats(same_sims)}")
        print(f"  {' ' * len(str(int(dur * 1000)))}    异人 {stats(diff_sims)}")

    print("=" * 70)
    print("EXP8 funasr AutoModel 加载验证（仅记录，不作主线依赖）")
    try:
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            from funasr import AutoModel
            am = AutoModel(model=str(MODEL_DIR), disable_update=True)
        print(f"  ✅ AutoModel 加载成功: {type(am.model).__name__}")
    except Exception as e:
        print(f"  ❌ AutoModel 加载失败: {type(e).__name__}: {e}")

    print("=" * 70)
    print("EXP9 CPU 性能（批量 64）")
    long_wav = np.tile(wavs["sp1_a"], 20)  # ~74s
    perf_windows = make_windows(0, len(long_wav) / SR)
    t0 = time.perf_counter()
    embed_windows(model, long_wav, perf_windows)
    cost = time.perf_counter() - t0
    audio_sec = len(long_wav) / SR
    print(f"  {len(perf_windows)} 窗 / {audio_sec:.0f}s 音频：{cost:.2f}s，"
          f"{cost / len(perf_windows) * 1000:.1f}ms/窗，RTF={cost / audio_sec:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
