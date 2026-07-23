"""app/runtime/speaker_service.py 单元测试（真 SpeakerStore 临时库 + fake 引擎/VAD）。

覆盖：enroll 质量门槛（时长/多人/consent）、模板均值入库、identify_file、
map_clusters 异常兜底、自动登记分支全集（过门槛/时长不足/开关关/序号递增/失败退回匿名）、
临时文件清理、留存音频。阈值只测逻辑分支（V0 标定铁律）。
"""
import os
import threading
import types

import numpy as np
import pytest

import app.config as cfg
from app.runtime.speaker_service import SpeakerService
from app.runtime.speaker_store import SpeakerClaimConflictError, SpeakerStore

DIM = 192
TAG = "campplus_cn_common@v1"


def unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


class FakeEngine:
    """窗起点 < split_at 秒 → unit(vec_idx)，否则 unit(vec_idx+1)（制造多人样本）。"""

    def __init__(self, vec_idx=0, split_at=None):
        self.vec_idx = vec_idx
        self.split_at = split_at

    def embed_windows(self, wav, windows, *, cancelled=None):
        out = []
        for st, _ in windows:
            i = self.vec_idx + (1 if self.split_at is not None and st >= self.split_at else 0)
            out.append(unit(i))
        return np.stack(out)


def make_vad(segments):
    return types.SimpleNamespace(detect=lambda p: segments)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """隔离上传目录/服务根 + fake 音频 IO（convert 落空文件、sf.read 给 8s 假音频）。"""
    monkeypatch.setattr(cfg, "UPLOADS_DIR", str(tmp_path / "up"))
    monkeypatch.setattr(cfg, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr("app.runtime.speaker_service.convert_to_wav",
                        lambda src, dst: open(dst, "wb").write(b"x"))
    monkeypatch.setattr("app.runtime.speaker_service.sf.read",
                        lambda p, dtype=None: (np.zeros(16000 * 8, dtype=np.float32), 16000))
    return tmp_path


@pytest.fixture
def store(tmp_path):
    s = SpeakerStore(str(tmp_path / "speakers.db"), model_tag=TAG)
    yield s
    s.close()


def make_service(store, engine=None, vad_segments=((0, 5000),)):
    return SpeakerService(store, engine or FakeEngine(), make_vad(list(vad_segments)))


def _src(tmp_path, name="a.mp3"):
    p = tmp_path / name
    p.write_bytes(b"fake")
    return str(p)


# ─── enroll ───

def test_enroll_ok_with_quality_hint(env, store):
    svc = make_service(store)
    resp = svc.enroll("张三", "备注", [_src(env)], consent=True)
    assert len(resp["speaker_id"]) == 32
    assert resp["templates"] == 1
    assert "quality_hint" in resp                       # 模板 <3 提示
    assert store.get_speaker(resp["speaker_id"])["source"] == "manual"


def test_enroll_three_samples_no_hint(env, store):
    svc = make_service(store)
    resp = svc.enroll("张三", None, [_src(env, f"{i}.mp3") for i in range(3)], consent=True)
    assert resp["templates"] == 3
    assert "quality_hint" not in resp


def test_enroll_rejects_short_speech(env, store):
    svc = make_service(store, vad_segments=[(0, 2000)])   # 2s < 3s 门槛
    with pytest.raises(ValueError, match="有效语音不足"):
        svc.enroll("x", None, [_src(env)], consent=True)


def test_enroll_rejects_multi_speaker_sample(env, store):
    svc = make_service(store, engine=FakeEngine(split_at=2.5))   # 前后两人
    with pytest.raises(ValueError, match="多个说话人"):
        svc.enroll("x", None, [_src(env)], consent=True)


def test_enroll_requires_consent(env, store):
    svc = make_service(store)
    with pytest.raises(ValueError, match="consent"):
        svc.enroll("x", None, [_src(env)], consent=False)


def test_enroll_rejects_empty_files(env, store):
    with pytest.raises(ValueError, match="样本"):
        make_service(store).enroll("x", None, [], consent=True)


def test_temp_wavs_cleaned_on_success_and_failure(env, store):
    svc = make_service(store)
    svc.enroll("a", None, [_src(env)], consent=True)
    svc_fail = make_service(store, vad_segments=[(0, 1000)])
    with pytest.raises(ValueError):
        svc_fail.enroll("b", None, [_src(env)], consent=True)
    leftovers = [f for f in os.listdir(cfg.UPLOADS_DIR) if f.startswith("spk_")]
    assert leftovers == []


def test_store_audio_kept_and_removed_on_delete(env, store, monkeypatch):
    monkeypatch.setattr(cfg, "SPEAKER_STORE_AUDIO", True)
    svc = make_service(store)
    sid = svc.enroll("a", None, [_src(env)], consent=True)["speaker_id"]
    audio_dir = os.path.join(str(env), "data", "speaker_audio", sid)
    assert os.path.isfile(os.path.join(audio_dir, "00.wav"))
    svc.delete_speaker(sid)
    assert not os.path.isdir(audio_dir)                  # 被遗忘权：音频同步清理


# ─── claim ───

def test_claim_auto_speaker_replaces_templates(env, store):
    sid = store.enroll_speaker(
        "说话人_01",
        None,
        [unit(5)],
        [12.0],
        consent=True,
        source="auto",
    )
    svc = make_service(store)

    response = svc.claim(
        sid,
        " 张三 ",
        "产品部",
        [_src(env, "claim.mp3")],
        True,
        " profile-1 ",
    )

    assert response["speaker_id"] == sid
    assert response["source"] == "manual"
    assert response["templates"] == 1
    assert len(response["template_ids"]) == 1
    assert response["quality_hint"]
    info = store.get_speaker(sid)
    assert info["name"] == "张三" and info["note"] == "产品部"
    assert info["source"] == "manual"
    assert store.identify(unit(0))["speaker_id"] == sid


def test_claim_replay_skips_audio_processing(env, store, monkeypatch):
    sid = store.enroll_speaker(
        "说话人_01", None, [unit(5)], [12.0], True, source="auto"
    )
    svc = make_service(store)
    first = svc.claim(
        sid, "张三", None, [_src(env)], True, "profile-1"
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("幂等重试不应重新处理音频")

    monkeypatch.setattr(svc, "_embed_file", fail_if_called)
    replay = svc.claim(
        sid, "其他名称", "其他备注", [_src(env, "retry.mp3")],
        True, "profile-1",
    )
    assert replay == first


def test_claim_manual_conflict_happens_before_audio_processing(
    env, store, monkeypatch
):
    sid = store.enroll_speaker(
        "张三", None, [unit(0)], [5.0], consent=True
    )
    svc = make_service(store)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("手动声纹冲突不应处理音频")

    monkeypatch.setattr(svc, "_embed_file", fail_if_called)
    with pytest.raises(SpeakerClaimConflictError) as exc:
        svc.claim(
            sid, "李四", None, [_src(env)], True, "profile-1"
        )
    assert exc.value.code == "speaker_not_claimable"


def test_claim_validates_consent_name_and_claim_key(env, store):
    sid = store.enroll_speaker(
        "说话人_01", None, [unit(5)], [12.0], True, source="auto"
    )
    svc = make_service(store)
    path = _src(env)

    with pytest.raises(ValueError, match="consent"):
        svc.claim(sid, "张三", None, [path], False, "profile-1")
    with pytest.raises(ValueError, match="名称"):
        svc.claim(sid, " ", None, [path], True, "profile-1")
    with pytest.raises(ValueError, match="claim_key"):
        svc.claim(sid, "张三", None, [path], True, " ")


def test_claim_replaces_retained_audio(env, store, monkeypatch):
    monkeypatch.setattr(cfg, "SPEAKER_STORE_AUDIO", True)
    sid = store.enroll_speaker(
        "说话人_01", None, [unit(5)], [12.0], True, source="auto"
    )
    audio_dir = os.path.join(str(env), "data", "speaker_audio", sid)
    os.makedirs(audio_dir, exist_ok=True)
    with open(os.path.join(audio_dir, "old.wav"), "wb") as file:
        file.write(b"old")

    response = make_service(store).claim(
        sid,
        "张三",
        None,
        [_src(env, "a.mp3"), _src(env, "b.mp3")],
        True,
        "profile-1",
    )

    expected = {f"{template_id}.wav" for template_id in response["template_ids"]}
    assert set(os.listdir(audio_dir)) == expected


def test_claim_and_delete_serialize_retained_audio(
    env, store, monkeypatch
):
    monkeypatch.setattr(cfg, "SPEAKER_STORE_AUDIO", True)
    sid = store.enroll_speaker(
        "说话人_01", None, [unit(5)], [12.0], True, source="auto"
    )
    svc = make_service(store)
    audio_dir = os.path.join(str(env), "data", "speaker_audio", sid)
    replace_started = threading.Event()
    allow_replace = threading.Event()
    delete_started = threading.Event()
    delete_finished = threading.Event()
    errors = []
    original_replace = svc._replace_retained_audio

    def blocking_replace(*args):
        replace_started.set()
        if not allow_replace.wait(timeout=2):
            raise AssertionError("测试未放行留存音频替换")
        original_replace(*args)

    def run_claim():
        try:
            svc.claim(
                sid, "张三", None, [_src(env)], True, "profile-1"
            )
        except Exception as exc:  # pragma: no cover - 失败由主线程断言
            errors.append(exc)

    def run_delete():
        delete_started.set()
        try:
            svc.delete_speaker(sid)
        except Exception as exc:  # pragma: no cover - 失败由主线程断言
            errors.append(exc)
        finally:
            delete_finished.set()

    monkeypatch.setattr(svc, "_replace_retained_audio", blocking_replace)
    claim_thread = threading.Thread(target=run_claim)
    delete_thread = threading.Thread(target=run_delete)
    claim_thread.start()
    assert replace_started.wait(timeout=2)
    delete_thread.start()
    assert delete_started.wait(timeout=2)
    try:
        assert not delete_finished.wait(timeout=0.1)
    finally:
        allow_replace.set()
        claim_thread.join(timeout=2)
        delete_thread.join(timeout=2)

    assert not claim_thread.is_alive() and not delete_thread.is_alive()
    assert errors == []
    assert store.get_speaker(sid) is None
    assert not os.path.isdir(audio_dir)


# ─── identify_file ───

def test_identify_file_hit_and_miss(env, store):
    svc = make_service(store)
    sid = svc.enroll("张三", None, [_src(env)], consent=True)["speaker_id"]
    hit = svc.identify_file(_src(env, "q.mp3"))
    assert hit["matched"] is True and hit["speaker_id"] == sid and hit["name"] == "张三"
    assert hit["source"] == "manual"

    svc_other = make_service(store, engine=FakeEngine(vec_idx=5))
    assert svc_other.identify_file(_src(env, "q2.mp3")) == {"matched": False}


def test_identify_file_returns_auto_source(env, store):
    sid = store.enroll_speaker(
        "说话人_01", None, [unit(0)], [12.0], True, source="auto"
    )
    hit = make_service(store).identify_file(_src(env))
    assert hit["speaker_id"] == sid
    assert hit["name"] == "说话人_01"
    assert hit["source"] == "auto"


# ─── map_clusters（实时联动：仅识别）───

def test_map_clusters_hit_and_miss(env, store):
    svc = make_service(store)
    sid = svc.enroll("张三", None, [_src(env)], consent=True)["speaker_id"]
    out = svc.map_clusters([
        {"label": "A", "centroid": unit(0)},
        {"label": "B", "centroid": unit(7)},
    ])
    assert out[0]["speaker_id"] == sid and out[0]["name"] == "张三"
    assert out[1] == {"label": "B", "speaker_id": None, "name": None, "score": None}


def test_map_clusters_exception_falls_back_anonymous(env, store):
    svc = make_service(store)
    out = svc.map_clusters([{"label": "A"}])             # 缺 centroid → 内部异常
    assert out == [{"label": "A", "speaker_id": None, "name": None, "score": None}]


def test_map_clusters_never_auto_enrolls(env, store):
    svc = make_service(store)
    svc.map_clusters([{"label": "A", "centroid": unit(3), "dur_sec": 99.0}])
    assert store.speaker_count == 0                      # 实时路径绝不建档


# ─── map_and_enroll_clusters（离线联动：识别 + 自动登记）───

def _cluster(label, vec, dur):
    return {"label": label, "centroid": vec, "dur_sec": dur}


def test_auto_enroll_above_threshold(env, store):
    svc = make_service(store)
    out = svc.map_and_enroll_clusters([_cluster("A", unit(0), 12.0)])
    assert out[0]["name"] == "说话人_01" and out[0]["auto_enrolled"] is True
    assert store.get_speaker(out[0]["speaker_id"])["source"] == "auto"


def test_auto_enroll_sequence_increments(env, store):
    svc = make_service(store)
    out = svc.map_and_enroll_clusters([
        _cluster("A", unit(0), 12.0), _cluster("B", unit(1), 15.0),
    ])
    assert [m["name"] for m in out] == ["说话人_01", "说话人_02"]


def test_auto_enroll_below_duration_stays_anonymous(env, store):
    svc = make_service(store)
    out = svc.map_and_enroll_clusters([_cluster("A", unit(0), 5.0)])  # <10s
    assert out[0]["speaker_id"] is None
    assert store.speaker_count == 0


def test_auto_enroll_disabled_stays_anonymous(env, store, monkeypatch):
    monkeypatch.setattr(cfg, "SPEAKER_AUTO_ENROLL", False)
    svc = make_service(store)
    out = svc.map_and_enroll_clusters([_cluster("A", unit(0), 12.0)])
    assert out[0]["speaker_id"] is None
    assert store.speaker_count == 0


def test_auto_enroll_hit_does_not_re_enroll(env, store):
    svc = make_service(store)
    sid = svc.enroll("张三", None, [_src(env)], consent=True)["speaker_id"]
    out = svc.map_and_enroll_clusters([_cluster("A", unit(0), 12.0)])
    assert out[0]["speaker_id"] == sid and out[0]["name"] == "张三"
    assert store.speaker_count == 1                      # 命中不重复建档（防投毒）


def test_auto_enroll_serializes_identify_and_enroll(
    env, store, monkeypatch
):
    svc = make_service(store)
    original_identify = store.identify
    first_identify_entered = threading.Event()
    second_identify_entered = threading.Event()
    second_worker_started = threading.Event()
    release_first_identify = threading.Event()
    call_count = 0
    call_count_lock = threading.Lock()

    def blocking_identify(*args, **kwargs):
        nonlocal call_count
        with call_count_lock:
            call_count += 1
            current_call = call_count
        if current_call == 1:
            first_identify_entered.set()
            if not release_first_identify.wait(timeout=2):
                raise AssertionError("测试未放行首次识别")
        else:
            second_identify_entered.set()
        return original_identify(*args, **kwargs)

    results = {}

    def run(label):
        if label == "B":
            second_worker_started.set()
        results[label] = svc.map_and_enroll_clusters(
            [_cluster(label, unit(0), 12.0)]
        )[0]

    monkeypatch.setattr(store, "identify", blocking_identify)
    first_thread = threading.Thread(target=run, args=("A",))
    second_thread = threading.Thread(target=run, args=("B",))
    first_thread.start()
    assert first_identify_entered.wait(timeout=2)
    second_thread.start()
    assert second_worker_started.wait(timeout=2)
    try:
        assert not second_identify_entered.wait(timeout=0.2)
    finally:
        release_first_identify.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert not first_thread.is_alive() and not second_thread.is_alive()
    assert second_identify_entered.is_set()
    assert store.speaker_count == 1
    assert results["A"]["speaker_id"] == results["B"]["speaker_id"]
    assert sum("auto_enrolled" in result for result in results.values()) == 1


def test_auto_enroll_failure_falls_back_anonymous(env, store):
    svc = make_service(store)
    store.close()                                        # alloc/enroll 将失败
    out = svc.map_and_enroll_clusters([_cluster("A", unit(0), 12.0)])
    assert out[0]["speaker_id"] is None                  # 退回匿名，不抛错


def test_auto_enroll_stops_when_cancelled_after_identify(env, store, monkeypatch):
    svc = make_service(store)
    cancelled = False
    identify = store.identify

    def identify_then_cancel(*args, **kwargs):
        nonlocal cancelled
        result = identify(*args, **kwargs)
        cancelled = True
        return result

    monkeypatch.setattr(store, "identify", identify_then_cancel)
    out = svc.map_and_enroll_clusters(
        [_cluster("A", unit(0), 12.0)],
        cancelled=lambda: cancelled,
    )

    assert out == []
    assert store.speaker_count == 0
