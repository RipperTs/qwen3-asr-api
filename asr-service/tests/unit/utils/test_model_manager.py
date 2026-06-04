"""app/utils/model_manager.py 测试。

snapshot_download 通过向 sys.modules 注入假 modelscope/huggingface_hub 模块拦截
（即使真实包已安装，sys.modules 覆盖优先），绝不触网。
行为依源码确认（model_manager.py:8/42）。
"""
import sys
import types
from unittest.mock import MagicMock

import pytest

from app.utils import model_manager as mm


def _inject_fake(name, monkeypatch):
    mod = types.ModuleType(name)
    mod.snapshot_download = MagicMock()
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


# ─── ensure_model ───

def test_ensure_model_skips_when_present(tmp_path, monkeypatch):
    d = tmp_path / "model"
    d.mkdir()
    (d / "weights.bin").write_bytes(b"x")
    fake = _inject_fake("modelscope", monkeypatch)
    monkeypatch.setattr(mm, "MODEL_SOURCE", "modelscope")

    mm.ensure_model("repo/id", str(d))
    fake.snapshot_download.assert_not_called()


def test_ensure_model_manual_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MODEL_SOURCE", "manual")
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        mm.ensure_model("repo/id", str(missing))


def test_ensure_model_modelscope_downloads(tmp_path, monkeypatch):
    d = tmp_path / "empty"
    d.mkdir()  # 存在但为空 -> 触发下载
    fake = _inject_fake("modelscope", monkeypatch)
    monkeypatch.setattr(mm, "MODEL_SOURCE", "modelscope")

    mm.ensure_model("repo/id", str(d))
    fake.snapshot_download.assert_called_once_with(model_id="repo/id", local_dir=str(d))


def test_ensure_model_huggingface_downloads(tmp_path, monkeypatch):
    d = tmp_path / "empty"
    d.mkdir()
    fake = _inject_fake("huggingface_hub", monkeypatch)
    monkeypatch.setattr(mm, "MODEL_SOURCE", "huggingface")

    mm.ensure_model("repo/id", str(d))
    fake.snapshot_download.assert_called_once_with(repo_id="repo/id", local_dir=str(d))


# ─── ensure_model_modelscope ───

def test_ensure_model_modelscope_skips_when_present(tmp_path, monkeypatch):
    d = tmp_path / "model"
    d.mkdir()
    (d / "f.bin").write_bytes(b"x")
    fake = _inject_fake("modelscope", monkeypatch)

    mm.ensure_model_modelscope("repo/id", str(d))
    fake.snapshot_download.assert_not_called()


def test_ensure_model_modelscope_manual_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MODEL_SOURCE", "manual")
    with pytest.raises(FileNotFoundError):
        mm.ensure_model_modelscope("repo/id", str(tmp_path / "nope"))


def test_ensure_model_modelscope_forces_modelscope(tmp_path, monkeypatch):
    d = tmp_path / "empty"
    d.mkdir()
    fake = _inject_fake("modelscope", monkeypatch)
    monkeypatch.setattr(mm, "MODEL_SOURCE", "huggingface")  # 仍强制走 modelscope

    mm.ensure_model_modelscope("repo/id", str(d))
    fake.snapshot_download.assert_called_once_with(model_id="repo/id", local_dir=str(d))
