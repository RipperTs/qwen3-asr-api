"""app/runtime/device.py 纯逻辑测试（行为依源码确认，见 device.py:6/27/52/61）。"""
import sys
import types

import pytest

from app.runtime import device


# ─── detect_device ───

def test_detect_device_cpu_when_cuda_unavailable(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    info = device.detect_device()
    assert info == {"type": "cpu", "vram_gb": None, "name": None}


def test_detect_device_cpu_when_torch_missing(monkeypatch):
    # 让 `import torch` 抛 ImportError（device.py 内部 try/except ImportError 兜底 CPU）
    monkeypatch.setitem(sys.modules, "torch", None)
    info = device.detect_device()
    assert info["type"] == "cpu"
    assert info["vram_gb"] is None


def test_detect_device_cuda_branch(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    props = types.SimpleNamespace(total_memory=8 * 1024 ** 3)  # 8 GB
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx: props)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "FakeGPU")
    info = device.detect_device()
    assert info == {"type": "cuda", "vram_gb": 8.0, "name": "FakeGPU"}


# ─── resolve_device ───

def test_resolve_device_cpu_forced():
    assert device.resolve_device("cpu", {"type": "cuda", "vram_gb": 8.0, "name": "x"}) == "cpu"


def test_resolve_device_cuda_ok():
    assert device.resolve_device("cuda", {"type": "cuda", "vram_gb": 8.0, "name": "x"}) == "cuda"


def test_resolve_device_cuda_unavailable_raises():
    with pytest.raises(RuntimeError):
        device.resolve_device("cuda", {"type": "cpu", "vram_gb": None, "name": None})


def test_resolve_device_auto_follows_hardware():
    assert device.resolve_device("auto", {"type": "cuda", "vram_gb": 8.0, "name": "x"}) == "cuda"
    assert device.resolve_device("auto", {"type": "cpu", "vram_gb": None, "name": None}) == "cpu"


def test_resolve_device_uses_detect_when_no_info(monkeypatch):
    monkeypatch.setattr(device, "detect_device", lambda: {"type": "cpu", "vram_gb": None, "name": None})
    assert device.resolve_device("auto") == "cpu"


# ─── auto_select_model_size ───

@pytest.mark.parametrize("vram,expected", [
    (None, "0.6b"),
    (6.0, "1.7b"),
    (8.0, "1.7b"),
    (5.9, "0.6b"),
    (4.0, "0.6b"),
])
def test_auto_select_model_size(vram, expected):
    assert device.auto_select_model_size(vram) == expected


# ─── should_disable_align ───

@pytest.mark.parametrize("dev,vram,expected", [
    ("cpu", 16.0, True),     # CPU 一律禁用对齐
    ("cpu", None, True),
    ("cuda", 3.9, True),     # 显存 < 4 禁用
    ("cuda", 3, True),
    ("cuda", 4.0, False),    # 显存 >= 4 允许
    ("cuda", None, False),   # 显存未知（cuda）不禁用
    ("cuda", 8.0, False),
])
def test_should_disable_align(dev, vram, expected):
    assert device.should_disable_align(dev, vram) is expected
