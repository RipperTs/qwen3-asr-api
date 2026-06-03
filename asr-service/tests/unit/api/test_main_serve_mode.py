"""app/main.py create_app 的 serve-mode 装配测试。

仅测 vllm 占位分支（不加载任何模型）：验证 T03 验收——vllm 模式仅挂共性接口、
mode 正确、不误挂离线接口。standard 分支需加载真实模型，留待 T10 集成测试。
隔离 setup_logger / cfg 全局副作用。
"""
import logging
import types

import pytest
from fastapi.testclient import TestClient


def _args(**over):
    base = dict(
        serve_mode="standard", device="cpu", model_size=None, enable_align=True,
        enable_punc=False, model_source="modelscope", host=None, port=None,
        web=False, max_segment=5, api_key=None, max_queue_size=None,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


@pytest.fixture
def isolated_create_app(tmp_path, monkeypatch):
    """隔离 create_app 的全局副作用：日志目录改到临时路径、root logger 与 cfg 可变项还原。"""
    import app.config as cfg
    from app.utils import logger as logger_mod

    monkeypatch.setattr(logger_mod, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(logger_mod, "LOG_FILE", str(tmp_path / "logs" / "asr.log"))

    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    keys = ("MODEL_SOURCE", "MAX_SEGMENT_DURATION", "HOST", "PORT", "API_KEY", "MAX_QUEUE_SIZE")
    snapshot = {k: getattr(cfg, k) for k in keys}

    yield

    for h in root.handlers[:]:
        try:
            h.close()
        except Exception:
            pass
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    for k, v in snapshot.items():
        setattr(cfg, k, v)


def test_vllm_mode_mounts_common_only(isolated_create_app):
    from app.main import create_app

    app = create_app(_args(serve_mode="vllm", device="cpu"))
    client = TestClient(app)

    # health 反映 vllm 模式
    health = client.get("/v1/health").json()
    assert health["mode"] == "vllm"
    assert health["capabilities"]["offline_api"] is False
    assert health["capabilities"]["stream"]["backend"] == "vllm-native"

    # capabilities 在 v1/v2 都可用
    assert client.get("/v1/capabilities").json()["mode"] == "vllm"
    assert client.get("/v2/capabilities").json()["mode"] == "vllm"

    # vllm 模式不挂离线接口
    assert client.get("/v1/tasks").status_code == 404
    assert client.get("/v2/tasks").status_code == 404
    assert client.post("/v1/asr", files={"file": ("a.wav", b"x", "audio/wav")}).status_code == 404
