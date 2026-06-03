"""app/api/common_routes.py 测试（health / capabilities，mode-aware）。

验证共性路由在 /v1、/v2 两前缀下一致，且 health 反映 service_info 的 mode/capabilities。
"""
from unittest.mock import MagicMock

import pytest

SERVICE_INFO_STANDARD = {
    "status": "ready",
    "mode": "standard",
    "device": "cpu",
    "model_size": "0.6b",
    "align_enabled": False,
    "punc_enabled": True,
    "asr_backend": "qwen_asr",
    "vad_backend": "pytorch",
    "punc_backend": "pytorch",
    "capabilities": {
        "mode": "standard",
        "offline_api": True,
        "stream": {
            "enabled": False,
            "backend": None,
            "path": None,
            "partial_results": False,
            "word_timestamps": False,
        },
    },
}

SERVICE_INFO_VLLM = {
    "status": "ready",
    "mode": "vllm",
    "device": "cuda",
    "capabilities": {
        "mode": "vllm",
        "offline_api": False,
        "stream": {
            "enabled": False,
            "backend": "vllm-native",
            "path": None,
            "partial_results": False,
            "word_timestamps": False,
        },
    },
}


# ─── health ───

@pytest.mark.parametrize("prefix", ["/v1", "/v2"])
def test_health_reports_mode(make_client, prefix):
    client = make_client(service_info=SERVICE_INFO_STANDARD)
    resp = client.get(f"{prefix}/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "standard"
    assert body["device"] == "cpu"
    assert body["capabilities"]["offline_api"] is True


def test_health_v1_v2_identical(make_client):
    client = make_client(service_info=SERVICE_INFO_STANDARD)
    assert client.get("/v1/health").json() == client.get("/v2/health").json()


def test_health_vllm_mode(make_client):
    client = make_client(service_info=SERVICE_INFO_VLLM, include_offline=False)
    body = client.get("/v1/health").json()
    assert body["mode"] == "vllm"
    assert body["capabilities"]["offline_api"] is False
    # vllm 占位：不适用字段为 null
    assert body["model_size"] is None
    assert body["asr_backend"] is None


def test_health_not_ready_503(make_client):
    client = make_client(service_info=None)
    assert client.get("/v1/health").status_code == 503


# ─── capabilities ───

@pytest.mark.parametrize("prefix", ["/v1", "/v2"])
def test_capabilities(make_client, prefix):
    client = make_client(service_info=SERVICE_INFO_STANDARD)
    resp = client.get(f"{prefix}/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "standard"
    assert body["offline_api"] is True
    assert body["stream"]["enabled"] is False


def test_capabilities_v1_v2_identical(make_client):
    client = make_client(service_info=SERVICE_INFO_STANDARD)
    assert client.get("/v1/capabilities").json() == client.get("/v2/capabilities").json()


def test_capabilities_missing_returns_503(make_client):
    # service_info 缺 capabilities 键 -> 503
    info = {k: v for k, v in SERVICE_INFO_STANDARD.items() if k != "capabilities"}
    client = make_client(service_info=info)
    assert client.get("/v1/capabilities").status_code == 503
