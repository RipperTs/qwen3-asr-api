"""app/api/routes.py 路由测试（TestClient + init_routes 注入假 TaskManager）。

不启动真实服务/模型；上传目录已由 make_client 重定向到临时目录。
行为依源码确认（routes.py:17/43/50/106/116/139/145/181）。
"""
import queue
from unittest.mock import MagicMock

import pytest

SERVICE_INFO = {
    "status": "ready",
    "device": "cpu",
    "model_size": "0.6b",
    "align_enabled": False,
    "punc_enabled": True,
    "asr_backend": "qwen_asr",
    "vad_backend": "pytorch",
    "punc_backend": "pytorch",
}


# ─── submit_asr ───

def test_submit_ok(make_client):
    tm = MagicMock()
    tm.submit.return_value = "tid-1"
    client = make_client(task_manager=tm)
    resp = client.post(
        "/v1/asr",
        files={"file": ("a.wav", b"abcdef", "audio/wav")},
        data={"language": "zh"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"task_id": "tid-1"}
    tm.submit.assert_called_once()
    assert tm.submit.call_args.kwargs["language"] == "zh"


def test_submit_bad_extension(make_client):
    tm = MagicMock()
    client = make_client(task_manager=tm)
    resp = client.post("/v1/asr", files={"file": ("a.txt", b"abc", "text/plain")})
    assert resp.status_code == 400
    tm.submit.assert_not_called()


def test_submit_too_large(make_client, monkeypatch):
    monkeypatch.setattr("app.api.routes.MAX_AUDIO_FILE_SIZE", 0)  # max_bytes=0 -> 任意非空即超限
    tm = MagicMock()
    client = make_client(task_manager=tm)
    resp = client.post("/v1/asr", files={"file": ("a.wav", b"abc", "audio/wav")})
    assert resp.status_code == 413
    tm.submit.assert_not_called()


def test_submit_queue_full(make_client):
    tm = MagicMock()
    tm.submit.side_effect = queue.Full()
    client = make_client(task_manager=tm)
    resp = client.post("/v1/asr", files={"file": ("a.wav", b"abc", "audio/wav")})
    assert resp.status_code == 503


def test_submit_not_ready(make_client):
    client = make_client(task_manager=None)
    resp = client.post("/v1/asr", files={"file": ("a.wav", b"abc", "audio/wav")})
    assert resp.status_code == 503


@pytest.mark.parametrize("method,path", [
    ("get", "/v1/tasks"),
    ("get", "/v1/tasks/t1"),
    ("delete", "/v1/tasks/t1"),
])
def test_endpoints_not_ready_return_503(make_client, method, path):
    client = make_client(task_manager=None)
    resp = getattr(client, method)(path)
    assert resp.status_code == 503


# ─── list_tasks ───

def test_list_tasks(make_client):
    tm = MagicMock()
    tm.list_tasks.return_value = [
        {"task_id": "t1", "status": "completed", "progress": 1.0,
         "language": "zh", "created_at": "2026-06-03T10:00:00",
         "finished_at": "2026-06-03T10:01:00", "error": None},
    ]
    client = make_client(task_manager=tm)
    resp = client.get("/v1/tasks")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["tasks"][0]["task_id"] == "t1"


def test_list_tasks_status_filter(make_client):
    tm = MagicMock()
    tm.list_tasks.return_value = []
    client = make_client(task_manager=tm)
    resp = client.get("/v1/tasks", params={"status": "pending"})
    assert resp.status_code == 200
    tm.list_tasks.assert_called_once_with(status="pending")


# ─── get_task_detail / get_task_status(deprecated) ───

def test_get_task_detail_found(make_client):
    tm = MagicMock()
    tm.get_task.return_value = {
        "task_id": "t1", "status": "completed", "progress": 1.0,
        "result": {"full_text": "hi"}, "error": None,
    }
    client = make_client(task_manager=tm)
    resp = client.get("/v1/tasks/t1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == "t1"
    assert body["status"] == "completed"
    assert body["result"] == {"full_text": "hi"}


def test_get_task_detail_not_found(make_client):
    tm = MagicMock()
    tm.get_task.return_value = None
    client = make_client(task_manager=tm)
    resp = client.get("/v1/tasks/unknown")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "not_found"
    assert body["progress"] == 0.0


def test_deprecated_asr_task_status_delegates(make_client):
    tm = MagicMock()
    tm.get_task.return_value = {
        "task_id": "t1", "status": "processing", "progress": 0.5,
        "result": None, "error": None,
    }
    client = make_client(task_manager=tm)
    resp = client.get("/v1/asr/t1")
    assert resp.status_code == 200
    assert resp.json()["status"] == "processing"


# ─── cancel_asr ───

@pytest.mark.parametrize("prev,expect_status", [
    (None, "not_found"),
    ("pending", "cancelled"),
    ("processing", "cancelled"),
    ("completed", "already_completed"),
    ("failed", "already_failed"),
])
def test_cancel(make_client, prev, expect_status):
    tm = MagicMock()
    tm.cancel_task.return_value = prev
    client = make_client(task_manager=tm)
    resp = client.delete("/v1/tasks/t1")
    assert resp.status_code == 200
    assert resp.json()["status"] == expect_status


# ─── health_check ───

def test_health_ok(make_client):
    client = make_client(task_manager=MagicMock(), service_info=SERVICE_INFO)
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["device"] == "cpu"


def test_health_not_ready(make_client):
    client = make_client(task_manager=MagicMock(), service_info=None)
    resp = client.get("/v1/health")
    assert resp.status_code == 503


# ─── 鉴权 verify_api_key ───

def test_auth_rejects_without_token(make_client, monkeypatch):
    monkeypatch.setattr("app.config.API_KEY", "secret")
    tm = MagicMock()
    tm.list_tasks.return_value = []
    client = make_client(task_manager=tm)
    resp = client.get("/v1/tasks")
    assert resp.status_code == 401


def test_auth_accepts_valid_token(make_client, monkeypatch):
    monkeypatch.setattr("app.config.API_KEY", "secret")
    tm = MagicMock()
    tm.list_tasks.return_value = []
    client = make_client(task_manager=tm)
    resp = client.get("/v1/tasks", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200


def test_auth_disabled_when_no_key(make_client, monkeypatch):
    monkeypatch.setattr("app.config.API_KEY", "")
    tm = MagicMock()
    tm.list_tasks.return_value = []
    client = make_client(task_manager=tm)
    resp = client.get("/v1/tasks")
    assert resp.status_code == 200


# ─── T02: /v2 同名路径复用 v1 控制器（契约一致性）───

def test_v2_submit_alias(make_client):
    tm = MagicMock()
    tm.submit.return_value = "tid-v2"
    client = make_client(task_manager=tm)
    resp = client.post("/v2/asr", files={"file": ("a.wav", b"abc", "audio/wav")})
    assert resp.status_code == 200
    assert resp.json() == {"task_id": "tid-v2"}


def test_v2_list_tasks_alias(make_client):
    tm = MagicMock()
    tm.list_tasks.return_value = []
    client = make_client(task_manager=tm)
    assert client.get("/v2/tasks").status_code == 200


def test_v2_task_detail_and_cancel_alias(make_client):
    tm = MagicMock()
    tm.get_task.return_value = {"task_id": "t1", "status": "completed", "progress": 1.0,
                               "result": None, "error": None}
    tm.cancel_task.return_value = "pending"
    client = make_client(task_manager=tm)
    assert client.get("/v2/tasks/t1").status_code == 200
    assert client.delete("/v2/tasks/t1").status_code == 200


def test_v2_has_no_deprecated_asr_task_route(make_client):
    # deprecated GET /asr/{id} 仅 v1 保留，v2 不注册 -> 404
    tm = MagicMock()
    client = make_client(task_manager=tm)
    assert client.get("/v2/asr/t1").status_code == 404
    # v1 仍保留 deprecated 别名
    tm.get_task.return_value = None
    assert client.get("/v1/asr/t1").status_code == 200
