"""app/web/views.py 路由冒烟测试（离线页 + 实时测试页）。

web_router 仅在 --web 时挂载；此处直接挂到 TestClient 验证返回与关键标记。
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.web.views import web_router


def _client():
    app = FastAPI()
    app.include_router(web_router)
    return TestClient(app)


def test_web_ui_offline_page():
    resp = _client().get("/web-ui")
    assert resp.status_code == 200
    html = resp.text
    assert "<!DOCTYPE html>" in html
    assert "/web-ui/stream" in html          # 导航指向实时页


def test_web_ui_stream_page():
    resp = _client().get("/web-ui/stream")
    assert resp.status_code == 200
    html = resp.text
    # 关键功能标记
    assert "/v2/asr/stream" in html          # 连接实时端点
    assert 'id="fileInput"' in html          # 文件模拟输入
    assert 'id="micStart"' in html           # 麦克风按钮
    assert 'id="transcript"' in html         # 结果区
    assert 'href="/web-ui"' in html          # 导航返回离线页


def test_stream_page_loaded_from_disk():
    # page.py 应已成功读入 stream.html（非空）
    from app.web.page import STREAM_PAGE
    assert STREAM_PAGE and len(STREAM_PAGE) > 500
