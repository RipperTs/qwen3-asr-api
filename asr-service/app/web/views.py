from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.web.page import HTML_PAGE, STREAM_PAGE

web_router = APIRouter()


@web_router.get("/web-ui", response_class=HTMLResponse)
async def web_ui():
    """返回 Web UI 单页应用（离线转写）"""
    return HTML_PAGE


@web_router.get("/web-ui/stream", response_class=HTMLResponse)
async def web_ui_stream():
    """返回实时语音转写测试页（麦克风 / 文件模拟推流）"""
    return STREAM_PAGE
