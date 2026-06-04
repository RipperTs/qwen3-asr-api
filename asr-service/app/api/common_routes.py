"""共性路由：health / capabilities。

两种 serve-mode（standard / vllm）都挂载，使 vllm 模式在不提供离线接口时
仍能通过 /health、/capabilities 暴露当前运行模式与能力。
"""
import logging
from fastapi import APIRouter, HTTPException
from app.api.schemas import HealthResponse, CapabilitiesResponse

logger = logging.getLogger(__name__)

# 运行时依赖，由 main.py 启动时注入
_service_info = None


def init_common(service_info: dict):
    """注入服务信息（供 /health、/capabilities 读取）"""
    global _service_info
    _service_info = service_info


async def health_check() -> HealthResponse:
    """健康检查，返回当前运行模式和加载的模型信息（mode-aware）"""
    if _service_info is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪，请稍后重试")
    return HealthResponse(**_service_info)


async def get_capabilities() -> CapabilitiesResponse:
    """返回当前服务能力（运行模式、是否提供离线 API、实时流能力）"""
    if _service_info is None or _service_info.get("capabilities") is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪，请稍后重试")
    return CapabilitiesResponse.model_validate(_service_info["capabilities"])


def build_common_router(prefix: str) -> APIRouter:
    """共性路由工厂；v1 与 v2 共用同一组实现。"""
    r = APIRouter(prefix=prefix)
    r.add_api_route("/health", health_check, methods=["GET"], response_model=HealthResponse)
    r.add_api_route("/capabilities", get_capabilities, methods=["GET"],
                    response_model=CapabilitiesResponse)
    return r
