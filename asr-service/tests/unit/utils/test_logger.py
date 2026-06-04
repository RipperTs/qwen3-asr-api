"""app/utils/logger.py::setup_logger 测试。

setup_logger 会清空并重置 root logger handler，测试中快照 root 状态并在结束后还原，
避免影响 pytest 自身的日志捕获。LOG_DIR/LOG_FILE 在 logger 模块命名空间内，
故 monkeypatch 模块级名字重定向到临时目录。
行为依源码确认（logger.py:6）。
"""
import logging

import pytest

from app.utils import logger as logger_mod


@pytest.fixture
def isolate_root_logger():
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    for h in root.handlers[:]:
        try:
            h.close()
        except Exception:
            pass
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def test_setup_logger_creates_dir_and_handlers(tmp_path, monkeypatch, isolate_root_logger):
    log_dir = tmp_path / "logs"
    log_file = log_dir / "asr.log"
    monkeypatch.setattr(logger_mod, "LOG_DIR", str(log_dir))
    monkeypatch.setattr(logger_mod, "LOG_FILE", str(log_file))

    logger_mod.setup_logger()

    root = logging.getLogger()
    assert log_dir.exists()
    assert root.level == logging.INFO
    handler_types = {type(h) for h in root.handlers}
    assert logging.FileHandler in handler_types
    assert any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers)

    access = logging.getLogger("uvicorn.access")
    assert access.level == logging.WARNING
    assert access.propagate is False
