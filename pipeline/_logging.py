"""统一日志配置：pipeline 内部用 logging，替代散落的 print。

configure_logging() 首次取 logger 时自动调用一次，给 "pipeline" logger 挂一个
StreamHandler（INFO 级，格式仅消息体，≈原来的 print 行为）。

- 环境变量 OIL_GAS_LOG_LEVEL 可覆盖级别（DEBUG/INFO/WARNING/ERROR）。
- 调用方也可自行 logging.basicConfig()；本模块只保证默认有输出、且不重复挂 handler。
"""
from __future__ import annotations

import logging
import os

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.environ.get("OIL_GAS_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger = logging.getLogger("pipeline")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"pipeline.{name}")
