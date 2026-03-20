"""
utils/logger.py
===============
日志系统。

优化内容 (Phase 2):
    1. 结构化 JSON 日志文件：每条日志以 JSON Lines 格式写入文件，
       便于 Dashboard 解析和 ELK 等日志平台接入
    2. 按天轮转：日志文件按日期自动轮转，保留最近 30 天，
       防止日志文件无限增长
    3. 控制台彩色输出：不同级别的日志使用不同颜色，提升可读性
    4. 统一的 get_logger() 工厂函数：所有模块通过此函数获取 logger，
       确保日志配置的一致性

日志文件结构:
    logs/
    ├── trading.log              # 当天的纯文本日志（按天轮转）
    ├── trading.log.2026-03-19   # 历史轮转日志
    └── trading_json.log         # 当天的 JSON Lines 结构化日志（供 Dashboard 使用）
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

# ============================================================
# ANSI 颜色代码
# ============================================================
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_YELLOW = "\033[93m"


# ============================================================
# 全局状态
# ============================================================
_LOG_DIR: Path = Path(os.environ.get("LOG_DIR", "logs"))
_LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
_initialized: bool = False
_ROOT_LOGGER_NAME: str = "lighter"


# ============================================================
# 自定义 Formatter
# ============================================================

class ColoredConsoleFormatter(logging.Formatter):
    """
    控制台彩色日志格式化器。

    根据日志级别为输出着色，提升终端可读性。
    格式: HH:MM:SS [LEVEL   ] [模块名] 消息内容
    """

    LEVEL_COLORS = {
        logging.DEBUG: Colors.GRAY,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.BRIGHT_YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BRIGHT_RED + Colors.BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)
        time_str = self.formatTime(record, "%H:%M:%S")
        level_str = f"{color}{record.levelname:<8}{Colors.RESET}"
        # 取 logger 名称最后一段（如 "lighter.risk_manager" -> "risk_manager"）
        name = record.name.split(".")[-1]
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return (
            f"{Colors.GRAY}{time_str}{Colors.RESET} "
            f"[{level_str}] "
            f"{Colors.CYAN}{name:<20}{Colors.RESET} "
            f"{msg}"
        )


class JsonLinesFormatter(logging.Formatter):
    """
    JSON Lines 结构化日志格式化器。

    每条日志输出为一行 JSON，包含时间戳、级别、模块名、消息等字段。
    便于 Dashboard WebSocket 服务实时解析和推送。

    输出格式示例:
        {"ts": 1742400000.123, "level": "INFO", "logger": "risk_manager",
         "msg": "[风控] 余额充足 $8000", "exc": null}
    """

    def format(self, record: logging.LogRecord) -> str:
        exc_text = None
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)

        entry = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name.split(".")[-1],
            "msg": record.getMessage(),
            "exc": exc_text,
        }
        return json.dumps(entry, ensure_ascii=False)


# ============================================================
# 初始化函数
# ============================================================

def setup_logging(
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    console: bool = True,
    json_file: bool = True,
    text_file: bool = True,
    backup_count: int = 30,
) -> None:
    """
    初始化日志系统。

    应在程序启动时调用一次。之后所有模块通过 get_logger() 获取 logger。

    Args:
        log_dir: 日志目录路径，默认为 ./logs
        log_level: 日志级别，默认为 INFO
        console: 是否输出到控制台（带颜色），默认 True
        json_file: 是否写入 JSON Lines 日志文件，默认 True
        text_file: 是否写入纯文本日志文件（按天轮转），默认 True
        backup_count: 保留的历史日志文件数量，默认 30 天
    """
    global _initialized, _LOG_DIR, _LOG_LEVEL

    if _initialized:
        return

    _LOG_LEVEL = log_level.upper()
    _LOG_DIR = Path(log_dir) if log_dir else _LOG_DIR
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, _LOG_LEVEL, logging.INFO)

    # 获取根 logger（lighter 命名空间）
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(logging.DEBUG)  # 根 logger 接受所有级别，由 handler 过滤
    root.handlers.clear()

    # ---- 控制台 Handler（彩色）----
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(numeric_level)
        ch.setFormatter(ColoredConsoleFormatter())
        root.addHandler(ch)

    # ---- 纯文本文件 Handler（按天轮转）----
    if text_file:
        text_path = _LOG_DIR / "trading.log"
        fh = logging.handlers.TimedRotatingFileHandler(
            filename=str(text_path),
            when="midnight",
            interval=1,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        text_fmt = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(text_fmt)
        root.addHandler(fh)

    # ---- JSON Lines 文件 Handler（按天轮转）----
    if json_file:
        json_path = _LOG_DIR / "trading_json.log"
        jh = logging.handlers.TimedRotatingFileHandler(
            filename=str(json_path),
            when="midnight",
            interval=1,
            backupCount=backup_count,
            encoding="utf-8",
        )
        jh.setLevel(logging.DEBUG)
        jh.setFormatter(JsonLinesFormatter())
        root.addHandler(jh)

    # 降低第三方库的日志级别（避免刷屏）
    for lib in ("openai", "httpx", "httpcore", "urllib3", "websockets", "aiohttp"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    # 防止日志向上传播到 Python 根 logger（避免重复输出）
    root.propagate = False

    _initialized = True
    root.info(
        f"[日志系统] 已初始化 | 级别: {_LOG_LEVEL} | "
        f"目录: {_LOG_DIR.resolve()} | "
        f"JSON日志: {'开启' if json_file else '关闭'} | "
        f"轮转保留: {backup_count} 天"
    )


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的 logger。

    所有模块应通过此函数获取 logger，而非直接使用 logging.getLogger()。
    这样可以确保所有 logger 都在 lighter 命名空间下，
    并继承统一的日志配置。

    Args:
        name: logger 名称（通常为模块名，如 "risk_manager"）

    Returns:
        配置好的 Logger 实例
    """
    if not _initialized:
        setup_logging()
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def get_json_log_path() -> Path:
    """
    获取当前 JSON 日志文件的路径。

    供 Dashboard WebSocket 服务读取日志时使用。

    Returns:
        JSON 日志文件的绝对路径
    """
    return (_LOG_DIR / "trading_json.log").resolve()


def get_text_log_path() -> Path:
    """
    获取当前纯文本日志文件的路径。

    Returns:
        纯文本日志文件的绝对路径
    """
    return (_LOG_DIR / "trading.log").resolve()
