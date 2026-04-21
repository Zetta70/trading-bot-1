"""
Advanced rotating log setup.

Creates three separate log streams:
  logs/system.log  — INFO+  (general operation)
  logs/trade.log   — trade events only (via "trade" logger)
  logs/error.log   — ERROR+ (failures and critical alerts)

Each file rotates at 10 MB with 5 backups.
Console output mirrors system.log.
"""

import os
import logging
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
MAX_BYTES = 10 * 1024 * 1024   # 10 MB
BACKUP_COUNT = 5

FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """Configure all log handlers. Call once at startup."""
    os.makedirs(LOG_DIR, exist_ok=True)

    formatter = logging.Formatter(FMT, datefmt=DATEFMT)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # ── Console → INFO+ ─────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    # ── system.log → INFO+ ──────────────────────────────────────────
    sys_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "system.log"),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
    )
    sys_handler.setLevel(logging.INFO)
    sys_handler.setFormatter(formatter)
    root.addHandler(sys_handler)

    # ── error.log → ERROR+ ──────────────────────────────────────────
    err_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "error.log"),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
    )
    err_handler.setLevel(logging.ERROR)
    err_handler.setFormatter(formatter)
    root.addHandler(err_handler)

    # ── trade.log → dedicated logger ────────────────────────────────
    trade_logger = logging.getLogger("trade")
    trade_logger.setLevel(logging.INFO)
    trade_logger.propagate = False  # Don't duplicate to root

    trade_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "trade.log"),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
    )
    trade_handler.setFormatter(formatter)
    trade_logger.addHandler(trade_handler)

    # Also echo trades to console for visibility
    trade_console = logging.StreamHandler()
    trade_console.setLevel(logging.INFO)
    trade_console.setFormatter(formatter)
    trade_logger.addHandler(trade_console)
