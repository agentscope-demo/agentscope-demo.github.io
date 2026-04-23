from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """
    Configure root logger for the entire project.
    Call once from run_all.py or api/app.py at startup.

    Args:
        level:    log level string — DEBUG / INFO / WARNING / ERROR
        log_file: optional path to write logs to disk in addition to stdout
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )

    # quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "autogen"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging initialised — level=%s file=%s", level, log_file
    )