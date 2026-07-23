"""
Logging setup for variations experiments.

Usage:
    from lib.logging_utils import setup_logging
    setup_logging("exp7")   # logs to logs/exp7.log + stderr
"""

import logging
import sys
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"


def setup_logging(exp_name: str) -> None:
    """Configure root logger to write to stderr and logs/<exp_name>.log."""
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"{exp_name}.log"

    fmt = "%(asctime)s %(levelname)s %(message)s"
    handlers = [
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(log_path, mode="a"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    logging.info(f"Logging to {log_path}")
