import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def configure_logging(log_dir: Path = Path(__file__).parents[2] / "logs") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.getLogger().setLevel(logging.DEBUG)

    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")

    # Create TimedRotatingFileHandler for all log messages
    rotating_handler = TimedRotatingFileHandler(
        log_dir / "debug.log", when="S", interval=300, backupCount=1, encoding="utf-8"
    )
    rotating_handler.setLevel(logging.DEBUG)
    rotating_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(rotating_handler)

    # Create TimedRotatingFileHandler for INFO and above
    info_handler = TimedRotatingFileHandler(log_dir / "info.log", when="H", interval=2, backupCount=7, encoding="utf-8")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(info_handler)

    # Make sure uncaught exceptions are logged
    sys.excepthook = lambda exctype, value, traceback: logging.error(
        "Uncaught exception:", exc_info=(exctype, value, traceback)
    )
