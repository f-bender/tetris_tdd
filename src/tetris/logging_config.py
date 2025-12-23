import logging
import sys
import threading
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def configure_logging(log_dir: Path = Path(__file__).parents[2] / "logs") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    file_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s - %(pathname)s:%(lineno)d - %(threadName)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create TimedRotatingFileHandler for all log messages
    rotating_handler = TimedRotatingFileHandler(
        log_dir / "debug.log", when="S", interval=300, backupCount=1, encoding="utf-8"
    )
    rotating_handler.setLevel(logging.DEBUG)
    rotating_handler.setFormatter(file_formatter)

    # Create TimedRotatingFileHandler for INFO and above
    info_handler = TimedRotatingFileHandler(log_dir / "info.log", when="H", interval=2, backupCount=7, encoding="utf-8")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(file_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []
    root_logger.addHandler(info_handler)
    root_logger.addHandler(rotating_handler)

    # Make sure uncaught exceptions are logged
    sys.excepthook = lambda exctype, value, traceback: root_logger.error(
        "Uncaught exception:", exc_info=(exctype, value, traceback)
    )
    threading.excepthook = lambda args: root_logger.error(
        "Uncaught exception in thread '%s':",
        args.thread.name if args.thread else "unknown",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),  # type: ignore[arg-type]
    )
