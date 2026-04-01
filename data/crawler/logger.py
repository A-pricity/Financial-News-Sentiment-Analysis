import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"crawler_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    crawler_logger = logging.getLogger("data.crawler")
    crawler_logger.setLevel(logging.DEBUG)

    return crawler_logger
