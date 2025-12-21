import logging
import sys
from pathlib import Path
from datetime import datetime
from functools import cached_property, lru_cache
from logging.handlers import TimedRotatingFileHandler

def setup_logging(log_level: str = "INFO", log_to_file: bool = True, file_name: str = "app") -> None:
    """Configure logging for the application."""
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path.cwd() / f"logs/{file_name}_{timestamp}.log"
        log_file.parent.mkdir(exist_ok=True)
        
        file_handler = TimedRotatingFileHandler(
            filename=str(log_file),
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8",)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


@lru_cache
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin to add logging to classes."""
    
    @cached_property
    def logger(self) -> logging.Logger:
        return get_logger(self.__class__.__name__)