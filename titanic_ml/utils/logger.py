"""
Production-ready logging configuration for Azure deployment.

Features:
- JSON formatting for cloud platforms (Azure, CloudWatch)
- Colored formatting for local development
- Environment auto-detection
- Third-party logger suppression
- Extra fields support for structured logging
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging compatible with Azure Application Insights.

    Produces log entries in JSON format that can be easily ingested by cloud logging services.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted string containing all log information.
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add any extra fields passed via the 'extra' parameter
        if hasattr(record, "extra_fields"):
            extra_fields = getattr(record, "extra_fields")
            if isinstance(extra_fields, dict):
                log_data.update(extra_fields)

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter for local development.

    Provides colored output for better readability during local development and debugging.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with ANSI color codes.

        Args:
            record: The log record to format.

        Returns:
            Colored formatted string.
        """
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def configure_logging(
    level: str = "INFO", use_json: Optional[bool] = None, include_uvicorn: bool = False
) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO.
        use_json: Whether to use JSON formatting. If None, auto-detects based on environment.
                 Set to True for production/cloud deployments, False for local development.
        include_uvicorn: Whether to include uvicorn logs. Default: False.

    Environment Detection:
        - Auto-enables JSON logging if AZURE_FUNCTIONS_ENVIRONMENT, AWS_EXECUTION_ENV,
          or KUBERNETES_SERVICE_HOST environment variables are set.
        - Falls back to colored console logging for local development.

    Example:
        >>> configure_logging(level="DEBUG", use_json=False)  # Local development
        >>> configure_logging(level="INFO", use_json=True)    # Production/Azure
        >>> configure_logging()  # Auto-detect environment
    """
    # Auto-detect cloud environment if not specified
    if use_json is None:
        use_json = any(
            [
                os.getenv("AZURE_FUNCTIONS_ENVIRONMENT"),
                os.getenv("AWS_EXECUTION_ENV"),
                os.getenv("KUBERNETES_SERVICE_HOST"),
                os.getenv("WEBSITE_INSTANCE_ID"),  # Azure App Service
            ]
        )

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    if use_json:
        # JSON formatter for production
        console_handler.setFormatter(JsonFormatter())
    else:
        # Colored formatter for local development
        console_formatter = ColoredFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)

    root_logger.addHandler(console_handler)

    # File handler (always uses detailed format)
    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file

    if use_json:
        file_handler.setFormatter(JsonFormatter())
    else:
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)

    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    suppress_loggers = [
        "chromadb",
        "transformers",
        "torch",
        "tensorflow",
        "httpx",
        "httpcore",
        "urllib3",
        "werkzeug",
        "matplotlib",
        "PIL",
    ]

    for logger_name in suppress_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Configure uvicorn loggers if requested
    if include_uvicorn:
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    else:
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # Log initialization message
    env_type = "PRODUCTION (JSON)" if use_json else "DEVELOPMENT (Colored)"
    root_logger.info(f"Logging configured: {env_type} mode at {level} level")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Name of the logger (typically __name__ from the calling module).

    Returns:
        Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
        >>> logger.error("An error occurred", extra={'extra_fields': {'user_id': 123}})
    """
    return logging.getLogger(name)


# Initialize logging on module import (can be reconfigured later)
if not logging.getLogger().handlers:
    configure_logging()
