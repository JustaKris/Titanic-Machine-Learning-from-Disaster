"""
Custom exception handling for the Titanic ML project.
"""

import logging
from types import ModuleType


def error_message_detail(error: Exception, error_detail: ModuleType) -> str:
    """
    Construct a detailed error message with file name and line number.

    Args:
        error: The exception object.
        error_detail: The sys module to extract exception details.

    Returns:
        A formatted error message string.
    """
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "unknown"
        line_number = 0

    error_message = (
        f"Error occurred in python script [{file_name}] "
        f"line number [{line_number}] "
        f"error message [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    """Custom exception class with detailed error tracking."""

    def __init__(self, error_message: Exception, error_detail: ModuleType):
        """
        Initialize the CustomException with a detailed error message.

        Args:
            error_message: The error message or exception.
            error_detail: The sys module to extract exception details.
        """
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        logging.error(self.error_message)

    def __str__(self) -> str:
        """Return the detailed error message."""
        return self.error_message
