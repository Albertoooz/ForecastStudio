"""
Error Recovery and Retry Logic for Agent System.

Professional error handling with retry, exponential backoff, and recovery strategies.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorSeverity(Enum):
    """Classification of error severity."""

    RECOVERABLE = "recoverable"  # Can retry
    TRANSIENT = "transient"  # Temporary issue (network, rate limit)
    FATAL = "fatal"  # Cannot recover


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_base: float = 2.0  # for exponential backoff
    jitter: bool = True  # add randomness to prevent thundering herd


class RecoverableError(Exception):
    """Error that can be recovered from by retrying."""

    pass


class TransientError(RecoverableError):
    """Temporary error (network, rate limit, etc.)."""

    pass


class FatalError(Exception):
    """Fatal error that cannot be recovered."""

    pass


def classify_error(error: Exception) -> ErrorSeverity:
    """
    Classify error severity.

    Args:
        error: The exception to classify

    Returns:
        ErrorSeverity classification
    """
    if isinstance(error, FatalError):
        return ErrorSeverity.FATAL

    if isinstance(error, TransientError):
        return ErrorSeverity.TRANSIENT

    # Check error message for common patterns
    error_msg = str(error).lower()

    # Network/rate limit errors
    if any(
        keyword in error_msg
        for keyword in ["timeout", "connection", "network", "rate limit", "429", "503"]
    ):
        return ErrorSeverity.TRANSIENT

    # Data validation errors - recoverable with fixes
    if any(
        keyword in error_msg
        for keyword in ["not found", "missing", "invalid format", "type mismatch"]
    ):
        return ErrorSeverity.RECOVERABLE

    # Likely fatal
    if any(
        keyword in error_msg
        for keyword in ["permission denied", "unauthorized", "forbidden", "401", "403"]
    ):
        return ErrorSeverity.FATAL

    # Default to recoverable
    return ErrorSeverity.RECOVERABLE


def retry_with_backoff[T](
    func: Callable[[], T],
    config: RetryConfig | None = None,
    operation_name: str = "operation",
) -> tuple[bool, T | None, str | None]:
    """
    Execute function with retry logic and exponential backoff.

    Args:
        func: Function to execute
        config: Retry configuration
        operation_name: Name for logging

    Returns:
        tuple: (success: bool, result: Optional[T], error: Optional[str])
    """
    if config is None:
        config = RetryConfig()

    last_error: Exception | None = None
    delay = config.initial_delay

    for attempt in range(config.max_retries + 1):
        try:
            result = func()

            if attempt > 0:
                logger.info(
                    f"✅ {operation_name} succeeded on attempt {attempt + 1}/{config.max_retries + 1}"
                )

            return (True, result, None)

        except Exception as e:
            last_error = e
            severity = classify_error(e)

            # Log error
            logger.warning(
                f"❌ {operation_name} failed (attempt {attempt + 1}/{config.max_retries + 1}): "
                f"{type(e).__name__}: {str(e)}"
            )

            # Check if we should retry
            if severity == ErrorSeverity.FATAL:
                logger.error(f"Fatal error in {operation_name}, cannot retry")
                break

            if attempt < config.max_retries:
                # Calculate backoff delay
                if config.jitter:
                    import random

                    jitter_factor = 0.5 + random.random()  # 0.5 to 1.5
                    actual_delay = min(delay * jitter_factor, config.max_delay)
                else:
                    actual_delay = min(delay, config.max_delay)

                logger.info(f"⏳ Retrying in {actual_delay:.1f}s...")
                time.sleep(actual_delay)

                # Exponential backoff
                delay *= config.exponential_base
            else:
                logger.error(f"Max retries ({config.max_retries}) exceeded for {operation_name}")

    return (False, None, str(last_error))


def with_error_recovery(
    operation_name: str,
    retry_config: RetryConfig | None = None,
):
    """
    Decorator for adding error recovery to functions.

    Usage:
        @with_error_recovery("create_column", retry_config=RetryConfig(max_retries=2))
        def create_column(...):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., dict[str, Any]]:
        def wrapper(*args, **kwargs) -> dict[str, Any]:
            success, result, error = retry_with_backoff(
                lambda: func(*args, **kwargs),
                config=retry_config,
                operation_name=operation_name,
            )

            return {
                "success": success,
                "result": result,
                "error": error,
            }

        return wrapper

    return decorator


def suggest_recovery_action(error: Exception) -> str:
    """
    Suggest recovery action based on error type.

    Args:
        error: The exception

    Returns:
        Human-readable suggestion
    """
    error_msg = str(error).lower()

    if "not found" in error_msg:
        if "column" in error_msg:
            return "💡 Check column names - they are case-sensitive. Use the Debug Info in sidebar to see available columns."
        return "💡 The requested item was not found. Please check the name and try again."

    if "missing" in error_msg or "null" in error_msg:
        return "💡 Consider filling missing values first using a data operation."

    if "type" in error_msg or "dtype" in error_msg:
        return "💡 Column type mismatch. Check if you're using numeric operations on text columns or vice versa."

    if "groupby" in error_msg:
        return (
            "💡 For grouped data, use expressions like: df.groupby(['store_id'])['sales'].shift(1)"
        )

    if "timeout" in error_msg or "network" in error_msg:
        return "💡 Network issue detected. The operation will retry automatically."

    if "rate limit" in error_msg or "429" in error_msg:
        return "💡 API rate limit reached. The system will wait and retry automatically."

    return "💡 An error occurred. Please check the error message and try again. Contact support if the issue persists."
