# src/openai_client/errors/errors.py

from openai import (
    AuthenticationError,
    BadRequestError,
    ConflictError,
    ContentFilterFinishReasonError,
    InternalServerError,
    InvalidWebhookSignatureError,
    LengthFinishReasonError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

API_EXCEPTIONS = (
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    NotFoundError,
    ConflictError,
    UnprocessableEntityError,
    RateLimitError,
    InternalServerError,
    LengthFinishReasonError,
    ContentFilterFinishReasonError,
    InvalidWebhookSignatureError,
)

SYSTEM_EXCEPTIONS = (
    SystemExit,
    EOFError,
    OSError,
    PermissionError,
    ConnectionError,
    TimeoutError,
    FileNotFoundError,
)


class ApiError(Exception):
    pass


class UnknownError(Exception):
    pass


class AppSystemError(Exception):
    pass
