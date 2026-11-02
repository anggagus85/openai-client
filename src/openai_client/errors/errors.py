# src/openai_client/errors/errors.py

from httpx2 import ReadTimeout
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
    UnprocessableEntityError,
)

API_EXCEPTIONS = (
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    NotFoundError,
    ConflictError,
    UnprocessableEntityError,
    InternalServerError,
    LengthFinishReasonError,
    ContentFilterFinishReasonError,
    InvalidWebhookSignatureError,
    ReadTimeout,
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
