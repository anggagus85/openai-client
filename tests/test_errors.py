# tests/test_errors.py

"""Tests for custom error classes."""

import pytest

from openai_client.errors import ApiError, AppSystemError, UnknownError


class TestApiError:
    def test_inherits_from_exception(self) -> None:
        assert issubclass(ApiError, Exception)

    def test_can_be_raised(self) -> None:
        def _raise_error() -> None:
            details = "API failed"
            raise ApiError(details)

        with pytest.raises(ApiError) as excinfo:
            _raise_error()
        assert str(excinfo.value) == "API failed"

    def test_can_be_instantiated_without_message(self) -> None:
        err = ApiError()
        assert isinstance(err, ApiError)


class TestUnknownError:
    def test_inherits_from_exception(self) -> None:
        assert issubclass(UnknownError, Exception)

    def test_can_be_raised(self) -> None:
        def _raise_error() -> None:
            details = "Something unknown happened"
            raise UnknownError(details)

        with pytest.raises(UnknownError) as excinfo:
            _raise_error()
        assert str(excinfo.value) == "Something unknown happened"


class TestAppSystemError:
    def test_inherits_from_exception(self) -> None:
        assert issubclass(AppSystemError, Exception)

    def test_can_be_raised(self) -> None:
        def _raise_error() -> None:
            details = "System failure"
            raise AppSystemError(details)

        with pytest.raises(AppSystemError) as excinfo:
            _raise_error()
        assert str(excinfo.value) == "System failure"
