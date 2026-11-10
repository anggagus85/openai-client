# tests/test_collect_models.py

"""Tests for the model catalogue collection orchestration in ``main``."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from openai_client import main as main_module
from openai_client.configs import ProviderConfig
from openai_client.configs import Providers as Prv


def _config(api_key: str | None = "test-key") -> ProviderConfig:
    return {
        "api_key": api_key,
        "base_url": "https://api.example.com/v1",
        "model": "test-model",
    }


class TestCollectModels:
    async def test_collects_each_provider_and_closes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        client = MagicMock()
        client.list_models = AsyncMock()
        client.close = AsyncMock()

        openai_cls = MagicMock(return_value=client)
        mistral_cls = MagicMock(return_value=client)

        monkeypatch.setattr(main_module, "OpenAIClient", openai_cls)
        monkeypatch.setattr(main_module, "MistralAI", mistral_cls)

        await main_module._collect_models(
            {
                Prv.ATMOROUTER: _config(),
                Prv.MISTRAL: _config(),
            },
        )

        assert client.list_models.await_count == 2
        assert client.close.await_count == 2

        # Mistral is constructed with keyword args so model/base_url cannot swap.
        assert mistral_cls.call_args.kwargs == {
            "api_key": "test-key",
            "model": "test-model",
            "base_url": "https://api.example.com/v1",
        }
        assert openai_cls.call_args.kwargs["prv"] == Prv.ATMOROUTER

        output = capsys.readouterr().out
        assert "Atmo Router" in output
        assert "Mistral" in output

    async def test_skips_provider_with_missing_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = MagicMock()
        client.list_models = AsyncMock()
        client.close = AsyncMock()

        monkeypatch.setattr(main_module, "OpenAIClient", MagicMock(return_value=client))

        await main_module._collect_models({Prv.ATMOROUTER: _config(api_key=None)})

        client.list_models.assert_not_awaited()
        client.close.assert_not_awaited()

    async def test_continues_after_provider_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        failing = MagicMock()
        failing.list_models = AsyncMock(side_effect=RuntimeError("boom"))
        failing.close = AsyncMock()

        good = MagicMock()
        good.list_models = AsyncMock()
        good.close = AsyncMock()

        monkeypatch.setattr(main_module, "OpenAIClient", MagicMock(side_effect=[failing, good]))

        await main_module._collect_models(
            {Prv.ATMOROUTER: _config(), Prv.DEEPSEEK: _config()},
        )

        assert failing.close.await_count == 1
        assert good.list_models.await_count == 1
        assert good.close.await_count == 1
