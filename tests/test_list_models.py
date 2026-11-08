# tests/test_list_models.py

"""Tests for the OpenAI client's model catalogue persistence."""

from json import loads
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from openai_client.client.openai_client import OpenAIClient
from openai_client.configs import Providers as Prv

CATALOGUE = "models.json"


def _client(prv: Prv) -> OpenAIClient:
    return OpenAIClient(
        api_key="test-key",
        base_url="https://api.example.com/v1",
        model="test-model",
        prv=prv,
    )


def _listing(client: OpenAIClient, *model_ids: str) -> None:
    models = MagicMock()
    models.list = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(id=i) for i in model_ids]),
    )
    object.__setattr__(client, "models", models)


def _read(path: Path) -> dict[str, list[str]]:
    return loads(path.read_text(encoding="utf-8"))


class TestProviderDisplayName:
    @pytest.mark.parametrize(
        ("prv", "expected"),
        [
            (Prv.ATMOROUTER, "Atmo Router"),
            (Prv.DEEPSEEK, "Deepseek"),
            (Prv.INCEPTION, "Inception"),
            (Prv.OPENROUTER, "Open Router"),
            (Prv.NVIDIA, "NVIDIA"),
            (Prv.MISTRAL, "Mistral"),
        ],
    )
    def test_reports_readable_name(self, prv: Prv, expected: str) -> None:
        assert _client(prv)._provider_name() == expected


class TestListModelsPersistence:
    async def test_writes_filtered_catalogue_under_provider_name(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        client = _client(Prv.INCEPTION)
        _listing(client, "mercury-3", "mercury-coder-small", "mercury-it", "text-embed-3")

        await client.list_models()

        assert _read(tmp_path / CATALOGUE) == {"inception": ["mercury-3"]}

    async def test_creates_catalogue_when_absent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        client = _client(Prv.NVIDIA)
        _listing(client, "nemotron-4")

        await client.list_models()

        assert _read(tmp_path / CATALOGUE) == {"nvidia": ["nemotron-4"]}

    async def test_preserves_other_providers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / CATALOGUE).write_text('{"inception": ["mercury-2"]}', encoding="utf-8")
        client = _client(Prv.DEEPSEEK)
        _listing(client, "deepseek-v4-pro")

        await client.list_models()

        assert _read(tmp_path / CATALOGUE) == {
            "inception": ["mercury-2"],
            "deepseek": ["deepseek-v4-pro"],
        }

    async def test_replaces_only_its_own_previous_entry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / CATALOGUE).write_text(
            '{"inception": ["mercury-1"], "nvidia": ["nemotron-3"]}',
            encoding="utf-8",
        )
        client = _client(Prv.INCEPTION)
        _listing(client, "mercury-3")

        await client.list_models()

        assert _read(tmp_path / CATALOGUE) == {
            "inception": ["mercury-3"],
            "nvidia": ["nemotron-3"],
        }

    async def test_openrouter_keeps_only_free_tier(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        client = _client(Prv.OPENROUTER)
        _listing(client, "meta/llama:free", "meta/llama", "zai/glm-5.3:free")

        await client.list_models()

        expected = {"openrouter": ["meta/llama:free", "zai/glm-5.3:free"]}
        assert _read(tmp_path / CATALOGUE) == expected

    async def test_announces_where_models_were_collected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.chdir(tmp_path)
        client = _client(Prv.ATMOROUTER)
        _listing(client, "ag/gemini-4-argon")

        await client.list_models()

        output = capsys.readouterr().out
        assert "Atmo Router" in output
        assert CATALOGUE in output
