# AGENTS.md

## Developer Commands

```bash
uv sync                          # Install deps (first time / after pyproject.toml change)
uv add [package_name]          # Add a new package
uv remove [package_name]       # Remove a package
uv list                          # List all packages
pyrefly check             # Type check (NOT mypy)
uv run pytest                    # Run tests (coverage auto-enabled via pytest.ini)
openai-client                    # Run the app (entry point: src/openai_client/main.py:main)
```

Order: lint -> typecheck -> test.

## Architecture

- **Package root**: `src/openai_client/` (hatchling build, `uv` manages the venv).
- **Entry point**: `main.py` -- `main()` is the sole entry point. Three features exist; the active one is selected by uncommenting the corresponding call:
  - `_do_chat(client, method)` -- interactive chatbot
  - `_generate_itinerary(client, method)` -- Bali itinerary generation (currently active)
  - `_email_inquiry(client, method)` -- email analysis + HTML response
- **API call methods**: `"completion"` (standard Async OpenAI) and `"stream"` (streaming). Both work with any provider config.
- **Default provider**: Inception (`kwargs["inception"]` in `main.py`). Provider configs are hardcoded dictionaries in `main()` -- edit there to switch.
- **Core client**: `OpenAIClient` extends `openai.AsyncOpenAI`, adds `completion()` and `list_models()` methods. All API calls use the `@with_retry` decorator (tenacity-based, settings-driven).
- **Settings**: `pydantic-settings` loads from root `.env` and `.env.configs`. The API keys (`INCEPTION_API_KEY`, `OPENROUTER_API_KEY`, `NVIDIA_API_KEY`, `EXA_API_KEY`, `BRAVE_API_KEY`, `MISTRAL_API_KEY`, `FIREWORKS_API_KEY`, `ATMOROUTER_API_KEY`) are **required** -- missing any will raise a validation error on import. `DEEPSEEK_API_KEY` is optional and validated only when that provider is selected.

## Testing

- Tests live in `tests/`. Currently only `test_whatsapp_formatter.py` exists.
- `pytest.ini` auto-enables coverage (`--cov=openai_client`) with HTML + term reports.
- `pythonpath = src` in pytest.ini means imports resolve as `from openai_client...`.
- No integration test fixtures or service mocks exist yet.

## Conventions

- Ruff line-length: 100 (not the default 88).
- Ruff ignores `E501` (line length) and `D1` (missing docstrings) -- do not add docstrings to every public function unless asked.
- Ruff `unfixable = ["B"]` -- bugbear violations must be fixed manually.
- `pyrefly` is the type checker, not mypy.
- Generated HTML output (`email_inquiry.html`) is gitignored.
