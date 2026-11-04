# OpenAI-Client

A standalone harness for the production **BaliRoam** OpenAI agent systems. It mirrors
the backend's OpenAI client, prompts, schemas, web search, streaming behavior, and
email templates so every OpenAI path can be exercised without touching the production
backend.

## Features

- **Interactive Chatbot** -- CLI assistant ("Made", the BaliRoam Concierge). Mirrors the
  production message construction, history mapping, Bali search augmentation, output
  humanization, prompt-leak detection, and SSE streaming.
- **Itinerary Generation** -- Builds the production itinerary prompt (dates, interests,
  budget), searches for current Bali information, cleans the Markdown, and emits the
  Markdown + WhatsApp variants.
- **Email Inquiry Analysis** -- Classifies inquiries with the production JSON schema,
  generates confirmation messages, and renders the branded analysis HTML email.
- **Web Search Augmentation** -- DeepSeek search first, then Exa, then Brave, with
  focused query planning and merged/cited DeepSeek results.
- **Production Completion/Streaming** -- High reasoning effort, thinking enabled via
  `extra_body`, retry with exponential backoff, circuit breaker, and explicit stream
  cleanup.
- **Multi-Provider Support** -- AtmoRouter, DeepSeek, Inception, OpenRouter, NVIDIA,
  and Mistral through OpenAI-compatible endpoints.
- **Email Templates** -- Responsive HTML builders plus plain-text fallbacks, all using
  the configured `LOGO_URL`.

## Project Structure

```text
src/openai_client/
|-- main.py                       # Entry point / provider selection
|-- client/
|   |-- openai_client.py          # Production completion + streaming client
|   `-- mistral_ai.py             # Mistral adapter with generate()
|-- configs/
|   |-- constants.py              # Shared constants (WhatsApp number, limits)
|   `-- settings.py               # Pydantic Settings (env vars)
|-- decorators/
|   |-- circuit_breaker.py        # Async circuit breaker + agent_cb
|   `-- with_retry.py             # Tenacity-based retry decorator
|-- errors/
|   `-- errors.py                 # Api/Agent error hierarchy + raise_ai_error
|-- schemas/ai/
|   |-- chatbot.py                # ChatPart/ChatMessage/ChatRequest/ChatResponse
|   |-- email.py                  # EmailInquiry, AnalysisFormat, EmailResponse
|   `-- itinerary.py              # ItineraryRequest (dates), responses
|-- services/
|   |-- chatbot.py                # Chat + search + streaming + injection gate
|   |-- itinerary.py              # Itinerary search + generation
|   |-- email_inquiry.py          # Inquiry analysis + confirmation
|   |-- web_search.py             # DeepSeek/Exa/Brave providers + SearchPlan
|   |-- prompts/
|   |   |-- prompts.py            # Production prompts and HTML email bodies
|   |   `-- humanizer.py          # Post-processing of AI writing patterns
|   `-- templates/
|       `-- email_template.py     # Responsive email builders + plain text
|-- typings/
|   `-- typings.py                # Message/stream type aliases
`-- utils/
    |-- async_utils.py            # Async stream cleanup
    |-- helpers.py                # Markdown/WhatsApp conversion + file output
    `-- query.py                  # Query fitting/truncation
```

## Requirements

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

## Installation

```bash
git clone <repository-url>
cd openai_client
uv sync
```

## Configuration

Create a `.env` file in the project root with the provider and search keys:

```env
# AI Provider Keys
INCEPTION_API_KEY=your_inception_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
NVIDIA_API_KEY=your_nvidia_api_key
MISTRAL_API_KEY=your_mistral_api_key
ATMOROUTER_API_KEY=your_atmorouter_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key

# Search Provider Keys (DeepSeek -> Exa -> Brave fallback)
EXA_API_KEY=your_exa_api_key
BRAVE_API_KEY=your_brave_api_key
```

Optional settings (with defaults):

| Variable | Default | Description |
| --- | --- | --- |
| `AI_REQUEST_TIMEOUT` | `60` | Request timeout in seconds |
| `AI_MAX_RETRIES` | `2` | Max retry attempts |
| `AI_RETRY_DELAY` | `1.0` | Base retry delay in seconds |
| `AI_BACKOFF_FACTOR` | `2.0` | Exponential backoff multiplier |
| `LOGO_URL` | BaliRoam asset URL | Logo used by all email templates |
| `COMPANY_TARGET_EMAIL` | `example@gmail.com` | Contact address in email templates |
| `TZ` | `Asia/Makassar` | Timezone used in email timestamps |
| `DEBUG` | `false` | When true, itineraries are saved to `itinerary.md`/`.txt` |

## Usage

```bash
uv run openai-client
```

The entry point (`main.py`) configures the provider and selects one of three features:

- `_openai_chat(client, method)` -- Interactive chatbot session
- `_openai_itinerary(client, method)` -- Generate a Bali travel itinerary
- `_openai_email_inquiry(client, method)` -- Analyze an inquiry and write `email_inquiry.html`

There is no runtime selection flag. Edit `provider = Prv.INCEPTION` in `main()` to choose the
provider, and change the `services[0]` index passed to `_openai_services` to select the feature
(`0` chat, `1` itinerary, `2` email).

Each feature supports two calling methods:

| Method | Description |
| --- | --- |
| `"completion"` | Standard non-streaming completion |
| `"stream"` | Chat emits SSE chunks; `Itinerary.run` and `EmailService.run` accept the value but always perform non-streaming completions. |

## AI Providers

| Provider | Base URL | Default Model |
| --- | --- | --- |
| AtmoRouter | `https://atmorouter.dev/v1` | `atmo/deepseek-v4.1-flash` |
| DeepSeek | `https://api.deepseek.com` | `deepseek-flash` |
| Inception | `https://api.inceptionlabs.ai/v1` | `mercury-2` |
| OpenRouter | `https://openrouter.ai/api/v1` | `stepfun/step-3.5-flash:free` |
| NVIDIA | `https://integrate.api.nvidia.com/v1` | `stepfun-ai/step-3.5-flash` |
| Mistral | `https://api.mistral.ai` | `mistral-large-latest` |

## Development

```bash
uv run pytest
ruff check src tests --fix
ruff format src tests
pyrefly check
```

Repository rules live in [`docs/rules/`](docs/rules/):

- [Python Refactoring Rules](docs/rules/python-refactoring-and-codebase-update.md)
- [Ruff And Pyrefly Check Fixes](docs/rules/ruff-pyrefly-fixes.md)

## Dependencies

| Package | Purpose |
| --- | --- |
| `openai` | Core OpenAI SDK |
| `mistralai` | Mistral provider SDK |
| `pydantic-settings` | Settings management from `.env` |
| `pydantic[email]` | Data validation and email validation |
| `rich` | Terminal formatting and logging |
| `tenacity` | Retry logic with exponential backoff |
| `httpx2` | HTTP client for search APIs |
| `exa-py` | Exa search API |
| `mistune` | AST-based Markdown → WhatsApp rendering |
| `mdformat` | Markdown formatting |
| `disposable-email-domains` | Disposable email domain blocklist |

## License

This project does not currently specify a license.