# OpenAI-Client

A Bali tourism AI assistant built on top of the OpenAI Python SDK. Provides an interactive chatbot, AI-powered itinerary generation, and email inquiry analysis with HTML response generation -- all tailored for **BaliBlissed**, a Bali travel concierge service.

## Features

- **Interactive Chatbot** -- A conversational CLI assistant ("Made", the BaliBlissed Concierge) specializing in Bali travel advice. Augments responses with real-time web search results for tourism-related queries.
- **Itinerary Generation** -- Generates detailed multi-day Bali trip plans in Markdown format, including daily breakdowns, accommodations, dining, cultural experiences, and budget guidance.
- **Email Inquiry Analysis** -- Analyzes incoming email inquiries using AI, classifies them by category and urgency, and generates responsive HTML emails with AI-suggested replies.
- **Web Search Augmentation** -- Integrates Exa and Brave search APIs to provide real-time, up-to-date information in chatbot and itinerary responses.
- **Multi-Provider Support** -- Works with multiple AI providers through OpenAI-compatible endpoints: OpenRouter, Inception (Mercury), and NVIDIA.

## Project Structure

```text
src/openai_client/
|-- main.py                  # Entry point
|-- client/
|   `-- openai_client.py     # Core client extending openai.AsyncOpenAI
|-- configs/
|   `-- settings.py          # Pydantic Settings (env vars)
|-- decorators/
|   `-- with_retry.py        # Tenacity-based retry decorator
|-- errors/
|   `-- errors.py            # Custom exception classes
|-- schemas/ai/
|   |-- chatbot.py           # Chat message and request schemas
|   |-- email.py             # Email inquiry and analysis schemas
|   `-- itinerary.py         # Itinerary request and response schemas
|-- services/
|   |-- chatbot.py           # Interactive chatbot with search
|   |-- itinerary.py         # Itinerary generation service
|   |-- email_inquiry.py     # Email analysis and HTML generation
|   |-- search.py            # Exa and Brave search clients
|   `-- caller/
|       `-- caller.py        # Unified API caller (completion/stream)
|-- types/
|   `-- types.py             # Type aliases
`-- utils/
    `-- helpers.py           # File I/O utilities
```

## Requirements

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd openai_client

# Install dependencies with uv
uv sync
```

## Configuration

Create a `.env` file in the project root with the following API keys:

```env
# AI Provider Keys (at least one required)
INCEPTION_API_KEY=your_inception_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
NVIDIA_API_KEY=your_nvidia_api_key

# Search Provider Keys (for web-augmented responses)
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

## Usage

Run the CLI application:

```bash
openai-client
```

The entry point (`main.py`) is configured to launch one of three features. Edit `main.py` to switch between them:

- `_do_chat(client, method)` -- Interactive chatbot session (default)
- `_generate_itinerary(client, method)` -- Generate a Bali travel itinerary
- `_email_inquiry(client, method)` -- Analyze an email inquiry and generate an HTML response

### API Call Methods

Each feature supports two calling methods:

| Method | Description |
| --- | --- |
| `"completion"` | Standard Async OpenAI chat completion |
| `"stream"` | Streaming response with real-time token output |

## AI Providers

The client wraps the OpenAI SDK and connects to multiple providers via OpenAI-compatible endpoints:

| Provider | Base URL | Default Model |
| --- | --- | --- |
| Inception | `https://api.inceptionlabs.ai/v1` | `mercury-coder-small-beta` |
| OpenRouter | `https://openrouter.ai/api/v1` | `nvidia/nemotron-nano-9b-v2:free` |
| NVIDIA | `https://integrate.api.nvidia.com/v1` | `mistralai/mistral-large-2-instruct` |

## Development

### Linting

```bash
uv run ruff check .
uv run ruff format .
```

### Type Checking

```bash
uv run pyrefly check
```

## Dependencies

| Package | Purpose |
| --- | --- |
| `openai` | Core OpenAI SDK |
| `pydantic-settings` | Settings management from `.env` |
| `pydantic[email]` | Data validation and email validation |
| `rich` | Terminal formatting and logging |
| `tenacity` | Retry logic with exponential backoff |
| `httpx` | HTTP client for search APIs |
| `exa-py` | Exa search API |
| `mdformat` | Markdown formatting |
| `disposable-email-domains` | Disposable email domain blocklist |

## License

This project does not currently specify a license.
