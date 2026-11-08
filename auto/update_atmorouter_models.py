"""Fetch all Atmorouter models and update the atmorouter model list in models.yml."""

from os import getenv
from pathlib import Path

from httpx2 import Client

MODELS_URL = "https://atmorouter.dev/v1/models"
REQUEST_TIMEOUT = 60.0
PROVIDER_LINE = "atmorouter:"
MODELS_LINE = "models:"
API_KEY_NAME = "ATMOROUTER_API_KEY"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
TARGET_YML = Path("/Users/madesuryawan/.omp/agent/models.yml")

INDENT = " " * 4

MISSING_KEY_MSG = "ATMOROUTER_API_KEY not set and {env} not found"
UNKNOWN_KEY_MSG = "ATMOROUTER_API_KEY not found in {env}"
MISSING_PROVIDER_MSG = "atmorouter provider block not found in {yml}"
MISSING_MODELS_MSG = "models list not found inside atmorouter block"
MISSING_LIST_MSG = "expected '[' after models: header in {yml}"
UNTERMINATED_LIST_MSG = "unterminated models list in {yml}"


def read_api_key() -> str:
    if key := getenv(API_KEY_NAME):
        return key.strip()
    if not ENV_FILE.is_file():
        message = MISSING_KEY_MSG.format(env=ENV_FILE)
        raise SystemExit(message)
    for line in ENV_FILE.read_text().splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() == API_KEY_NAME:
            return value.strip().strip("\"'")
    message = UNKNOWN_KEY_MSG.format(env=ENV_FILE)
    raise SystemExit(message)


def fetch_models(api_key: str) -> list[str]:
    with Client(
        timeout=REQUEST_TIMEOUT,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    ) as client:
        response = client.get(MODELS_URL)
        response.raise_for_status()
        payload = response.json()
    return sorted({item["id"] for item in payload.get("data", []) if item.get("id")})


def render_models(models: list[str], list_indent: str) -> list[str]:
    item_indent = list_indent + INDENT
    field_indent = item_indent + INDENT
    lines = [f"{list_indent}["]
    for model in models:
        lines.append(f"{item_indent}{{")
        lines.append(f"{field_indent}id: {model},")
        lines.append(f"{field_indent}name: {model},")
        lines.append(f"{item_indent}}},")
    lines.append(f"{list_indent}],")
    return lines


def find_atmorouter_models_list(lines: list[str]) -> tuple[int, int, str]:
    provider = next((i for i, line in enumerate(lines) if line.strip() == PROVIDER_LINE), None)
    if provider is None:
        message = MISSING_PROVIDER_MSG.format(yml=TARGET_YML)
        raise SystemExit(message)

    header = next(
        (i for i in range(provider, len(lines)) if lines[i].strip() == MODELS_LINE),
        None,
    )
    if header is None:
        raise SystemExit(MISSING_MODELS_MSG)

    open_idx = header + 1
    if not lines[open_idx].strip().startswith("["):
        message = MISSING_LIST_MSG.format(yml=TARGET_YML)
        raise SystemExit(message)

    list_indent = lines[open_idx][: len(lines[open_idx]) - len(lines[open_idx].lstrip())]

    depth = 0
    for i in range(open_idx, len(lines)):
        stripped = lines[i].strip().rstrip(",")
        depth += stripped.count("[") - stripped.count("]")
        if depth == 0:
            return open_idx, i, list_indent
    message = UNTERMINATED_LIST_MSG.format(yml=TARGET_YML)
    raise SystemExit(message)


def main() -> None:
    models = fetch_models(read_api_key())
    lines = TARGET_YML.read_text().splitlines()

    start, end, list_indent = find_atmorouter_models_list(lines)
    updated = lines[:start] + render_models(models, list_indent) + lines[end + 1 :]
    TARGET_YML.write_text("\n".join(updated) + "\n")

    print(f"Updated {TARGET_YML} with {len(models)} atmorouter models")


if __name__ == "__main__":
    main()
