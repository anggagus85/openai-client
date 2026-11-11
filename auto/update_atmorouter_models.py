"""Fetch all Atmorouter models and update models.yml and opencode.json."""

import json
import stat
import tempfile
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
TARGET_JSON = Path("/Users/madesuryawan/.config/opencode/opencode.json")
JSON_PROVIDER_LINE = '"atmorouter": {'
JSON_MODELS_LINE = '"models": {'

INDENT = " " * 4

MISSING_KEY_MSG = "ATMOROUTER_API_KEY not set and {env} not found"
UNKNOWN_KEY_MSG = "ATMOROUTER_API_KEY not found in {env}"
MISSING_PROVIDER_MSG = "atmorouter provider block not found in {yml}"
MISSING_MODELS_MSG = "models list not found inside atmorouter block"
MISSING_LIST_MSG = "expected '[' after models: header in {yml}"
UNTERMINATED_LIST_MSG = "unterminated models list in {yml}"
MISSING_JSON_PROVIDER_MSG = "atmorouter provider block not found in {json_file}"
MISSING_JSON_MODELS_MSG = "models object not found inside atmorouter block in {json_file}"
UNTERMINATED_JSON_PROVIDER_MSG = "unterminated atmorouter provider block in {json_file}"
UNTERMINATED_JSON_MODELS_MSG = "unterminated models object in {json_file}"
MISSING_FILE_MSG = "target file not found: {path}"
EMPTY_MODELS_MSG = "atmorouter returned no models; leaving files untouched"
INVALID_JSON_MSG = "rendered models are not valid JSON; nothing written to {json_file}"


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


def find_yaml_block_end(lines: list[str], provider: int) -> int:
    provider_indent = len(lines[provider]) - len(lines[provider].lstrip())
    for i in range(provider + 1, len(lines)):
        if not lines[i].strip():
            continue
        indent = len(lines[i]) - len(lines[i].lstrip())
        if indent <= provider_indent:
            return i
    return len(lines)


def find_atmorouter_models_list(lines: list[str]) -> tuple[int, int, str]:
    provider = next((i for i, line in enumerate(lines) if line.strip() == PROVIDER_LINE), None)
    if provider is None:
        message = MISSING_PROVIDER_MSG.format(yml=TARGET_YML)
        raise SystemExit(message)

    block_end = find_yaml_block_end(lines, provider)
    header = next(
        (i for i in range(provider, block_end) if lines[i].strip() == MODELS_LINE),
        None,
    )
    if header is None:
        raise SystemExit(MISSING_MODELS_MSG)

    open_idx = header + 1
    if open_idx >= block_end or not lines[open_idx].strip().startswith("["):
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


def render_json_models(models: list[str], entry_indent: str) -> list[str]:
    field_indent = entry_indent + INDENT
    lines = []
    for index, model in enumerate(models):
        key = json.dumps(model)
        comma = "," if index < len(models) - 1 else ""
        lines.append(f"{entry_indent}{key}: {{")
        lines.append(f'{field_indent}"name": {key}')
        lines.append(f"{entry_indent}}}{comma}")
    return lines


def check_target_file(path: Path) -> None:
    if not path.is_file():
        message = MISSING_FILE_MSG.format(path=path)
        raise SystemExit(message)


def write_atomic(path: Path, text: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with open(handle, "w") as stream:
            stream.write(text)
        temp_path.chmod(mode)
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def find_json_block_end(lines: list[str], start: int) -> int | None:
    depth = 0
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        depth += stripped.count("{") - stripped.count("}")
        if depth == 0:
            return i
    return None


def find_json_models_block(lines: list[str]) -> tuple[int, int, str]:
    provider = next(
        (i for i, line in enumerate(lines) if line.strip().startswith(JSON_PROVIDER_LINE)),
        None,
    )
    if provider is None:
        message = MISSING_JSON_PROVIDER_MSG.format(json_file=TARGET_JSON)
        raise SystemExit(message)

    provider_end = find_json_block_end(lines, provider)
    if provider_end is None:
        message = UNTERMINATED_JSON_PROVIDER_MSG.format(json_file=TARGET_JSON)
        raise SystemExit(message)

    header = next(
        (i for i in range(provider, provider_end) if lines[i].strip() == JSON_MODELS_LINE),
        None,
    )
    if header is None:
        message = MISSING_JSON_MODELS_MSG.format(json_file=TARGET_JSON)
        raise SystemExit(message)

    header_indent = lines[header][: len(lines[header]) - len(lines[header].lstrip())]
    entry_indent = None
    depth = 0
    for i in range(header, provider_end):
        stripped = lines[i].strip()
        if i > header and entry_indent is None and stripped.endswith("{"):
            entry_indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
        depth += stripped.count("{") - stripped.count("}")
        if depth == 0:
            return header, i, entry_indent or header_indent + INDENT
    message = UNTERMINATED_JSON_MODELS_MSG.format(json_file=TARGET_JSON)
    raise SystemExit(message)


def main() -> None:
    check_target_file(TARGET_YML)
    check_target_file(TARGET_JSON)

    models = fetch_models(read_api_key())
    if not models:
        raise SystemExit(EMPTY_MODELS_MSG)

    yml_lines = TARGET_YML.read_text().splitlines()
    start, end, list_indent = find_atmorouter_models_list(yml_lines)
    yml_updated = (
        "\n".join(yml_lines[:start] + render_models(models, list_indent) + yml_lines[end + 1 :])
        + "\n"
    )

    json_lines = TARGET_JSON.read_text().splitlines()
    json_start, json_end, entry_indent = find_json_models_block(json_lines)
    merged = (
        json_lines[: json_start + 1]
        + render_json_models(models, entry_indent)
        + json_lines[json_end:]
    )
    json_updated = "\n".join(merged) + "\n"

    try:
        json.loads(json_updated)
    except json.JSONDecodeError:
        message = INVALID_JSON_MSG.format(json_file=TARGET_JSON)
        raise SystemExit(message) from None

    write_atomic(TARGET_YML, yml_updated)
    write_atomic(TARGET_JSON, json_updated)

    print(f"Updated {TARGET_YML} and {TARGET_JSON} with {len(models)} atmorouter models")


if __name__ == "__main__":
    main()
