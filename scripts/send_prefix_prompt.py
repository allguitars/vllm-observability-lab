#!/usr/bin/env python3

"""Send one prefix file to an OpenAI-compatible Chat Completions API."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.6-luna"
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
ENV_KEYS = {
    "OPENAI_API_KEY",
    "LLM_BASE_URL",
    "MODEL_NAME",
    "TOKEN_LIMIT_PARAMETER",
    "REASONING_EFFORT",
}
TOKEN_LIMIT_PARAMETERS = {"max_completion_tokens", "max_tokens"}


def load_env_file(path: Path) -> None:
    """Load supported values from .env without overwriting shell variables."""

    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()

        key, separator, value = line.partition("=")
        key = key.strip()
        if separator and key in ENV_KEYS and key not in os.environ:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a prefix text file with a 16-token completion limit.",
    )
    parser.add_argument(
        "prefix_file",
        type=Path,
        help="UTF-8 text file to use as the user prompt.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME") or DEFAULT_MODEL,
        help=f"Model name (default: MODEL_NAME or {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL,
        help=f"API base URL (default: LLM_BASE_URL or {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="HTTP timeout in seconds (default: 300).",
    )
    return parser.parse_args()


def read_prefix(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Prefix file does not exist: {path}")

    prompt = path.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError(f"Prefix file is empty: {path}")
    return prompt


def main() -> int:
    try:
        load_env_file(ENV_FILE)
    except (OSError, UnicodeError) as error:
        print(f"Unable to read {ENV_FILE}: {error}", file=sys.stderr)
        return 2

    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")

    try:
        prompt = read_prefix(args.prefix_file)
    except (OSError, UnicodeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    token_limit_parameter = os.environ.get(
        "TOKEN_LIMIT_PARAMETER",
        "max_completion_tokens",
    )
    if token_limit_parameter not in TOKEN_LIMIT_PARAMETERS:
        supported_parameters = ", ".join(sorted(TOKEN_LIMIT_PARAMETERS))
        print(
            "TOKEN_LIMIT_PARAMETER must be one of: "
            f"{supported_parameters}.",
            file=sys.stderr,
        )
        return 2

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        token_limit_parameter: 16,
    }
    reasoning_effort = os.environ.get("REASONING_EFFORT")
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(
        url=f"{args.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    started_at = time.monotonic()
    try:
        with urlopen(request, timeout=args.timeout) as response:
            response_body = response.read().decode("utf-8")
            request_id = response.headers.get("x-request-id")
            response_headers = dict(response.headers.items())
    except HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        print(f"OpenAI API returned HTTP {error.code}: {error_body}", file=sys.stderr)
        return 1
    except URLError as error:
        print(f"OpenAI API request failed: {error.reason}", file=sys.stderr)
        return 1

    elapsed_seconds = time.monotonic() - started_at
    try:
        result = json.loads(response_body)
    except json.JSONDecodeError as error:
        print(f"Invalid JSON response: {error}", file=sys.stderr)
        return 1

    usage = result.get("usage", {})
    prompt_details = usage.get("prompt_tokens_details") or {}
    choices = result.get("choices") or []
    assistant_content = None
    if choices:
        assistant_content = choices[0].get("message", {}).get("content")

    summary = {
        "prefix_file": str(args.prefix_file.resolve()),
        "model": result.get("model", args.model),
        "response_id": result.get("id"),
        "request_id": request_id,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": prompt_details.get("cached_tokens"),
        "cache_write_tokens": prompt_details.get("cache_write_tokens"),
        "assistant_content": assistant_content,
        "response_headers": response_headers,
        "api_response": result,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
