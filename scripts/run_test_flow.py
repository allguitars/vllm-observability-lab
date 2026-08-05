#!/usr/bin/env python3

"""Run a configurable prefix-cache test flow and record raw events as JSONL."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"
SEND_PROMPT_SCRIPT = REPO_ROOT / "scripts" / "send_prefix_prompt.py"
SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_ACTIONS = {"capture_metrics", "send_prompt"}
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class StepExecutionError(RuntimeError):
    """A flow step failed with details suitable for the run log."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class EventWriter:
    """Append complete JSON events to a JSONL file."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self.event_sequence = 0
        self.file = path.open("x", encoding="utf-8")

    def close(self) -> None:
        self.file.close()

    def write(self, event: str, **fields: object) -> None:
        self.event_sequence += 1
        record = {
            "schema_version": SUPPORTED_SCHEMA_VERSION,
            "event": event,
            "event_sequence": self.event_sequence,
            "run_id": self.run_id,
            "timestamp": current_timestamp(),
            **fields,
        }
        self.file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        self.file.write("\n")
        self.file.flush()


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def load_env_file(path: Path) -> None:
    """Load the metrics URL without overwriting the shell environment."""

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
        if separator and key == "VLLM_METRICS_URL" and key not in os.environ:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a JSON test flow and save raw metrics and responses as JSONL.",
    )
    parser.add_argument(
        "flow_file",
        type=Path,
        help="Flow definition file, such as flows/prefix-cache-aaba.flow.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs",
        help="Parent directory for run output (default: repo/runs).",
    )
    parser.add_argument(
        "--metrics-url",
        default=os.environ.get("VLLM_METRICS_URL"),
        help="vLLM metrics URL (default: VLLM_METRICS_URL).",
    )
    parser.add_argument(
        "--metrics-timeout",
        type=float,
        default=10.0,
        help="Metrics request timeout in seconds (default: 10).",
    )
    parser.add_argument(
        "--prompt-timeout",
        type=float,
        default=300.0,
        help="LLM request timeout in seconds (default: 300).",
    )
    return parser.parse_args()


def load_flow(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Flow file does not exist: {path}")

    try:
        flow = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid flow JSON: {error}") from error

    validate_flow(flow)
    return flow


def validate_flow(flow: object) -> None:
    if not isinstance(flow, dict):
        raise ValueError("Flow root must be a JSON object.")
    if flow.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SUPPORTED_SCHEMA_VERSION}."
        )

    flow_id = flow.get("id")
    if not isinstance(flow_id, str) or not SAFE_ID_PATTERN.fullmatch(flow_id):
        raise ValueError(
            "Flow id must contain only letters, numbers, dots, underscores, or hyphens."
        )
    if "description" in flow and not isinstance(flow["description"], str):
        raise ValueError("Flow description must be a string.")

    steps = flow.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Flow steps must be a non-empty array.")

    step_ids = set()
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Step {index} must be a JSON object.")

        step_id = step.get("id")
        if not isinstance(step_id, str) or not SAFE_ID_PATTERN.fullmatch(step_id):
            raise ValueError(f"Step {index} has an invalid id.")
        if step_id in step_ids:
            raise ValueError(f"Duplicate step id: {step_id}")
        step_ids.add(step_id)

        action = step.get("action")
        if action not in SUPPORTED_ACTIONS:
            supported = ", ".join(sorted(SUPPORTED_ACTIONS))
            raise ValueError(
                f"Step {step_id} action must be one of: {supported}."
            )
        if "label" in step and not isinstance(step["label"], str):
            raise ValueError(f"Step {step_id} label must be a string.")

        if action == "send_prompt":
            prompt_file = step.get("prompt_file")
            if not isinstance(prompt_file, str) or not prompt_file:
                raise ValueError(
                    f"Step {step_id} must define a non-empty prompt_file."
                )
            resolve_prompt_file(prompt_file, step_id)


def resolve_prompt_file(prompt_file: str, step_id: str) -> Path:
    path = (REPO_ROOT / prompt_file).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError(
            f"Step {step_id} prompt_file must be inside the repository."
        ) from error
    if not path.is_file():
        raise ValueError(f"Step {step_id} prompt_file does not exist: {prompt_file}")
    return path


def create_event_writer(
    output_root: Path,
    flow_id: str,
) -> tuple[str, Path, EventWriter]:
    output_root.mkdir(parents=True, exist_ok=True)
    base_run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{flow_id}"

    for suffix in range(1, 1000):
        run_id = base_run_id if suffix == 1 else f"{base_run_id}_{suffix}"
        event_path = output_root / f"run_{run_id}.jsonl"
        try:
            writer = EventWriter(event_path, run_id)
        except FileExistsError:
            continue
        return run_id, event_path, writer

    raise OSError(f"Unable to create a unique run file under {output_root}")


def capture_metrics(metrics_url: str, timeout: float) -> dict:
    try:
        request = Request(
            metrics_url,
            headers={"Accept": "text/plain"},
            method="GET",
        )
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return {
                "format": "prometheus_text",
                "source": {
                    "url": response.geturl(),
                    "http_status": response.status,
                    "content_type": response.headers.get("Content-Type"),
                    "response_headers": dict(response.headers.items()),
                },
                "raw": raw,
            }
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise StepExecutionError(
            f"Metrics endpoint returned HTTP {error.code}.",
            {
                "url": metrics_url,
                "http_status": error.code,
                "response_body": body,
            },
        ) from error
    except (URLError, TimeoutError, ValueError) as error:
        reason = getattr(error, "reason", str(error))
        raise StepExecutionError(
            f"Metrics request failed: {reason}",
            {"url": metrics_url},
        ) from error


def send_prompt(prompt_file: Path, timeout: float) -> dict:
    command = [
        sys.executable,
        str(SEND_PROMPT_SCRIPT),
        str(prompt_file),
        "--timeout",
        str(timeout),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise StepExecutionError(
            f"Unable to run send_prefix_prompt.py: {error}",
        ) from error

    if completed.returncode != 0:
        raise StepExecutionError(
            f"send_prefix_prompt.py exited with code {completed.returncode}.",
            {
                "return_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )

    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise StepExecutionError(
            "send_prefix_prompt.py returned invalid JSON.",
            {
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        ) from error

    return {
        "format": "json",
        "response": response,
    }


def step_metadata(step: dict, step_index: int) -> dict:
    metadata = {
        "index": step_index,
        "id": step["id"],
        "action": step["action"],
    }
    if "label" in step:
        metadata["label"] = step["label"]
    if "prompt_file" in step:
        metadata["prompt_file"] = step["prompt_file"]
    return metadata


def run_flow(args: argparse.Namespace, flow: dict, flow_path: Path) -> int:
    needs_metrics = any(
        step["action"] == "capture_metrics" for step in flow["steps"]
    )
    if needs_metrics and not args.metrics_url:
        raise ValueError(
            "VLLM_METRICS_URL is required for capture_metrics steps."
        )
    if args.metrics_timeout <= 0 or args.prompt_timeout <= 0:
        raise ValueError("Request timeouts must be greater than zero.")

    run_id, event_path, writer = create_event_writer(
        args.output_root.resolve(),
        flow["id"],
    )
    active_step = None
    active_step_started_at = None

    print(f"Run ID: {run_id}")
    print(f"Output: {event_path}")

    try:
        writer.write(
            "run_started",
            flow_file=str(flow_path.resolve()),
            flow=flow,
        )

        for step_index, step in enumerate(flow["steps"], start=1):
            active_step = step_metadata(step, step_index)
            active_step_started_at = time.monotonic()
            writer.write("step_started", step=active_step)
            print(
                f"[{step_index}/{len(flow['steps'])}] "
                f"{step['id']}: {step['action']}"
            )

            try:
                if step["action"] == "capture_metrics":
                    payload = capture_metrics(
                        args.metrics_url,
                        args.metrics_timeout,
                    )
                else:
                    prompt_file = resolve_prompt_file(
                        step["prompt_file"],
                        step["id"],
                    )
                    payload = send_prompt(prompt_file, args.prompt_timeout)
            except StepExecutionError as error:
                elapsed_seconds = round(
                    time.monotonic() - active_step_started_at,
                    3,
                )
                writer.write(
                    "step_failed",
                    step=active_step,
                    status="failed",
                    elapsed_seconds=elapsed_seconds,
                    error={
                        "message": str(error),
                        "details": error.details,
                    },
                )
                writer.write(
                    "run_completed",
                    status="failed",
                    failed_step_id=step["id"],
                )
                print(f"Step {step['id']} failed: {error}", file=sys.stderr)
                return 1

            elapsed_seconds = round(
                time.monotonic() - active_step_started_at,
                3,
            )
            writer.write(
                "step_completed",
                step=active_step,
                status="success",
                elapsed_seconds=elapsed_seconds,
                payload=payload,
            )
            active_step = None
            active_step_started_at = None

        writer.write(
            "run_completed",
            status="success",
            completed_steps=len(flow["steps"]),
        )
        print("Flow completed successfully.")
        return 0
    except KeyboardInterrupt:
        if active_step is not None and active_step_started_at is not None:
            elapsed_seconds = round(
                time.monotonic() - active_step_started_at,
                3,
            )
            writer.write(
                "step_cancelled",
                step=active_step,
                status="cancelled",
                elapsed_seconds=elapsed_seconds,
            )
        writer.write(
            "run_completed",
            status="cancelled",
            cancelled_step_id=(
                active_step["id"] if active_step is not None else None
            ),
        )
        print("Flow cancelled.", file=sys.stderr)
        return 130
    finally:
        writer.close()


def main() -> int:
    try:
        load_env_file(ENV_FILE)
        args = parse_args()
        flow_path = args.flow_file.resolve()
        flow = load_flow(flow_path)
        return run_flow(args, flow, flow_path)
    except (OSError, UnicodeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
