#!/usr/bin/env python3
"""Drive IR-Camera recordings from the command line.

Uses only the standard library so it runs on any machine with Python 3.12,
including the Raspberry Pi itself. Every request bypasses proxies because the
corporate proxy otherwise hijacks both localhost and the Pi's hostname.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = os.environ.get(
    "IR_CAMERA_URL", "http://nbnlle0004.bsh.corp.bshg.com/IR-Camera/"
)
FORMATS = ("images", "json", "both")
FILENAME_PATTERN = re.compile(r'filename="([^"]+)"')
# The recorder keeps a session in memory; a stopped recording stays
# downloadable until the next start or a container restart.
POLL_INTERVAL_SECONDS = 1.0


class CommandError(RuntimeError):
    """A request the server refused, or an unreachable server."""


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _request(
    base_url: str,
    path: str,
    *,
    payload: dict[str, object] | None = None,
    method: str = "GET",
    timeout: float = 30.0,
) -> tuple[bytes, str | None]:
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    try:
        with _opener().open(request, timeout=timeout) as response:
            return response.read(), response.headers.get("Content-Disposition")
    except urllib.error.HTTPError as error:
        body = error.read()
        try:
            message = json.loads(body)["error"]
        except (ValueError, KeyError, TypeError):
            message = body.decode("utf-8", "replace").strip() or error.reason
        raise CommandError(f"{url} returned {error.code}: {message}") from error
    except OSError as error:
        raise CommandError(f"Cannot reach {url}: {error}") from error


def _json_request(
    base_url: str,
    path: str,
    *,
    payload: dict[str, object] | None = None,
    method: str = "GET",
) -> dict[str, object]:
    body, _ = _request(base_url, path, payload=payload, method=method)
    return json.loads(body)


def _describe(status: dict[str, object]) -> str:
    state = status["state"]
    if state == "idle":
        return "idle - no recording collected yet"
    limit = " (frame limit reached)" if status["limit_reached"] else ""
    return (
        f"{state} - session {status['session_id']}, format {status['format']}, "
        f"{status['frames']} frames, {status['bytes']} bytes{limit}"
    )


def command_status(args: argparse.Namespace) -> int:
    status = _json_request(args.url, "recording/status")
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print(_describe(status))
    return 0


def command_start(args: argparse.Namespace) -> int:
    status = _json_request(
        args.url, "recording/start", payload={"format": args.format}, method="POST"
    )
    print(f"Started {_describe(status)}")
    return 0


def command_stop(args: argparse.Namespace) -> int:
    status = _json_request(args.url, "recording/stop", method="POST")
    print(f"Stopped {_describe(status)}")
    return 0


def command_save(args: argparse.Namespace) -> int:
    body, disposition = _request(args.url, "recording/download", timeout=600.0)
    match = FILENAME_PATTERN.search(disposition or "")
    server_name = match.group(1) if match else "ir-camera-recording"
    destination = Path(args.output) if args.output else Path(server_name)
    if destination.is_dir():
        destination = destination / server_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(body)
    print(f"Saved {len(body)} bytes to {destination}")
    return 0


def command_record(args: argparse.Namespace) -> int:
    started = _json_request(
        args.url, "recording/start", payload={"format": args.format}, method="POST"
    )
    print(f"Started {_describe(started)}")
    deadline = time.monotonic() + args.duration
    try:
        while time.monotonic() < deadline:
            time.sleep(min(POLL_INTERVAL_SECONDS, max(0.0, deadline - time.monotonic())))
            status = _json_request(args.url, "recording/status")
            if status["state"] != "recording":
                print(f"Recording ended early: {_describe(status)}")
                break
            print(f"  {status['frames']} frames", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted; stopping the recording")
    print()
    status = _json_request(args.url, "recording/status")
    if status["state"] == "recording":
        status = _json_request(args.url, "recording/stop", method="POST")
    print(f"Stopped {_describe(status)}")
    if status["frames"] == 0:
        raise CommandError(
            "No frames were collected; the camera reported no live frames."
        )
    return command_save(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="irctl",
        description="Start, stop and save IR-Camera recordings.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Camera base URL (default: {DEFAULT_URL}; or set IR_CAMERA_URL)",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    status = subcommands.add_parser("status", help="Show the recorder state")
    status.add_argument(
        "--json", action="store_true", help="Print the raw status payload"
    )
    status.set_defaults(handler=command_status)

    start = subcommands.add_parser("start", help="Begin collecting frames")
    start.add_argument("--format", choices=FORMATS, default="both")
    start.set_defaults(handler=command_start)

    stop = subcommands.add_parser("stop", help="End the running recording")
    stop.set_defaults(handler=command_stop)

    save = subcommands.add_parser(
        "save", help="Download the finished recording to a file"
    )
    save.add_argument(
        "-o",
        "--output",
        help="File or directory to write (default: the server's filename here)",
    )
    save.set_defaults(handler=command_save)

    record = subcommands.add_parser(
        "record", help="Start, wait, stop and save in one command"
    )
    record.add_argument(
        "--duration", type=float, default=30.0, help="Seconds to record"
    )
    record.add_argument("--format", choices=FORMATS, default="both")
    record.add_argument("-o", "--output", help="File or directory to write")
    record.set_defaults(handler=command_record)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except CommandError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
