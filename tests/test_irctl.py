from __future__ import annotations

import json
import sys
import threading
import time
import zipfile
from pathlib import Path
from wsgiref.simple_server import WSGIRequestHandler, make_server

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import irctl  # noqa: E402

from ir_camera import FRAME_HEIGHT, FRAME_WIDTH, create_app  # noqa: E402


class PacedCamera:
    last_error = None

    def __init__(self, interval: float = 0.02) -> None:
        self._interval = interval
        self._offset = 0.0

    def read_frame(self) -> np.ndarray:
        time.sleep(self._interval)
        self._offset += 0.1
        return (
            np.linspace(
                20.0, 40.0, FRAME_WIDTH * FRAME_HEIGHT, dtype=np.float32
            ).reshape((FRAME_HEIGHT, FRAME_WIDTH))
            + self._offset
        )


class QuietHandler(WSGIRequestHandler):
    def log_message(self, *args):  # noqa: D102 - keep pytest output readable
        pass


@pytest.fixture
def camera_url():
    server = make_server(
        "127.0.0.1", 0, create_app(camera=PacedCamera()), handler_class=QuietHandler
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def run(*argv: str) -> int:
    return irctl.main(list(argv))


def test_cli_records_and_saves_an_archive(camera_url, tmp_path, capsys):
    assert run("--url", camera_url, "status") == 0
    assert "idle" in capsys.readouterr().out

    assert run("--url", camera_url, "start", "--format", "both") == 0
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = irctl._json_request(camera_url, "recording/status")
        if status["frames"] >= 2:
            break
        time.sleep(0.05)
    assert run("--url", camera_url, "stop") == 0
    assert run("--url", camera_url, "save", "-o", str(tmp_path)) == 0

    saved = list(tmp_path.glob("ir-camera-*.zip"))
    assert len(saved) == 1
    archive = zipfile.ZipFile(saved[0])
    metadata = json.loads(archive.read("metadata.json"))
    jpegs = [name for name in archive.namelist() if name.endswith(".jpg")]
    frames = json.loads(archive.read("frames.json"))
    assert len(jpegs) == metadata["frame_count"] == len(frames["frames"])
    assert frames["unit"] == "celsius"


def test_record_command_runs_the_whole_cycle(camera_url, tmp_path):
    destination = tmp_path / "session.json"

    assert (
        run(
            "--url",
            camera_url,
            "record",
            "--duration",
            "1.5",
            "--format",
            "json",
            "-o",
            str(destination),
        )
        == 0
    )

    payload = json.loads(destination.read_text())
    assert payload["frame_count"] == len(payload["frames"]) >= 1
    assert len(payload["frames"][0]) == FRAME_HEIGHT
    assert len(payload["frames"][0][0]) == FRAME_WIDTH
    assert irctl._json_request(camera_url, "recording/status")["state"] == "ready"


def test_cli_reports_server_refusals_without_a_traceback(camera_url, capsys):
    assert run("--url", camera_url, "stop") == 1
    assert "No recording is running." in capsys.readouterr().err

    assert run("--url", camera_url, "save") == 1
    assert "No finished recording" in capsys.readouterr().err


def test_cli_reports_an_unreachable_camera(capsys):
    # Port 1 is reserved and never serves the camera.
    assert run("--url", "http://127.0.0.1:1/", "status") == 1
    assert "Cannot reach" in capsys.readouterr().err
