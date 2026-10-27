#!/usr/bin/env python3
"""MLX90640 Raspberry Pi thermal camera web streamer."""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

FRAME_WIDTH = 32
FRAME_HEIGHT = 24
DISPLAY_SCALE = 20
# The MLX90640 measures object temperatures from -40 C to 300 C, so the color
# map spans its full specified range.
MIN_TEMPERATURE_C = -40.0
MAX_TEMPERATURE_C = 300.0
FRAME_READ_ATTEMPTS = 3
FRAME_RETRY_DELAY_SECONDS = 0.1
CAPTURE_FAILURE_DELAY_SECONDS = 0.5
# Gunicorn runs one worker with a fixed thread pool and every MJPEG viewer
# holds its thread for the whole response. Capping concurrent viewers keeps
# threads free for the page, the status poll and the health check.
MAX_STREAM_VIEWERS = 4
# Frames averaged for the published capture rate.
FPS_SAMPLE_FRAMES = 10
# Longest a snapshot caller waits for the next published frame.
SNAPSHOT_TIMEOUT_SECONDS = 6.0
# Upper bound for one in-memory recording; ~16 minutes at 2 full frames/s.
MAX_RECORDING_FRAMES = 2000
RECORDING_POLL_TIMEOUT_SECONDS = 2.0
# MLX90640 control-register encoding for 4 Hz, or about 2 complete frames/s.
SENSOR_REFRESH_RATE = 0x03
RECORDING_FORMATS = ("images", "json", "both")
FRAME_READ_ERRORS = (RuntimeError, ValueError, OSError)
LOGGER = logging.getLogger(__name__)


class StreamCapacityError(RuntimeError):
    """Raised when the thermal stream already serves its viewer limit."""


class Camera(Protocol):
    def read_frame(self) -> np.ndarray | None: ...


class Mlx90640Camera:
    """Read the MLX90640 on Raspberry Pi I2C bus 1."""

    def __init__(
        self,
        *,
        sensor=None,
        read_attempts: int = FRAME_READ_ATTEMPTS,
        retry_delay: float = FRAME_RETRY_DELAY_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        logger: logging.Logger | None = None,
    ) -> None:
        if read_attempts < 1:
            raise ValueError("read_attempts must be at least 1")

        self._read_lock = threading.Lock()
        self._sensor = sensor
        self._last_error: str | None = None
        self._frame = [0.0] * (FRAME_WIDTH * FRAME_HEIGHT)
        self._read_attempts = read_attempts
        self._retry_delay = retry_delay
        self._sleep = sleep
        self._logger = logger if logger is not None else LOGGER

    @property
    def last_error(self) -> str | None:
        with self._read_lock:
            return self._last_error

    def _initialize(self) -> None:
        import adafruit_mlx90640
        import board
        import busio

        self._logger.info(
            "Initializing MLX90640 address=0x33, requested I2C frequency=800000 Hz, "
            "refresh rate=4 Hz (~2 full frames/s)"
        )
        i2c = busio.I2C(board.SCL, board.SDA, frequency=800_000)
        sensor = adafruit_mlx90640.MLX90640(i2c)
        sensor.refresh_rate = SENSOR_REFRESH_RATE
        self._sensor = sensor
        self._logger.info("MLX90640 initialization completed")

    def read_frame(self) -> np.ndarray | None:
        with self._read_lock:
            if self._sensor is None:
                try:
                    self._initialize()
                except FRAME_READ_ERRORS as error:
                    self._last_error = (
                        f"{type(error).__name__}: {error}"
                    )
                    self._logger.error(
                        "MLX90640 initialization failed: %s",
                        error,
                        exc_info=True,
                    )
                    return None

            for attempt in range(1, self._read_attempts + 1):
                try:
                    actual_rate = self._sensor.refresh_rate
                    if actual_rate != SENSOR_REFRESH_RATE:
                        self._logger.warning(
                            "MLX90640 refresh register changed from %r to %r; "
                            "restoring 4 Hz",
                            SENSOR_REFRESH_RATE,
                            actual_rate,
                        )
                        self._sensor.refresh_rate = SENSOR_REFRESH_RATE
                    self._sensor.getFrame(self._frame)
                except FRAME_READ_ERRORS as error:
                    if attempt == self._read_attempts:
                        self._last_error = (
                            f"{type(error).__name__}: {error}"
                        )
                        self._logger.error(
                            "MLX90640 frame read failed on attempt %d/%d; "
                            "stream remains active: %s",
                            attempt,
                            self._read_attempts,
                            error,
                        )
                        return None

                    self._logger.warning(
                        "MLX90640 frame read failed on attempt %d/%d: %s; "
                        "retrying in %.3f seconds",
                        attempt,
                        self._read_attempts,
                        error,
                        self._retry_delay,
                    )
                    self._sleep(self._retry_delay)
                    continue

                self._last_error = None
                temperatures = np.asarray(
                    self._frame,
                    dtype=np.float32,
                ).reshape((FRAME_HEIGHT, FRAME_WIDTH))
                return np.flipud(temperatures)

        return None


def render_thermal_frame(temperatures: np.ndarray) -> np.ndarray:
    """Color one 32 x 24 temperature frame; the browser does the scaling."""
    if temperatures.shape != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(
            f"Expected a {FRAME_HEIGHT} x {FRAME_WIDTH} frame, "
            f"received {temperatures.shape}"
        )

    normalized = np.clip(
        temperatures,
        MIN_TEMPERATURE_C,
        MAX_TEMPERATURE_C,
    )
    normalized = (normalized - MIN_TEMPERATURE_C) / (
        MAX_TEMPERATURE_C - MIN_TEMPERATURE_C
    )
    grayscale = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(grayscale, cv2.COLORMAP_INFERNO)


@dataclass(frozen=True)
class FrameSnapshot:
    """The newest published frame, as both JPEG bytes and raw temperatures."""

    sequence: int
    captured_at: datetime
    jpeg: bytes
    temperatures: np.ndarray


class FrameBroadcaster:
    """Capture once and broadcast each MJPEG frame to every connected viewer."""

    def __init__(
        self,
        camera: Camera,
        *,
        failure_delay: float = CAPTURE_FAILURE_DELAY_SECONDS,
        max_viewers: int = MAX_STREAM_VIEWERS,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        if max_viewers < 1:
            raise ValueError("max_viewers must be at least 1")

        self._camera = camera
        self._failure_delay = failure_delay
        self._max_viewers = max_viewers
        self._clock = clock
        self._logger = logger if logger is not None else LOGGER
        self._condition = threading.Condition()
        self._capture_thread: threading.Thread | None = None
        self._subscriber_count = 0
        self._stop_requested = False
        self._latest_frame: bytes | None = None
        self._latest_jpeg: bytes | None = None
        self._latest_temperatures: np.ndarray | None = None
        self._latest_captured_at: datetime | None = None
        self._snapshot_waiters = 0
        self._sequence = 0
        self._frame_times: list[float] = []

    @property
    def subscriber_count(self) -> int:
        with self._condition:
            return self._subscriber_count

    @property
    def max_viewers(self) -> int:
        return self._max_viewers

    def _capture_fps(self) -> float | None:
        """Mean capture rate of the recent frames, or None below two samples."""
        if len(self._frame_times) < 2:
            return None
        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed <= 0:
            return None
        return (len(self._frame_times) - 1) / elapsed

    def latest_temperatures(self) -> np.ndarray | None:
        """The newest published temperature grid, or None before the first frame."""
        with self._condition:
            return self._latest_temperatures

    def camera_status(self) -> dict[str, object]:
        error = getattr(self._camera, "last_error", None)
        with self._condition:
            has_frame = self._latest_frame is not None
            capture_running = self._capture_thread is not None
            viewers = self._subscriber_count
            fps = self._capture_fps()
            temperatures = self._latest_temperatures

        average_celsius = (
            None if temperatures is None else round(float(temperatures.mean()), 1)
        )
        status: dict[str, object] = {
            "viewers": viewers,
            "max_viewers": self._max_viewers,
            "fps": None if fps is None else round(fps, 1),
            "average_celsius": average_celsius,
            "scale_min_celsius": MIN_TEMPERATURE_C,
            "scale_max_celsius": MAX_TEMPERATURE_C,
        }
        if error:
            status.update(
                state="offline",
                message="CAMERA OFFLINE",
                detail=(
                    "MLX90640 at I2C address 0x33 is not responding. "
                    f"{error}"
                ),
            )
        elif has_frame:
            status.update(
                state="online",
                message="CAMERA ONLINE",
                detail="Live thermal frames are available.",
            )
        else:
            status.update(
                state="connecting",
                message="CONNECTING TO CAMERA",
                detail=(
                    "Waiting for the first MLX90640 frame."
                    if capture_running
                    else "The thermal stream has not started yet."
                ),
            )
        return status

    def _start_capture_locked(self) -> None:
        """Start the shared capture thread; the caller holds the condition."""
        self._stop_requested = False
        if self._capture_thread is None:
            self._capture_thread = threading.Thread(
                target=self._capture_frames,
                name="mlx90640-capture",
                daemon=True,
            )
            self._capture_thread.start()

    def latest_snapshot(
        self,
        timeout: float = SNAPSHOT_TIMEOUT_SECONDS,
    ) -> FrameSnapshot | None:
        """Wait for the next published frame and return it, or None on timeout.

        Snapshot waiters are deliberately exempt from ``max_viewers``: that cap
        protects the thread pool from long-lived MJPEG responses, while a
        snapshot holds its thread for at most ``timeout`` seconds.
        """
        with self._condition:
            self._snapshot_waiters += 1
            try:
                self._start_capture_locked()
                previous_sequence = self._sequence
                self._condition.wait_for(
                    lambda: self._sequence != previous_sequence,
                    timeout=timeout,
                )
                jpeg = self._latest_jpeg
                temperatures = self._latest_temperatures
                captured_at = self._latest_captured_at
                sequence = self._sequence
            finally:
                self._snapshot_waiters -= 1
                if self._snapshot_waiters == 0 and self._subscriber_count == 0:
                    self._stop_requested = True
                self._condition.notify_all()

        if jpeg is None or temperatures is None or captured_at is None:
            return None
        return FrameSnapshot(
            sequence=sequence,
            captured_at=captured_at,
            jpeg=jpeg,
            temperatures=temperatures,
        )

    def frames(self) -> Iterator[bytes]:
        """Subscribe to the shared stream, refusing viewers beyond the cap."""
        with self._condition:
            if self._subscriber_count >= self._max_viewers:
                self._logger.warning(
                    "Refusing thermal stream viewer; viewers=%d reached the cap",
                    self._subscriber_count,
                )
                raise StreamCapacityError(
                    f"The thermal stream already serves {self._max_viewers} "
                    "viewers. Close an open tab and retry."
                )
            self._subscriber_count += 1
            previous_sequence = self._sequence
            self._start_capture_locked()
            self._logger.info(
                "Thermal stream viewer connected; viewers=%d",
                self._subscriber_count,
            )
        return self._subscribed_frames(previous_sequence)

    def _subscribed_frames(self, previous_sequence: int) -> Iterator[bytes]:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._sequence != previous_sequence
                    )
                    previous_sequence = self._sequence
                    frame = self._latest_frame
                if frame is not None:
                    yield frame
        finally:
            with self._condition:
                self._subscriber_count -= 1
                if self._subscriber_count == 0:
                    self._stop_requested = True
                self._condition.notify_all()
                self._logger.info(
                    "Thermal stream viewer disconnected; viewers=%d",
                    self._subscriber_count,
                )

    def _is_idle_locked(self) -> bool:
        """True when nothing wants frames; the caller holds the condition."""
        return (
            self._stop_requested
            and self._subscriber_count == 0
            and self._snapshot_waiters == 0
        )

    def _capture_frames(self) -> None:
        self._logger.info("MLX90640 shared capture loop started")
        try:
            while True:
                with self._condition:
                    if self._is_idle_locked():
                        return

                try:
                    temperatures = self._camera.read_frame()
                except FRAME_READ_ERRORS as error:
                    self._logger.error(
                        "Camera read escaped its bounded retry handler: %s",
                        error,
                        exc_info=True,
                    )
                    temperatures = None

                if temperatures is None:
                    with self._condition:
                        self._frame_times.clear()
                        if self._condition.wait_for(
                            self._is_idle_locked,
                            timeout=self._failure_delay,
                        ):
                            return
                    continue

                try:
                    image = render_thermal_frame(temperatures)
                    encoded, jpeg = cv2.imencode(".jpg", image)
                except (ValueError, cv2.error) as error:
                    self._logger.error(
                        "Thermal frame encoding failed: %s",
                        error,
                        exc_info=True,
                    )
                    continue

                if not encoded:
                    self._logger.error("OpenCV could not encode the thermal frame")
                    continue

                jpeg_bytes = jpeg.tobytes()
                multipart_frame = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg_bytes
                    + b"\r\n"
                )
                with self._condition:
                    self._latest_frame = multipart_frame
                    self._latest_jpeg = jpeg_bytes
                    self._latest_temperatures = temperatures
                    self._latest_captured_at = datetime.now(timezone.utc)
                    self._sequence += 1
                    self._frame_times.append(self._clock())
                    if len(self._frame_times) > FPS_SAMPLE_FRAMES:
                        del self._frame_times[:-FPS_SAMPLE_FRAMES]
                    self._condition.notify_all()
        finally:
            with self._condition:
                self._capture_thread = None
                self._condition.notify_all()
            self._logger.info("MLX90640 shared capture loop stopped")


class RecordingError(RuntimeError):
    """Raised when a recording request does not match the recorder state."""


@dataclass(frozen=True)
class RecordedFrame:
    """One captured frame kept for the current recording session."""

    sequence: int
    captured_at: datetime
    jpeg: bytes
    temperatures: np.ndarray


def _isoformat(moment: datetime | None) -> str | None:
    return None if moment is None else moment.isoformat()


class Recorder:
    """Collect frames from the shared stream into one downloadable session."""

    def __init__(
        self,
        stream: FrameBroadcaster,
        *,
        max_frames: int = MAX_RECORDING_FRAMES,
        poll_timeout: float = RECORDING_POLL_TIMEOUT_SECONDS,
        logger: logging.Logger | None = None,
    ) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be at least 1")

        self._stream = stream
        self._max_frames = max_frames
        self._poll_timeout = poll_timeout
        self._logger = logger if logger is not None else LOGGER
        self._lock = threading.Lock()
        self._state = "idle"
        self._format = RECORDING_FORMATS[0]
        self._session_id: str | None = None
        self._started_at: datetime | None = None
        self._stopped_at: datetime | None = None
        self._limit_reached = False
        self._frames: list[RecordedFrame] = []
        self._worker: threading.Thread | None = None

    def start(self, output_format: str) -> dict[str, object]:
        """Begin a new recording, discarding any previously collected frames."""
        if output_format not in RECORDING_FORMATS:
            raise RecordingError("Unknown format; use images, json or both.")

        with self._lock:
            if self._state == "recording":
                raise RecordingError("A recording is already running.")

            started_at = datetime.now(timezone.utc)
            self._frames = []
            self._format = output_format
            self._session_id = started_at.strftime("%Y%m%d-%H%M%S")
            self._started_at = started_at
            self._stopped_at = None
            self._limit_reached = False
            self._state = "recording"
            self._worker = threading.Thread(
                target=self._collect,
                name="ir-camera-recorder",
                daemon=True,
            )
            self._worker.start()
            self._logger.info(
                "Recording %s started; session=%s",
                output_format,
                self._session_id,
            )
            return self._status_locked()

    def stop(self) -> dict[str, object]:
        """End the running recording and keep its frames for download."""
        with self._lock:
            if self._state != "recording":
                raise RecordingError("No recording is running.")
            self._state = "ready"
            self._stopped_at = datetime.now(timezone.utc)
            self._logger.info(
                "Recording stopped; session=%s frames=%d",
                self._session_id,
                len(self._frames),
            )
            return self._status_locked()

    def status(self) -> dict[str, object]:
        with self._lock:
            return self._status_locked()

    def download(self) -> tuple[bytes, str, str]:
        """Return (payload, filename, mimetype) for the finished recording."""
        with self._lock:
            if self._state != "ready" or not self._frames:
                raise RecordingError(
                    "No finished recording is available to download."
                )
            frames = list(self._frames)
            session_id = self._session_id or "recording"
            output_format = self._format
            started_at = self._started_at
            stopped_at = self._stopped_at

        timeframes = [frame.captured_at.isoformat() for frame in frames]
        if output_format in ("json", "both"):
            payload = {
                "session_id": session_id,
                "format": output_format,
                "started_at": _isoformat(started_at),
                "stopped_at": _isoformat(stopped_at),
                "width": FRAME_WIDTH,
                "height": FRAME_HEIGHT,
                "unit": "celsius",
                "frame_count": len(frames),
                "timeframes": timeframes,
                "frames": [
                    [
                        [round(float(value), 2) for value in row]
                        for row in frame.temperatures
                    ]
                    for frame in frames
                ],
            }
            json_payload = json.dumps(payload).encode("utf-8")
            if output_format == "json":
                return (
                    json_payload,
                    f"ir-camera-{session_id}.json",
                    "application/json",
                )

        buffer = io.BytesIO()
        # JPEG payloads are already compressed, so storing them is faster and
        # no larger than deflating them a second time.
        with zipfile.ZipFile(
            buffer,
            "w",
            compression=zipfile.ZIP_STORED,
        ) as archive:
            for index, frame in enumerate(frames, start=1):
                archive.writestr(f"frame-{index:06d}.jpg", frame.jpeg)
            if output_format == "both":
                archive.writestr("frames.json", json_payload)
            archive.writestr(
                "metadata.json",
                json.dumps(
                    {
                        "session_id": session_id,
                        "format": output_format,
                        "started_at": _isoformat(started_at),
                        "stopped_at": _isoformat(stopped_at),
                        "frame_count": len(frames),
                        "timeframes": timeframes,
                    }
                ),
            )
        return (
            buffer.getvalue(),
            f"ir-camera-{session_id}.zip",
            "application/zip",
        )

    def _status_locked(self) -> dict[str, object]:
        size = 0
        if self._format in ("json", "both"):
            size += sum(frame.temperatures.nbytes for frame in self._frames)
        if self._format in ("images", "both"):
            size += sum(len(frame.jpeg) for frame in self._frames)
        return {
            "state": self._state,
            "session_id": self._session_id,
            "format": self._format,
            "started_at": _isoformat(self._started_at),
            "stopped_at": _isoformat(self._stopped_at),
            "frames": len(self._frames),
            "bytes": size,
            "limit_reached": self._limit_reached,
            "max_frames": self._max_frames,
        }

    def _collect(self) -> None:
        last_sequence: int | None = None
        while True:
            with self._lock:
                if (
                    self._state != "recording"
                    or self._worker is not threading.current_thread()
                ):
                    return

            # Reuses the shared capture loop, so recording never opens a second
            # sensor read path and never counts against the viewer cap.
            snapshot = self._stream.latest_snapshot(timeout=self._poll_timeout)
            if snapshot is None:
                # The camera is offline or slow; stay open so a recovering
                # sensor is picked up without the operator restarting.
                continue
            if snapshot.sequence == last_sequence:
                time.sleep(0.05)
                continue

            last_sequence = snapshot.sequence
            with self._lock:
                if (
                    self._state != "recording"
                    or self._worker is not threading.current_thread()
                ):
                    return
                self._frames.append(
                    RecordedFrame(
                        sequence=snapshot.sequence,
                        captured_at=snapshot.captured_at,
                        jpeg=snapshot.jpeg,
                        temperatures=snapshot.temperatures.copy(),
                    )
                )
                if len(self._frames) >= self._max_frames:
                    self._limit_reached = True
                    self._state = "ready"
                    self._stopped_at = datetime.now(timezone.utc)
                    self._logger.warning(
                        "Recording reached the %d frame limit; session=%s",
                        self._max_frames,
                        self._session_id,
                    )
                    return


def render_temperature_scale(
    width: int = 28,
    height: int = FRAME_HEIGHT * DISPLAY_SCALE,
) -> np.ndarray:
    """Render the inferno ramp the feed uses, hottest at the top."""
    ramp = np.linspace(255, 0, height).astype(np.uint8).reshape(height, 1)
    return cv2.applyColorMap(np.repeat(ramp, width, axis=1), cv2.COLORMAP_INFERNO)


_TEMPERATURE_SCALE_PNG: bytes | None = None


def temperature_scale_png() -> bytes:
    """PNG of the color scale; cached because the range is a fixed constant."""
    global _TEMPERATURE_SCALE_PNG
    if _TEMPERATURE_SCALE_PNG is None:
        encoded, png = cv2.imencode(".png", render_temperature_scale())
        if not encoded:
            raise RuntimeError("OpenCV could not encode the temperature scale")
        _TEMPERATURE_SCALE_PNG = png.tobytes()
    return _TEMPERATURE_SCALE_PNG



def create_app(camera: Camera | None = None) -> Flask:
    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)
    thermal_camera = (
        camera if camera is not None else Mlx90640Camera(logger=app.logger)
    )
    stream = FrameBroadcaster(thermal_camera, logger=app.logger)
    recorder = Recorder(stream, logger=app.logger)
    app.extensions["thermal_stream"] = stream
    app.extensions["thermal_recorder"] = recorder

    @app.get("/")
    def index():
        return """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>IR Camera</title>
            <style>
              :root {
                color-scheme: dark;
                font-family: system-ui, sans-serif;
              }
              body {
                margin: 0;
                background: #111827;
                color: #f9fafb;
              }
              main {
                width: min(94vw, 760px);
                margin: 32px auto;
              }
              h1 {
                margin: 0 0 6px;
              }
              .page-link {
                margin: 0 0 16px;
                font-size: 0.9rem;
              }
              .page-link a {
                color: #38bdf8;
              }
              .status {
                border: 3px solid;
                border-radius: 8px;
                margin-bottom: 16px;
                padding: 16px 18px;
                font-size: 1.15rem;
                font-weight: 800;
                letter-spacing: 0.03em;
              }
              .status.online {
                border-color: #22c55e;
                background: #052e16;
                color: #bbf7d0;
              }
              .status.connecting {
                border-color: #f59e0b;
                background: #451a03;
                color: #fde68a;
              }
              .status.offline {
                border-color: #ef4444;
                background: #450a0a;
                color: #fecaca;
              }
              .camera {
                position: relative;
                overflow: hidden;
                border: 2px solid #374151;
                border-radius: 8px;
                background: #030712;
                aspect-ratio: 4 / 3;
              }
              .camera img {
                display: block;
                width: 100%;
                height: 100%;
                object-fit: contain;
              }
              .pixel-readout {
                position: absolute;
                transform: translate(-50%, -140%);
                padding: 4px 8px;
                border: 1px solid #e5e7eb;
                border-radius: 4px;
                background: rgb(17 24 39 / 92%);
                color: #f9fafb;
                font-size: 0.8rem;
                font-variant-numeric: tabular-nums;
                white-space: nowrap;
                pointer-events: none;
              }
              .pixel-readout[hidden] {
                display: none;
              }
              .measure-overlay {
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
              }
              .measure-overlay text {
                font-family: system-ui, sans-serif;
                font-size: 1.1px;
                font-weight: 700;
                fill: #f9fafb;
                paint-order: stroke;
                stroke: #111827;
                stroke-width: 0.3px;
              }
              .measure-overlay text.cell-value {
                font-size: 0.5px;
                font-weight: 600;
                stroke-width: 0.1px;
              }
              .measure {
                margin-top: 16px;
              }
              .measure-tools {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                align-items: center;
              }
              .tool {
                padding: 8px 14px;
                border: 2px solid #374151;
                border-radius: 6px;
                background: #0b1220;
                color: #f9fafb;
                font-weight: 700;
                font-size: 0.9rem;
                cursor: pointer;
              }
              .tool.is-active {
                border-color: #38bdf8;
                background: #082f49;
              }
              #measure-hint {
                color: #9ca3af;
                font-size: 0.85rem;
              }
              .measure-table {
                margin-top: 12px;
                border-collapse: collapse;
                font-variant-numeric: tabular-nums;
              }
              .measure-table th,
              .measure-table td {
                padding: 6px 14px 6px 0;
                text-align: left;
                font-size: 0.9rem;
              }
              .measure-table th {
                color: #9ca3af;
                font-size: 0.75rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                font-weight: 600;
              }
              .measure-table td {
                border-top: 1px solid #1f2937;
              }
              .template-picker {
                display: flex;
                gap: 8px;
                align-items: center;
                font-size: 0.75rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #9ca3af;
              }
              .template-picker select {
                padding: 8px 10px;
                border: 2px solid #374151;
                border-radius: 6px;
                background: #0b1220;
                color: #f9fafb;
                font-size: 0.85rem;
                letter-spacing: normal;
                text-transform: none;
                cursor: pointer;
              }
              .metrics-column {
                display: flex;
                flex-direction: column;
                gap: 10px;
                min-width: 0;
              }
              .unit-toggle {
                display: flex;
                border: 2px solid #374151;
                border-radius: 6px;
                overflow: hidden;
                background: #0b1220;
              }
              .unit-option {
                flex: 1 1 0;
                padding: 8px 0;
                border: 0;
                background: transparent;
                color: #9ca3af;
                font-weight: 700;
                font-size: 0.95rem;
                cursor: pointer;
              }
              .unit-option.is-active {
                background: #082f49;
                color: #f9fafb;
                box-shadow: inset 0 0 0 2px #38bdf8;
              }
              .template-picker input {
                padding: 8px 10px;
                border: 2px solid #374151;
                border-radius: 6px;
                background: #0b1220;
                color: #f9fafb;
                font-size: 0.85rem;
                letter-spacing: normal;
                text-transform: none;
                width: 11rem;
              }
              #camera-frame {
                cursor: crosshair;
              }
              .camera.is-offline img {
                visibility: hidden;
              }
              .camera-offline {
                position: absolute;
                inset: 0;
                display: grid;
                place-content: center;
                gap: 12px;
                padding: 32px;
                background: rgb(69 10 10 / 92%);
                color: white;
                text-align: center;
              }
              .camera-offline[hidden] {
                display: none;
              }
              .camera-offline strong {
                font-size: clamp(2rem, 8vw, 4rem);
                line-height: 1;
              }
              .camera-offline span {
                font-size: 1.1rem;
              }
            </style>
            <style>
              .layout {
                display: grid;
                grid-template-columns: minmax(0, 1fr) 180px;
                gap: 16px;
                align-items: start;
              }
              @media (max-width: 640px) {
                .layout {
                  grid-template-columns: minmax(0, 1fr);
                }
              }
              .metrics {
                border: 2px solid #374151;
                border-radius: 8px;
                padding: 14px;
                background: #0b1220;
              }
              .metric + .metric {
                margin-top: 14px;
              }
              .metric-label {
                font-size: 0.75rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #9ca3af;
              }
              .metric-value {
                font-size: 2rem;
                font-weight: 800;
                font-variant-numeric: tabular-nums;
              }
              .metric-unit {
                font-size: 0.9rem;
                color: #9ca3af;
                margin-left: 4px;
              }
              .feed {
                display: flex;
                gap: 10px;
                align-items: stretch;
              }
              .feed .camera {
                flex: 1 1 auto;
                min-width: 0;
              }
              .scale {
                display: flex;
                gap: 6px;
                margin: 0;
                align-self: stretch;
              }
              .scale img {
                display: block;
                width: 22px;
                height: 100%;
                border: 1px solid #374151;
                border-radius: 4px;
              }
              .scale-ticks {
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                font-size: 0.75rem;
                line-height: 1;
                color: #9ca3af;
                font-variant-numeric: tabular-nums;
                white-space: nowrap;
              }
              .scale-ticks span:first-child {
                margin-top: -0.5em;
              }
              .scale-ticks span:last-child {
                margin-bottom: -0.5em;
              }
              .recorder {
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                align-items: center;
                margin-top: 16px;
              }
              .recorder fieldset {
                display: flex;
                flex-wrap: wrap;
                flex-basis: 100%;
                min-width: 0;
                gap: 12px;
                align-items: center;
                margin: 0;
                padding: 8px 12px;
                border: 2px solid #374151;
                border-radius: 6px;
              }
              .recorder legend {
                padding: 0 6px;
                font-size: 0.75rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #9ca3af;
              }
              .recorder label {
                display: flex;
                gap: 6px;
                align-items: center;
              }
              .recorder button {
                padding: 10px 16px;
                border-radius: 6px;
                font-weight: 700;
                font-size: 0.95rem;
                color: #f9fafb;
                cursor: pointer;
              }
              #start-recording {
                border: 2px solid #22c55e;
                background: #052e16;
              }
              #stop-recording {
                border: 2px solid #f59e0b;
                background: #451a03;
              }
              #download-recording {
                border: 2px solid #38bdf8;
                background: #082f49;
              }
              button[disabled] {
                opacity: 0.45;
                cursor: not-allowed;
              }
              #recording-state {
                margin: 0;
                color: #9ca3af;
              }
            </style>
          </head>
          <body>
            <main>
              <h1>Live Thermal Camera Feed</h1>
              <p class="page-link"><a href="viewer">Open a saved recording</a></p>
              <div id="camera-status" class="status connecting"
                   role="alert" aria-live="assertive">
                CONNECTING TO CAMERA
              </div>
              <div class="layout">
                <div class="feed">
                  <div id="camera-frame" class="camera">
                    <img id="camera-image" src="video_feed" width="32" height="24"
                         alt="Live MLX90640 thermal camera feed">
                    <svg id="measure-overlay" class="measure-overlay"
                         viewBox="0 0 32 24" preserveAspectRatio="none"
                         aria-hidden="true"></svg>
                    <div id="pixel-readout" class="pixel-readout" hidden></div>
                    <div id="camera-offline" class="camera-offline" hidden>
                      <strong>CAMERA OFFLINE</strong>
                      <span id="camera-error">
                        MLX90640 at I2C address 0x33 is not responding.
                      </span>
                    </div>
                  </div>
                  <figure class="scale">
                    <img src="temperature_scale.png" width="22" height="480"
                         alt="Inferno color scale from the hottest to the coldest displayed temperature">
                    <figcaption id="scale-ticks" class="scale-ticks"></figcaption>
                  </figure>
                </div>
                <div class="metrics-column">
                  <div class="unit-toggle" role="group"
                       aria-label="Temperature unit">
                    <button id="unit-celsius" type="button"
                            class="unit-option is-active">C</button>
                    <button id="unit-fahrenheit" type="button"
                            class="unit-option">F</button>
                  </div>
                  <aside class="metrics" aria-label="Stream metrics">
                  <div class="metric">
                    <div class="metric-label">Capture rate</div>
                    <div>
                      <span id="fps-value" class="metric-value">--</span
                      ><span class="metric-unit">fps</span>
                    </div>
                  </div>
                  <div class="metric">
                    <div class="metric-label">Resolution</div>
                    <div>
                      <span class="metric-value">32 × 24</span
                      ><span class="metric-unit">px</span>
                    </div>
                  </div>
                  <div class="metric">
                    <div class="metric-label">Average</div>
                    <div>
                      <span id="average-value" class="metric-value">--</span
                      ><span class="metric-unit unit-label">C</span>
                    </div>
                  </div>
                  <div class="metric">
                    <div class="metric-label">Cursor</div>
                    <div>
                      <span id="cursor-value" class="metric-value">--</span
                      ><span class="metric-unit unit-label">C</span>
                    </div>
                  </div>
                  </aside>
                </div>
              </div>
              <section class="measure" aria-label="Measurement tools">
                <div class="measure-tools">
                  <button id="tool-point" type="button" class="tool is-active">Point</button>
                  <button id="tool-rect" type="button" class="tool">Rectangle</button>
                  <button id="tool-circle" type="button" class="tool">Circle</button>
                  <button id="clear-measurements" type="button" class="tool">Clear</button>
                  <button id="toggle-values" type="button" class="tool">Numbers</button>
                  <label class="template-picker">
                    Saved layouts
                    <select id="template-select">
                      <option value="">No saved layouts</option>
                    </select>
                    <button id="template-load" type="button" class="tool">Load</button>
                    <button id="template-delete" type="button" class="tool">Delete</button>
                  </label>
                  <label class="template-picker">
                    Save current
                    <input id="template-name" type="text" maxlength="40"
                           placeholder="Layout name">
                    <button id="template-save" type="button" class="tool">Save</button>
                  </label>
                  <span id="measure-hint">Click for a point; drag for a rectangle or circle.</span>
                </div>
                <table class="measure-table">
                  <thead>
                    <tr>
                      <th>Marker</th>
                      <th>Min <span class="unit-label">C</span></th>
                      <th>Max <span class="unit-label">C</span></th>
                      <th>Avg <span class="unit-label">C</span></th>
                    </tr>
                  </thead>
                  <tbody id="measurement-rows">
                    <tr id="measurement-empty">
                      <td colspan="4">No markers yet.</td>
                    </tr>
                  </tbody>
                </table>
              </section>
              <section class="recorder" aria-label="Data collection">
                <fieldset>
                  <legend>Format</legend>
                  <label>
                    <input type="radio" name="capture-format" value="images" checked>
                    Images (JPEG)
                  </label>
                  <label>
                    <input type="radio" name="capture-format" value="json">
                    Numbers (JSON)
                  </label>
                  <label>
                    <input type="radio" name="capture-format" value="both">
                    Both (JPEG + JSON)
                  </label>
                </fieldset>
                <button id="start-recording" type="button">Start collecting</button>
                <button id="stop-recording" type="button" disabled>Stop collecting</button>
                <button id="download-recording" type="button" disabled>Download</button>
                <p id="recording-state">Not recording.</p>
              </section>
            </main>
            <script>
              const banner = document.getElementById("camera-status");
              const offline = document.getElementById("camera-offline");
              const error = document.getElementById("camera-error");
              const frame = document.getElementById("camera-frame");
              const image = document.getElementById("camera-image");
              let wasOffline = false;
              const fpsValue = document.getElementById("fps-value");
              const averageValue = document.getElementById("average-value");
              const cursorValue = document.getElementById("cursor-value");
              const pixelReadout = document.getElementById("pixel-readout");
              const startButton = document.getElementById("start-recording");
              const stopButton = document.getElementById("stop-recording");
              const downloadButton =
                document.getElementById("download-recording");
              const recordingState = document.getElementById("recording-state");
              const formatInputs = Array.from(
                document.querySelectorAll('input[name="capture-format"]')
              );
              const celsiusButton = document.getElementById("unit-celsius");
              const fahrenheitButton = document.getElementById("unit-fahrenheit");
              const unitLabels = Array.from(
                document.querySelectorAll(".unit-label")
              );
              const scaleTicks = document.getElementById("scale-ticks");
              const templateSelect = document.getElementById("template-select");
              // The server owns the color map range; the page mirrors whatever
              // /camera_status reports so the ticks always match the image.
              let scaleMinCelsius = null;
              let scaleMaxCelsius = null;

              // The sensor and every download stay in Celsius; the unit choice
              // only changes what this page displays.
              let showFahrenheit =
                window.localStorage.getItem("ir-camera-unit") === "F";

              function toDisplay(celsius) {
                return showFahrenheit ? celsius * 9 / 5 + 32 : celsius;
              }

              function formatTemperature(celsius) {
                return toDisplay(celsius).toFixed(1);
              }

              function unitSymbol() {
                return showFahrenheit ? "F" : "C";
              }

              function renderScaleTicks() {
                if (scaleMinCelsius === null || scaleMaxCelsius === null) {
                  scaleTicks.innerHTML = "";
                  return;
                }
                const steps = 6;
                const span = scaleMaxCelsius - scaleMinCelsius;
                const labels = [];
                for (let step = 0; step <= steps; step += 1) {
                  const celsius = scaleMaxCelsius - (step * span) / steps;
                  const value = Math.round(toDisplay(celsius));
                  const edge = step === 0 || step === steps;
                  labels.push(
                    `<span>${value}${edge ? " " + unitSymbol() : ""}</span>`
                  );
                }
                scaleTicks.innerHTML = labels.join("");
              }

              let averageCelsius = null;

              function renderAverage() {
                averageValue.textContent =
                  averageCelsius === null ? "--" : formatTemperature(averageCelsius);
              }

              function applyUnit() {
                window.localStorage.setItem(
                  "ir-camera-unit",
                  showFahrenheit ? "F" : "C"
                );
                celsiusButton.classList.toggle("is-active", !showFahrenheit);
                celsiusButton.setAttribute("aria-pressed", String(!showFahrenheit));
                fahrenheitButton.classList.toggle("is-active", showFahrenheit);
                fahrenheitButton.setAttribute("aria-pressed", String(showFahrenheit));
                unitLabels.forEach((label) => {
                  label.textContent = unitSymbol();
                });
                renderScaleTicks();
                // Re-render the cached Celsius average immediately; waiting for
                // the next status poll would briefly show a Celsius number
                // under the new unit label.
                renderAverage();
                updatePixelReadout();
                renderMeasurements();
                renderOverlay();
                refreshCameraStatus();
              }

              celsiusButton.addEventListener("click", () => {
                if (showFahrenheit) {
                  showFahrenheit = false;
                  applyUnit();
                }
              });
              fahrenheitButton.addEventListener("click", () => {
                if (!showFahrenheit) {
                  showFahrenheit = true;
                  applyUnit();
                }
              });


              async function refreshCameraStatus() {
                try {
                  const response = await fetch("camera_status", {
                    cache: "no-store",
                  });
                  if (!response.ok) {
                    throw new Error(`status request returned ${response.status}`);
                  }
                  const status = await response.json();
                  banner.className = `status ${status.state}`;
                  banner.textContent = `${status.message} - ${status.detail}`;
                  const isOffline = status.state === "offline";
                  offline.hidden = !isOffline;
                  frame.classList.toggle("is-offline", isOffline);
                  // The MJPEG response ended while the sensor was down; ask
                  // the browser for a fresh stream once frames return.
                  if (wasOffline && !isOffline) {
                    image.src = `video_feed?retry=${Date.now()}`;
                  }
                  wasOffline = isOffline;
                  error.textContent = status.detail;
                  fpsValue.textContent =
                    status.fps === null ? "--" : status.fps.toFixed(1);
                  averageCelsius = status.average_celsius;
                  renderAverage();
                  if (
                    status.scale_min_celsius !== scaleMinCelsius ||
                    status.scale_max_celsius !== scaleMaxCelsius
                  ) {
                    scaleMinCelsius = status.scale_min_celsius;
                    scaleMaxCelsius = status.scale_max_celsius;
                    renderScaleTicks();
                  }
                  // Viewer counts stay in the status payload for the server
                  // side viewer cap; the page no longer shows them.
                } catch (requestError) {
                  banner.className = "status offline";
                  banner.textContent =
                    `CAMERA STATUS UNAVAILABLE - ${requestError.message}`;
                  offline.hidden = false;
                  frame.classList.add("is-offline");
                  wasOffline = true;
                  error.textContent =
                    "The application cannot report the MLX90640 status.";
                  fpsValue.textContent = "--";
                  averageCelsius = null;
                  renderAverage();
                }
              }

              let temperatureGrid = null;
              let cursorPosition = null;

              async function refreshTemperatureGrid() {
                try {
                  const response = await fetch("latest_frame.json", {
                    cache: "no-store",
                  });
                  if (!response.ok) {
                    throw new Error(`frame request returned ${response.status}`);
                  }
                  const payload = await response.json();
                  temperatureGrid = payload.temperatures;
                } catch (requestError) {
                  temperatureGrid = null;
                }
                updatePixelReadout();
                renderMeasurements();
                renderOverlay();
              }

              function updatePixelReadout() {
                if (cursorPosition === null || temperatureGrid === null) {
                  pixelReadout.hidden = true;
                  cursorValue.textContent = "--";
                  return;
                }
                const rows = temperatureGrid.length;
                const columns = temperatureGrid[0].length;
                const column = Math.min(
                  columns - 1,
                  Math.floor(cursorPosition.fx * columns)
                );
                const row = Math.min(
                  rows - 1,
                  Math.floor(cursorPosition.fy * rows)
                );
                const celsius = temperatureGrid[row][column];
                cursorValue.textContent = formatTemperature(celsius);
                pixelReadout.hidden = false;
                pixelReadout.style.left = `${cursorPosition.x}px`;
                pixelReadout.style.top = `${cursorPosition.y}px`;
                pixelReadout.textContent =
                  `${formatTemperature(celsius)} ${unitSymbol()}` +
                  `  (x ${column}, y ${row})`;
              }

              image.addEventListener("mousemove", (event) => {
                cursorPosition = positionFromEvent(event);
                if (draft !== null) {
                  draft.x2 = cursorPosition.sx;
                  draft.y2 = cursorPosition.sy;
                  renderOverlay();
                }
                updatePixelReadout();
              });
              image.addEventListener("mouseleave", () => {
                cursorPosition = null;
                updatePixelReadout();
              });

              // Measurement markers: P# points, R# rectangles and C# circles.
              // All geometry is kept in sensor cell coordinates so the overlay
              // and the statistics stay correct at any display size.
              const overlay = document.getElementById("measure-overlay");
              const measurementRows = document.getElementById("measurement-rows");
              const toolButtons = {
                point: document.getElementById("tool-point"),
                rect: document.getElementById("tool-rect"),
                circle: document.getElementById("tool-circle"),
              };
              const clearButton = document.getElementById("clear-measurements");
              const markers = [];
              const counters = { point: 0, rect: 0, circle: 0 };
              let activeTool = "point";
              let draft = null;

              function positionFromEvent(event) {
                const bounds = image.getBoundingClientRect();
                const x = event.clientX - bounds.left;
                const y = event.clientY - bounds.top;
                const fx = Math.min(0.999999, Math.max(0, x / bounds.width));
                const fy = Math.min(0.999999, Math.max(0, y / bounds.height));
                return { x, y, fx, fy, sx: fx * 32, sy: fy * 24 };
              }

              function cellAt(column, row) {
                if (temperatureGrid === null) {
                  return null;
                }
                const rows = temperatureGrid.length;
                const columns = temperatureGrid[0].length;
                const r = Math.min(rows - 1, Math.max(0, Math.floor(row)));
                const c = Math.min(columns - 1, Math.max(0, Math.floor(column)));
                return temperatureGrid[r][c];
              }

              function summarize(values) {
                if (values.length === 0) {
                  return null;
                }
                let min = values[0];
                let max = values[0];
                let total = 0;
                for (const value of values) {
                  min = Math.min(min, value);
                  max = Math.max(max, value);
                  total += value;
                }
                return { min, max, avg: total / values.length };
              }

              function markerValues(marker) {
                if (temperatureGrid === null) {
                  return null;
                }
                if (marker.kind === "point") {
                  const value = cellAt(marker.x1, marker.y1);
                  return value === null ? null : [value];
                }
                if (marker.kind === "circle") {
                  const radius = Math.hypot(
                    marker.x2 - marker.x1,
                    marker.y2 - marker.y1
                  );
                  const values = [];
                  const left = Math.floor(marker.x1 - radius);
                  const right = Math.ceil(marker.x1 + radius);
                  const top = Math.floor(marker.y1 - radius);
                  const bottom = Math.ceil(marker.y1 + radius);
                  for (let row = top; row <= bottom; row += 1) {
                    for (let column = left; column <= right; column += 1) {
                      // Compare cell centres so a circle covers the cells it
                      // visually encloses.
                      const dx = column + 0.5 - marker.x1;
                      const dy = row + 0.5 - marker.y1;
                      if (Math.hypot(dx, dy) <= radius) {
                        values.push(cellAt(column, row));
                      }
                    }
                  }
                  return values.length > 0
                    ? values
                    : [cellAt(marker.x1, marker.y1)];
                }
                const left = Math.floor(Math.min(marker.x1, marker.x2));
                const right = Math.floor(Math.max(marker.x1, marker.x2));
                const top = Math.floor(Math.min(marker.y1, marker.y2));
                const bottom = Math.floor(Math.max(marker.y1, marker.y2));
                const values = [];
                for (let row = top; row <= bottom; row += 1) {
                  for (let column = left; column <= right; column += 1) {
                    values.push(cellAt(column, row));
                  }
                }
                return values;
              }

              // Per-cell numbers are dense at 32 x 24, so they are rounded to
              // whole degrees and only drawn when the operator asks for them.
              let showCellValues = false;
              const valuesButton = document.getElementById("toggle-values");

              function renderCellValues() {
                if (!showCellValues || temperatureGrid === null) {
                  return "";
                }
                const parts = [];
                for (let row = 0; row < temperatureGrid.length; row += 1) {
                  for (let column = 0; column < temperatureGrid[row].length; column += 1) {
                    const value = Math.round(
                      toDisplay(temperatureGrid[row][column])
                    );
                    parts.push(
                      `<text class="cell-value" x="${column + 0.5}"` +
                      ` y="${row + 0.68}" text-anchor="middle">${value}</text>`
                    );
                  }
                }
                return parts.join("");
              }

              valuesButton.addEventListener("click", () => {
                showCellValues = !showCellValues;
                valuesButton.classList.toggle("is-active", showCellValues);
                renderOverlay();
              });

              function renderOverlay() {
                const shapes = draft === null ? markers : markers.concat([draft]);
                overlay.innerHTML = renderCellValues() + shapes
                  .map((marker) => {
                    const label = marker.id || "";
                    if (marker.kind === "point") {
                      return `<circle cx="${marker.x1}" cy="${marker.y1}" r="0.7"` +
                        ` fill="none" stroke="#f9fafb" stroke-width="1.5"` +
                        ` vector-effect="non-scaling-stroke"></circle>` +
                        `<circle cx="${marker.x1}" cy="${marker.y1}" r="0.12"` +
                        ` fill="#f9fafb"></circle>` +
                        `<text x="${marker.x1 + 1}" y="${marker.y1 - 0.8}">${label}</text>`;
                    }
                    if (marker.kind === "circle") {
                      const radius = Math.hypot(
                        marker.x2 - marker.x1,
                        marker.y2 - marker.y1
                      );
                      return `<circle cx="${marker.x1}" cy="${marker.y1}"` +
                        ` r="${radius}" fill="none" stroke="#f9fafb"` +
                        ` stroke-width="1.5" vector-effect="non-scaling-stroke"></circle>` +
                        `<text x="${marker.x1 - radius + 0.4}"` +
                        ` y="${marker.y1 - radius - 0.6}">${label}</text>`;
                    }
                    const x = Math.min(marker.x1, marker.x2);
                    const y = Math.min(marker.y1, marker.y2);
                    const width = Math.abs(marker.x2 - marker.x1);
                    const height = Math.abs(marker.y2 - marker.y1);
                    return `<rect x="${x}" y="${y}" width="${width}" height="${height}"` +
                      ` fill="none" stroke="#f9fafb" stroke-width="1.5"` +
                      ` vector-effect="non-scaling-stroke"></rect>` +
                      `<text x="${x + 0.4}" y="${y - 0.6}">${label}</text>`;
                  })
                  .join("");
              }

              function renderMeasurements() {
                if (markers.length === 0) {
                  measurementRows.innerHTML =
                    '<tr id="measurement-empty"><td colspan="4">No markers yet.</td></tr>';
                  return;
                }
                measurementRows.innerHTML = markers
                  .map((marker) => {
                    const summary = summarize(markerValues(marker) || []);
                    if (summary === null) {
                      return `<tr><td>${marker.id}</td><td>--</td><td>--</td><td>--</td></tr>`;
                    }
                    return `<tr><td>${marker.id}</td>` +
                      `<td>${formatTemperature(summary.min)}</td>` +
                      `<td>${formatTemperature(summary.max)}</td>` +
                      `<td>${formatTemperature(summary.avg)}</td></tr>`;
                  })
                  .join("");
              }

              function selectTool(tool) {
                activeTool = tool;
                Object.entries(toolButtons).forEach(([name, button]) => {
                  button.classList.toggle("is-active", name === tool);
                });
              }

              Object.entries(toolButtons).forEach(([name, button]) => {
                button.addEventListener("click", () => selectTool(name));
              });
              clearButton.addEventListener("click", () => {
                markers.length = 0;
                counters.point = 0;
                counters.rect = 0;
                counters.circle = 0;
                draft = null;
                renderOverlay();
                renderMeasurements();
              });

              image.addEventListener("mousedown", (event) => {
                event.preventDefault();
                const position = positionFromEvent(event);
                if (activeTool === "point") {
                  counters.point += 1;
                  markers.push({
                    kind: "point",
                    id: `P${counters.point}`,
                    x1: position.sx,
                    y1: position.sy,
                  });
                  renderOverlay();
                  renderMeasurements();
                  return;
                }
                draft = {
                  kind: activeTool,
                  id: "",
                  x1: position.sx,
                  y1: position.sy,
                  x2: position.sx,
                  y2: position.sy,
                };
              });

              window.addEventListener("mouseup", () => {
                if (draft === null) {
                  return;
                }
                const pending = draft;
                draft = null;
                const spread = Math.hypot(
                  pending.x2 - pending.x1,
                  pending.y2 - pending.y1
                );
                if (spread >= 1) {
                  if (pending.kind === "circle") {
                    counters.circle += 1;
                    pending.id = `C${counters.circle}`;
                  } else {
                    counters.rect += 1;
                    pending.id = `R${counters.rect}`;
                  }
                  markers.push(pending);
                }
                renderOverlay();
                renderMeasurements();
              });

              // Named layouts the operator saves. Marker geometry is stored in
              // sensor cell coordinates, so a layout lands on the same part of
              // the scene whatever the browser window size is.
              const TEMPLATE_STORAGE_KEY = "ir-camera-templates";
              const templateName = document.getElementById("template-name");
              const templateSave = document.getElementById("template-save");
              const templateLoad = document.getElementById("template-load");
              const templateDelete = document.getElementById("template-delete");
              const measureHint = document.getElementById("measure-hint");
              const DEFAULT_HINT = measureHint.textContent;

              function readTemplates() {
                try {
                  const stored = JSON.parse(
                    window.localStorage.getItem(TEMPLATE_STORAGE_KEY) || "{}"
                  );
                  return stored && typeof stored === "object" ? stored : {};
                } catch (storageError) {
                  return {};
                }
              }

              function writeTemplates(templates) {
                window.localStorage.setItem(
                  TEMPLATE_STORAGE_KEY,
                  JSON.stringify(templates)
                );
              }

              function escapeHtml(value) {
                return value
                  .replace(/&/g, "&amp;")
                  .replace(/</g, "&lt;")
                  .replace(/>/g, "&gt;")
                  .replace(/"/g, "&quot;");
              }

              function renderTemplateOptions(selected) {
                const names = Object.keys(readTemplates()).sort((a, b) =>
                  a.localeCompare(b)
                );
                const placeholder = names.length
                  ? "Choose a layout"
                  : "No saved layouts";
                templateSelect.innerHTML =
                  `<option value="">${placeholder}</option>` +
                  names
                    .map((name) => {
                      const safe = escapeHtml(name);
                      return `<option value="${safe}">${safe}</option>`;
                    })
                    .join("");
                if (selected && names.includes(selected)) {
                  templateSelect.value = selected;
                }
                templateLoad.disabled = names.length === 0;
                templateDelete.disabled = names.length === 0;
              }

              templateSave.addEventListener("click", () => {
                const name = templateName.value.trim();
                if (!name) {
                  measureHint.textContent = "Name the layout before saving it.";
                  return;
                }
                if (markers.length === 0) {
                  measureHint.textContent =
                    "Place at least one marker before saving a layout.";
                  return;
                }
                const templates = readTemplates();
                const replacing = Object.prototype.hasOwnProperty.call(
                  templates,
                  name
                );
                templates[name] = markers.map((marker) => ({
                  kind: marker.kind,
                  x1: marker.x1,
                  y1: marker.y1,
                  x2: marker.x2,
                  y2: marker.y2,
                }));
                writeTemplates(templates);
                renderTemplateOptions(name);
                templateName.value = "";
                measureHint.textContent =
                  `${replacing ? "Replaced" : "Saved"} "${name}" with ` +
                  `${markers.length} marker(s).`;
              });

              templateLoad.addEventListener("click", () => {
                const name = templateSelect.value;
                const saved = readTemplates()[name];
                if (!saved) {
                  measureHint.textContent = "Choose a saved layout first.";
                  return;
                }
                markers.length = 0;
                counters.point = 0;
                counters.rect = 0;
                counters.circle = 0;
                draft = null;
                saved.forEach((marker) => {
                  if (marker.kind === "point") {
                    counters.point += 1;
                    markers.push({ ...marker, id: `P${counters.point}` });
                  } else if (marker.kind === "circle") {
                    counters.circle += 1;
                    markers.push({ ...marker, id: `C${counters.circle}` });
                  } else {
                    counters.rect += 1;
                    markers.push({ ...marker, id: `R${counters.rect}` });
                  }
                });
                renderOverlay();
                renderMeasurements();
                measureHint.textContent =
                  `Loaded "${name}" with ${markers.length} marker(s).`;
              });

              templateDelete.addEventListener("click", () => {
                const name = templateSelect.value;
                const templates = readTemplates();
                if (!name || !(name in templates)) {
                  measureHint.textContent = "Choose a saved layout first.";
                  return;
                }
                delete templates[name];
                writeTemplates(templates);
                renderTemplateOptions();
                measureHint.textContent = `Deleted "${name}". ${DEFAULT_HINT}`;
              });

              renderTemplateOptions();

              function selectedFormat() {
                const checked = formatInputs.find((input) => input.checked);
                return checked ? checked.value : "images";
              }

              async function postRecording(path) {
                try {
                  const response = await fetch(path, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ format: selectedFormat() }),
                  });
                  if (!response.ok) {
                    const failure = await response.json();
                    recordingState.textContent =
                      failure.error || `Request failed (${response.status}).`;
                  }
                } catch (requestError) {
                  recordingState.textContent = requestError.message;
                }
                await refreshRecording();
              }

              async function refreshRecording() {
                let status;
                try {
                  const response = await fetch("recording/status", {
                    cache: "no-store",
                  });
                  if (!response.ok) {
                    throw new Error(`status request returned ${response.status}`);
                  }
                  status = await response.json();
                } catch (requestError) {
                  recordingState.textContent = "Recording status unavailable.";
                  return;
                }
                const isRecording = status.state === "recording";
                startButton.disabled = isRecording;
                stopButton.disabled = !isRecording;
                downloadButton.disabled =
                  !(status.state === "ready" && status.frames > 0);
                formatInputs.forEach((input) => {
                  input.disabled = isRecording;
                });
                const limit = status.limit_reached ? " (frame limit reached)" : "";
                if (isRecording) {
                  recordingState.textContent =
                    `Recording ${status.format} - ${status.frames} frames collected${limit}`;
                } else if (status.state === "ready") {
                  recordingState.textContent =
                    `Ready - ${status.frames} frames collected, press Download${limit}`;
                } else {
                  recordingState.textContent = "Not recording.";
                }
              }

              startButton.addEventListener("click", () => {
                postRecording("recording/start");
              });
              stopButton.addEventListener("click", () => {
                postRecording("recording/stop");
              });
              downloadButton.addEventListener("click", () => {
                // Content-Disposition on the response drives the save dialog.
                window.location.href = "recording/download";
              });

              applyUnit();
              refreshCameraStatus();
              refreshRecording();
              refreshTemperatureGrid();
              window.setInterval(() => {
                refreshCameraStatus();
                refreshRecording();
              }, 1000);
              // The sensor publishes about 2 full frames/s; poll at the same rate so
              // the hover readout follows the picture.
              window.setInterval(refreshTemperatureGrid, 500);
            </script>
          </body>
        </html>
        """

    @app.get("/viewer")
    def viewer():
        """Replay a downloaded recording. The file never leaves the browser."""
        return """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>IR Camera recording viewer</title>
            <style>
              :root { color-scheme: dark; font-family: system-ui, sans-serif; }
              body { margin: 0; background: #111827; color: #f9fafb; }
              main { width: min(94vw, 980px); margin: 32px auto; }
              h1 { margin: 0 0 6px; }
              .lede { margin: 0 0 18px; color: #9ca3af; }
              a { color: #38bdf8; }
              .loader {
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                align-items: center;
                margin-bottom: 16px;
              }
              .playback {
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                align-items: center;
                margin-top: 14px;
              }
              button, select, input[type="file"] {
                padding: 9px 14px;
                border: 2px solid #374151;
                border-radius: 6px;
                background: #0b1220;
                color: #f9fafb;
                font-weight: 600;
                font-size: 0.9rem;
                cursor: pointer;
              }
              #play { border-color: #22c55e; background: #052e16; }
              button[disabled] { opacity: 0.45; cursor: not-allowed; }
              #frame-slider { flex: 1 1 240px; min-width: 180px; padding: 0; }
              .viewer-layout {
                display: grid;
                grid-template-columns: minmax(0, 1fr) 190px;
                gap: 16px;
                align-items: start;
              }
              @media (max-width: 680px) {
                .viewer-layout { grid-template-columns: minmax(0, 1fr); }
              }
              .stage { display: flex; gap: 10px; align-items: stretch; }
              .canvas-wrap {
                position: relative;
                flex: 1 1 auto;
                min-width: 0;
                border: 2px solid #374151;
                border-radius: 8px;
                overflow: hidden;
                background: #030712;
                aspect-ratio: 4 / 3;
                cursor: crosshair;
              }
              #frame-canvas, #frame-image {
                /* Absolute so the hidden element never stretches the stage and
                   the wrapper's aspect-ratio alone sets the height. */
                position: absolute;
                inset: 0;
                display: block;
                width: 100%;
                height: 100%;
                object-fit: contain;
              }
              #frame-canvas[hidden], #frame-image[hidden] {
                display: none;
              }
              .pixel-readout {
                position: absolute;
                transform: translate(-50%, -140%);
                padding: 4px 8px;
                border: 1px solid #e5e7eb;
                border-radius: 4px;
                background: rgb(17 24 39 / 92%);
                font-size: 0.8rem;
                font-variant-numeric: tabular-nums;
                white-space: nowrap;
                pointer-events: none;
              }
              .pixel-readout[hidden] { display: none; }
              .value-overlay {
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
              }
              .value-overlay text {
                font-family: system-ui, sans-serif;
                font-size: 0.5px;
                font-weight: 600;
                fill: #f9fafb;
                paint-order: stroke;
                stroke: #111827;
                stroke-width: 0.1px;
              }
              #toggle-values.is-active {
                border-color: #38bdf8;
                background: #082f49;
              }
              .scale {
                position: relative;
                display: flex;
                align-self: stretch;
                margin: 0;
                padding-left: 28px;
              }
              .scale canvas {
                /* Absolute so the canvas cannot impose its 1 x 256 intrinsic
                   ratio on the row height. */
                position: absolute;
                left: 0;
                top: 0;
                display: block;
                width: 22px;
                height: 100%;
                border: 1px solid #374151;
                border-radius: 4px;
              }
              .scale-ticks {
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                font-size: 0.75rem;
                line-height: 1;
                color: #9ca3af;
                font-variant-numeric: tabular-nums;
                white-space: nowrap;
              }
              .scale-ticks span:first-child { margin-top: -0.5em; }
              .scale-ticks span:last-child { margin-bottom: -0.5em; }
              .unit-toggle {
                display: flex;
                border: 2px solid #374151;
                border-radius: 6px;
                overflow: hidden;
                background: #0b1220;
                margin-bottom: 10px;
              }
              .unit-option {
                flex: 1 1 0;
                padding: 8px 0;
                border: 0;
                border-radius: 0;
                background: transparent;
                color: #9ca3af;
              }
              .unit-option.is-active {
                background: #082f49;
                color: #f9fafb;
                box-shadow: inset 0 0 0 2px #38bdf8;
              }
              .metrics {
                border: 2px solid #374151;
                border-radius: 8px;
                padding: 14px;
                background: #0b1220;
              }
              .metric + .metric { margin-top: 14px; }
              .metric-label {
                font-size: 0.75rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #9ca3af;
              }
              .metric-value {
                font-size: 1.6rem;
                font-weight: 800;
                font-variant-numeric: tabular-nums;
              }
              .metric-unit { font-size: 0.85rem; color: #9ca3af; margin-left: 4px; }
              #load-state { color: #9ca3af; font-size: 0.9rem; }
              #load-state.is-error { color: #fecaca; }
            </style>
          </head>
          <body>
            <main>
              <h1>Recording viewer</h1>
              <p class="lede">
                Open a downloaded <code>ir-camera-&lt;session&gt;.zip</code> or
                <code>.json</code>. The file is read in this browser and never
                uploaded. <a href="./">Back to the live feed</a>
              </p>
              <div class="loader">
                <input id="file-input" type="file" accept=".zip,.json">
                <span id="load-state">No recording opened.</span>
              </div>
              <div class="viewer-layout">
                <div>
                  <div class="stage">
                    <div id="canvas-wrap" class="canvas-wrap">
                      <canvas id="frame-canvas" width="32" height="24"></canvas>
                      <img id="frame-image" alt="Recorded thermal frame" hidden>
                      <svg id="value-overlay" class="value-overlay"
                           viewBox="0 0 32 24" preserveAspectRatio="none"
                           aria-hidden="true"></svg>
                      <div id="pixel-readout" class="pixel-readout" hidden></div>
                    </div>
                    <figure class="scale">
                      <canvas id="scale-canvas" width="1" height="256"></canvas>
                      <figcaption id="scale-ticks" class="scale-ticks"></figcaption>
                    </figure>
                  </div>
                  <div class="playback">
                    <button id="play" type="button" disabled>Play</button>
                    <button id="previous" type="button" disabled>Previous</button>
                    <button id="next" type="button" disabled>Next</button>
                    <button id="toggle-values" type="button" disabled>Numbers</button>
                    <input id="frame-slider" type="range" min="0" max="0" value="0" disabled>
                    <label>
                      Speed
                      <select id="speed">
                        <option value="0.5">0.5x</option>
                        <option value="1" selected>1x</option>
                        <option value="2">2x</option>
                        <option value="4">4x</option>
                      </select>
                    </label>
                  </div>
                </div>
                <div>
                  <div class="unit-toggle" role="group" aria-label="Temperature unit">
                    <button id="unit-celsius" type="button" class="unit-option is-active">C</button>
                    <button id="unit-fahrenheit" type="button" class="unit-option">F</button>
                  </div>
                  <aside class="metrics" aria-label="Frame metrics">
                    <div class="metric">
                      <div class="metric-label">Frame</div>
                      <div><span id="frame-index" class="metric-value">--</span></div>
                    </div>
                    <div class="metric">
                      <div class="metric-label">Timestamp</div>
                      <div><span id="frame-time" class="metric-value">--</span></div>
                    </div>
                    <div class="metric">
                      <div class="metric-label">Min</div>
                      <div><span id="frame-min" class="metric-value">--</span
                        ><span class="metric-unit unit-label">C</span></div>
                    </div>
                    <div class="metric">
                      <div class="metric-label">Max</div>
                      <div><span id="frame-max" class="metric-value">--</span
                        ><span class="metric-unit unit-label">C</span></div>
                    </div>
                    <div class="metric">
                      <div class="metric-label">Average</div>
                      <div><span id="frame-avg" class="metric-value">--</span
                        ><span class="metric-unit unit-label">C</span></div>
                    </div>
                    <div class="metric">
                      <div class="metric-label">Cursor</div>
                      <div><span id="cursor-value" class="metric-value">--</span
                        ><span class="metric-unit unit-label">C</span></div>
                    </div>
                  </aside>
                </div>
              </div>
            </main>
            <script>
              const fileInput = document.getElementById("file-input");
              const loadState = document.getElementById("load-state");
              const canvas = document.getElementById("frame-canvas");
              const context = canvas.getContext("2d");
              const frameImage = document.getElementById("frame-image");
              const canvasWrap = document.getElementById("canvas-wrap");
              const pixelReadout = document.getElementById("pixel-readout");
              const scaleCanvas = document.getElementById("scale-canvas");
              const scaleTicks = document.getElementById("scale-ticks");
              const playButton = document.getElementById("play");
              const previousButton = document.getElementById("previous");
              const nextButton = document.getElementById("next");
              const valuesButton = document.getElementById("toggle-values");
              const valueOverlay = document.getElementById("value-overlay");
              const slider = document.getElementById("frame-slider");
              const speedSelect = document.getElementById("speed");
              const celsiusButton = document.getElementById("unit-celsius");
              const fahrenheitButton = document.getElementById("unit-fahrenheit");
              const unitLabels = Array.from(document.querySelectorAll(".unit-label"));
              const frameIndexValue = document.getElementById("frame-index");
              const frameTimeValue = document.getElementById("frame-time");
              const frameMinValue = document.getElementById("frame-min");
              const frameMaxValue = document.getElementById("frame-max");
              const frameAvgValue = document.getElementById("frame-avg");
              const cursorValue = document.getElementById("cursor-value");

              let grids = [];
              let imageUrls = [];
              let timeframes = [];
              let rangeMin = 0;
              let rangeMax = 1;
              let current = 0;
              let timer = null;
              let cursorCell = null;
              let palette = null;
              let showFahrenheit =
                window.localStorage.getItem("ir-camera-unit") === "F";

              function toDisplay(celsius) {
                return showFahrenheit ? celsius * 9 / 5 + 32 : celsius;
              }

              function formatTemperature(celsius) {
                return toDisplay(celsius).toFixed(1);
              }

              function unitSymbol() {
                return showFahrenheit ? "F" : "C";
              }

              // Sample the server's own scale image so the viewer and the live
              // page use exactly the same inferno ramp.
              async function loadPalette() {
                const image = new Image();
                image.src = "temperature_scale.png";
                await image.decode();
                const sampler = document.createElement("canvas");
                sampler.width = 1;
                sampler.height = 256;
                const samplerContext = sampler.getContext("2d");
                samplerContext.drawImage(image, 0, 0, 1, 256);
                const pixels = samplerContext.getImageData(0, 0, 1, 256).data;
                // The scale image runs hot at the top; index 0 must be coldest.
                palette = [];
                for (let step = 255; step >= 0; step -= 1) {
                  const offset = step * 4;
                  palette.push([
                    pixels[offset],
                    pixels[offset + 1],
                    pixels[offset + 2],
                  ]);
                }
                const scaleContext = scaleCanvas.getContext("2d");
                for (let step = 0; step < 256; step += 1) {
                  const [r, g, b] = palette[255 - step];
                  scaleContext.fillStyle = `rgb(${r} ${g} ${b})`;
                  scaleContext.fillRect(0, step, 1, 1);
                }
              }

              function renderScaleTicks() {
                if (grids.length === 0) {
                  scaleTicks.innerHTML = "";
                  return;
                }
                const steps = 6;
                const labels = [];
                for (let step = 0; step <= steps; step += 1) {
                  const celsius =
                    rangeMax - (step * (rangeMax - rangeMin)) / steps;
                  const edge = step === 0 || step === steps;
                  labels.push(
                    `<span>${Math.round(toDisplay(celsius))}` +
                    `${edge ? " " + unitSymbol() : ""}</span>`
                  );
                }
                scaleTicks.innerHTML = labels.join("");
              }

              function summarize(grid) {
                let min = Infinity;
                let max = -Infinity;
                let total = 0;
                let count = 0;
                for (const row of grid) {
                  for (const value of row) {
                    min = Math.min(min, value);
                    max = Math.max(max, value);
                    total += value;
                    count += 1;
                  }
                }
                return { min, max, avg: total / count };
              }

              function drawGrid(grid) {
                const rows = grid.length;
                const columns = grid[0].length;
                canvas.width = columns;
                canvas.height = rows;
                const picture = context.createImageData(columns, rows);
                const span = rangeMax - rangeMin || 1;
                for (let row = 0; row < rows; row += 1) {
                  for (let column = 0; column < columns; column += 1) {
                    const ratio = (grid[row][column] - rangeMin) / span;
                    const index = Math.max(
                      0,
                      Math.min(255, Math.round(ratio * 255))
                    );
                    const [r, g, b] = palette[index];
                    const offset = (row * columns + column) * 4;
                    picture.data[offset] = r;
                    picture.data[offset + 1] = g;
                    picture.data[offset + 2] = b;
                    picture.data[offset + 3] = 255;
                  }
                }
                context.putImageData(picture, 0, 0);
              }

              // Per-cell numbers are dense at 32 x 24, so they are rounded to
              // whole degrees and only drawn when the operator asks for them.
              let showCellValues = false;

              function renderCellValues() {
                if (!showCellValues || grids.length === 0) {
                  valueOverlay.innerHTML = "";
                  return;
                }
                const grid = grids[current];
                const parts = [];
                for (let row = 0; row < grid.length; row += 1) {
                  for (let column = 0; column < grid[row].length; column += 1) {
                    const value = Math.round(toDisplay(grid[row][column]));
                    parts.push(
                      `<text x="${column + 0.5}" y="${row + 0.68}"` +
                      ` text-anchor="middle">${value}</text>`
                    );
                  }
                }
                valueOverlay.innerHTML = parts.join("");
              }

              valuesButton.addEventListener("click", () => {
                showCellValues = !showCellValues;
                valuesButton.classList.toggle("is-active", showCellValues);
                renderCellValues();
              });

              function showFrame(index) {
                if (grids.length === 0 && imageUrls.length === 0) {
                  return;
                }
                const total = grids.length || imageUrls.length;
                current = Math.max(0, Math.min(total - 1, index));
                slider.value = String(current);
                frameIndexValue.textContent = `${current + 1} / ${total}`;
                frameTimeValue.textContent = timeframes[current]
                  ? new Date(timeframes[current]).toLocaleTimeString()
                  : "--";
                if (grids.length > 0) {
                  drawGrid(grids[current]);
                  const summary = summarize(grids[current]);
                  frameMinValue.textContent = formatTemperature(summary.min);
                  frameMaxValue.textContent = formatTemperature(summary.max);
                  frameAvgValue.textContent = formatTemperature(summary.avg);
                } else {
                  frameImage.src = imageUrls[current];
                  frameMinValue.textContent = "--";
                  frameMaxValue.textContent = "--";
                  frameAvgValue.textContent = "--";
                }
                renderCellValues();
                updateCursorReadout();
              }

              function updateCursorReadout() {
                if (cursorCell === null || grids.length === 0) {
                  pixelReadout.hidden = true;
                  cursorValue.textContent = "--";
                  return;
                }
                const grid = grids[current];
                const rows = grid.length;
                const columns = grid[0].length;
                const column = Math.min(
                  columns - 1,
                  Math.floor(cursorCell.fx * columns)
                );
                const row = Math.min(rows - 1, Math.floor(cursorCell.fy * rows));
                const celsius = grid[row][column];
                cursorValue.textContent = formatTemperature(celsius);
                pixelReadout.hidden = false;
                pixelReadout.style.left = `${cursorCell.x}px`;
                pixelReadout.style.top = `${cursorCell.y}px`;
                pixelReadout.textContent =
                  `${formatTemperature(celsius)} ${unitSymbol()}` +
                  `  (x ${column}, y ${row})`;
              }

              canvasWrap.addEventListener("mousemove", (event) => {
                const bounds = canvasWrap.getBoundingClientRect();
                const x = event.clientX - bounds.left;
                const y = event.clientY - bounds.top;
                cursorCell = {
                  x,
                  y,
                  fx: Math.min(0.999999, Math.max(0, x / bounds.width)),
                  fy: Math.min(0.999999, Math.max(0, y / bounds.height)),
                };
                updateCursorReadout();
              });
              canvasWrap.addEventListener("mouseleave", () => {
                cursorCell = null;
                updateCursorReadout();
              });

              function applyUnit() {
                window.localStorage.setItem(
                  "ir-camera-unit",
                  showFahrenheit ? "F" : "C"
                );
                celsiusButton.classList.toggle("is-active", !showFahrenheit);
                fahrenheitButton.classList.toggle("is-active", showFahrenheit);
                unitLabels.forEach((label) => {
                  label.textContent = unitSymbol();
                });
                renderScaleTicks();
                if (grids.length > 0 || imageUrls.length > 0) {
                  showFrame(current);
                }
              }

              celsiusButton.addEventListener("click", () => {
                if (showFahrenheit) {
                  showFahrenheit = false;
                  applyUnit();
                }
              });
              fahrenheitButton.addEventListener("click", () => {
                if (!showFahrenheit) {
                  showFahrenheit = true;
                  applyUnit();
                }
              });

              function stop() {
                if (timer !== null) {
                  window.clearInterval(timer);
                  timer = null;
                }
                playButton.textContent = "Play";
              }

              function frameInterval() {
                const speed = Number(speedSelect.value) || 1;
                if (timeframes.length < 2) {
                  return 500 / speed;
                }
                const first = Date.parse(timeframes[0]);
                const last = Date.parse(timeframes[timeframes.length - 1]);
                if (!Number.isFinite(first) || !Number.isFinite(last) || last <= first) {
                  return 500 / speed;
                }
                return (last - first) / (timeframes.length - 1) / speed;
              }

              function play() {
                stop();
                playButton.textContent = "Pause";
                timer = window.setInterval(() => {
                  const total = grids.length || imageUrls.length;
                  if (current + 1 >= total) {
                    showFrame(0);
                    return;
                  }
                  showFrame(current + 1);
                }, Math.max(40, frameInterval()));
              }

              playButton.addEventListener("click", () => {
                if (timer === null) {
                  play();
                } else {
                  stop();
                }
              });
              previousButton.addEventListener("click", () => {
                stop();
                showFrame(current - 1);
              });
              nextButton.addEventListener("click", () => {
                stop();
                showFrame(current + 1);
              });
              slider.addEventListener("input", () => {
                stop();
                showFrame(Number(slider.value));
              });
              speedSelect.addEventListener("change", () => {
                if (timer !== null) {
                  play();
                }
              });

              // Minimal ZIP reader. Recordings are written with stored entries,
              // and deflated entries fall back to the browser's own inflater.
              async function readZipEntries(buffer) {
                const view = new DataView(buffer);
                const bytes = new Uint8Array(buffer);
                let directoryStart = -1;
                for (let offset = buffer.byteLength - 22; offset >= 0; offset -= 1) {
                  if (view.getUint32(offset, true) === 0x06054b50) {
                    directoryStart = view.getUint32(offset + 16, true);
                    break;
                  }
                }
                if (directoryStart < 0) {
                  throw new Error("This file is not a ZIP archive.");
                }
                const entries = new Map();
                let cursor = directoryStart;
                while (cursor < buffer.byteLength &&
                       view.getUint32(cursor, true) === 0x02014b50) {
                  const method = view.getUint16(cursor + 10, true);
                  const compressedSize = view.getUint32(cursor + 20, true);
                  const nameLength = view.getUint16(cursor + 28, true);
                  const extraLength = view.getUint16(cursor + 30, true);
                  const commentLength = view.getUint16(cursor + 32, true);
                  const localOffset = view.getUint32(cursor + 42, true);
                  const name = new TextDecoder().decode(
                    bytes.subarray(cursor + 46, cursor + 46 + nameLength)
                  );
                  const localNameLength = view.getUint16(localOffset + 26, true);
                  const localExtraLength = view.getUint16(localOffset + 28, true);
                  const dataStart =
                    localOffset + 30 + localNameLength + localExtraLength;
                  let data = bytes.slice(dataStart, dataStart + compressedSize);
                  if (method === 8) {
                    const stream = new Blob([data])
                      .stream()
                      .pipeThrough(new DecompressionStream("deflate-raw"));
                    data = new Uint8Array(await new Response(stream).arrayBuffer());
                  } else if (method !== 0) {
                    throw new Error(`Unsupported ZIP compression (${method}).`);
                  }
                  entries.set(name, data);
                  cursor += 46 + nameLength + extraLength + commentLength;
                }
                return entries;
              }

              function resetRecording() {
                stop();
                imageUrls.forEach((url) => URL.revokeObjectURL(url));
                grids = [];
                imageUrls = [];
                timeframes = [];
                current = 0;
                cursorCell = null;
              }

              function adoptRecording(payloadFrames, payloadTimeframes, jpegs) {
                grids = payloadFrames || [];
                timeframes = payloadTimeframes || [];
                imageUrls = jpegs || [];
                if (grids.length > 0) {
                  // Auto-range over the whole recording so the replay stays
                  // readable whatever range the live page was using.
                  rangeMin = Infinity;
                  rangeMax = -Infinity;
                  grids.forEach((grid) => {
                    const summary = summarize(grid);
                    rangeMin = Math.min(rangeMin, summary.min);
                    rangeMax = Math.max(rangeMax, summary.max);
                  });
                  if (rangeMax - rangeMin < 1) {
                    rangeMax = rangeMin + 1;
                  }
                }
                const total = grids.length || imageUrls.length;
                const hasFrames = total > 0;
                canvas.hidden = grids.length === 0;
                frameImage.hidden = grids.length > 0;
                slider.max = String(Math.max(0, total - 1));
                slider.disabled = !hasFrames;
                playButton.disabled = !hasFrames;
                previousButton.disabled = !hasFrames;
                nextButton.disabled = !hasFrames;
                valuesButton.disabled = grids.length === 0;
                renderScaleTicks();
                if (hasFrames) {
                  showFrame(0);
                }
                return total;
              }

              fileInput.addEventListener("change", async () => {
                const file = fileInput.files && fileInput.files[0];
                if (!file) {
                  return;
                }
                loadState.classList.remove("is-error");
                loadState.textContent = `Reading ${file.name} ...`;
                try {
                  if (palette === null) {
                    await loadPalette();
                  }
                  resetRecording();
                  let total = 0;
                  if (file.name.toLowerCase().endsWith(".json")) {
                    const payload = JSON.parse(await file.text());
                    total = adoptRecording(payload.frames, payload.timeframes, []);
                    loadState.textContent =
                      `${file.name}: ${total} frames with temperatures.`;
                  } else {
                    const entries = await readZipEntries(await file.arrayBuffer());
                    const metadata = entries.has("metadata.json")
                      ? JSON.parse(new TextDecoder().decode(entries.get("metadata.json")))
                      : {};
                    const numbers = entries.has("frames.json")
                      ? JSON.parse(new TextDecoder().decode(entries.get("frames.json")))
                      : null;
                    const jpegNames = Array.from(entries.keys())
                      .filter((name) => name.toLowerCase().endsWith(".jpg"))
                      .sort();
                    const jpegs = jpegNames.map((name) =>
                      URL.createObjectURL(
                        new Blob([entries.get(name)], { type: "image/jpeg" })
                      )
                    );
                    total = adoptRecording(
                      numbers ? numbers.frames : null,
                      (numbers && numbers.timeframes) || metadata.timeframes,
                      jpegs
                    );
                    if (total === 0) {
                      throw new Error("No frames found in this archive.");
                    }
                    loadState.textContent = numbers
                      ? `${file.name}: ${total} frames with temperatures.`
                      : `${file.name}: ${total} JPEG frames (no temperatures in` +
                        ` this archive; record with Both or Numbers for readouts).`;
                  }
                } catch (readError) {
                  resetRecording();
                  adoptRecording([], [], []);
                  loadState.classList.add("is-error");
                  loadState.textContent = `Could not open ${file.name}: ${readError.message}`;
                }
              });

              applyUnit();
            </script>
          </body>
        </html>
        """

    @app.get("/video_feed")
    def video_feed():
        try:
            subscription = stream.frames()
        except StreamCapacityError as error:
            return jsonify(error=str(error)), 503
        return Response(
            subscription,
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/camera_status")
    def camera_status():
        return jsonify(stream.camera_status())

    @app.get("/latest_frame.json")
    def latest_frame():
        """The newest temperature grid, for the page's per-pixel readout."""
        temperatures = stream.latest_temperatures()
        if temperatures is None:
            return jsonify(width=FRAME_WIDTH, height=FRAME_HEIGHT, temperatures=None)
        return jsonify(
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            temperatures=[
                [round(float(value), 1) for value in row] for row in temperatures
            ],
        )

    @app.get("/temperature_scale.png")
    def temperature_scale():
        return Response(
            temperature_scale_png(),
            mimetype="image/png",
            # The scale only depends on MIN/MAX_TEMPERATURE_C, so it is
            # immutable for the lifetime of a deployed image.
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.post("/recording/start")
    def recording_start():
        payload = request.get_json(silent=True) or {}
        try:
            return jsonify(recorder.start(payload.get("format", "images")))
        except RecordingError as error:
            return jsonify(error=str(error)), 409

    @app.post("/recording/stop")
    def recording_stop():
        try:
            return jsonify(recorder.stop())
        except RecordingError as error:
            return jsonify(error=str(error)), 409

    @app.get("/recording/status")
    def recording_status():
        return jsonify(recorder.status())

    @app.get("/recording/download")
    def recording_download():
        try:
            payload, filename, mimetype = recorder.download()
        except RecordingError as error:
            return jsonify(error=str(error)), 409
        return Response(
            payload,
            mimetype=mimetype,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(payload)),
            },
        )

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "3000")),
        debug=False,
        threaded=True,
    )
