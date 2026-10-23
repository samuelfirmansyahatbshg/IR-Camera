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
MIN_TEMPERATURE_C = 15.0
MAX_TEMPERATURE_C = 45.0
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
# Upper bound for one in-memory recording; ~16 minutes at the sensor's 2 Hz.
MAX_RECORDING_FRAMES = 2000
RECORDING_POLL_TIMEOUT_SECONDS = 2.0
RECORDING_FORMATS = ("images", "json")
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
            "Initializing MLX90640 address=0x33, I2C frequency=800000 Hz, "
            "refresh rate=2 Hz"
        )
        i2c = busio.I2C(board.SCL, board.SDA, frequency=800_000)
        sensor = adafruit_mlx90640.MLX90640(i2c)
        sensor.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
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
            raise RecordingError("Unknown format; use images or json.")

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
        if output_format == "json":
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
            return (
                json.dumps(payload).encode("utf-8"),
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
        if self._format == "json":
            size = sum(frame.temperatures.nbytes for frame in self._frames)
        else:
            size = sum(len(frame.jpeg) for frame in self._frames)
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
                margin: 0 0 16px;
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
                         alt="Inferno color scale from 45 to 15 degrees Celsius">
                    <figcaption class="scale-ticks">
                      <span>45 C</span>
                      <span>40</span>
                      <span>35</span>
                      <span>30</span>
                      <span>25</span>
                      <span>20</span>
                      <span>15 C</span>
                    </figcaption>
                  </figure>
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
                    <div class="metric-label">Average</div>
                    <div>
                      <span id="average-value" class="metric-value">--</span
                      ><span class="metric-unit">C</span>
                    </div>
                  </div>
                  <div class="metric">
                    <div class="metric-label">Cursor</div>
                    <div>
                      <span id="cursor-value" class="metric-value">--</span
                      ><span class="metric-unit">C</span>
                    </div>
                  </div>
                </aside>
              </div>
              <section class="measure" aria-label="Measurement tools">
                <div class="measure-tools">
                  <button id="tool-point" type="button" class="tool is-active">Point</button>
                  <button id="tool-line" type="button" class="tool">Line</button>
                  <button id="tool-rect" type="button" class="tool">Rectangle</button>
                  <button id="clear-measurements" type="button" class="tool">Clear</button>
                  <span id="measure-hint">Click for a point; drag for a line or rectangle.</span>
                </div>
                <table class="measure-table">
                  <thead>
                    <tr>
                      <th>Marker</th>
                      <th>Min C</th>
                      <th>Max C</th>
                      <th>Avg C</th>
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
                  averageValue.textContent =
                    status.average_celsius === null
                      ? "--"
                      : status.average_celsius.toFixed(1);
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
                  averageValue.textContent = "--";
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
                cursorValue.textContent = celsius.toFixed(1);
                pixelReadout.hidden = false;
                pixelReadout.style.left = `${cursorPosition.x}px`;
                pixelReadout.style.top = `${cursorPosition.y}px`;
                pixelReadout.textContent =
                  `${celsius.toFixed(1)} C  (x ${column}, y ${row})`;
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

              // Measurement markers: P# points, L# lines, R# rectangles. All
              // geometry is kept in sensor cell coordinates so the overlay and
              // the statistics stay correct at any display size.
              const overlay = document.getElementById("measure-overlay");
              const measurementRows = document.getElementById("measurement-rows");
              const toolButtons = {
                point: document.getElementById("tool-point"),
                line: document.getElementById("tool-line"),
                rect: document.getElementById("tool-rect"),
              };
              const clearButton = document.getElementById("clear-measurements");
              const markers = [];
              const counters = { point: 0, line: 0, rect: 0 };
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
                if (marker.kind === "line") {
                  const dx = marker.x2 - marker.x1;
                  const dy = marker.y2 - marker.y1;
                  let column = Math.floor(marker.x1);
                  let row = Math.floor(marker.y1);
                  const endColumn = Math.floor(marker.x2);
                  const endRow = Math.floor(marker.y2);
                  const stepX = Math.sign(dx);
                  const stepY = Math.sign(dy);
                  const deltaX = dx === 0 ? Infinity : Math.abs(1 / dx);
                  const deltaY = dy === 0 ? Infinity : Math.abs(1 / dy);
                  let nextX = dx === 0 ? Infinity :
                    ((stepX > 0 ? column + 1 : column) - marker.x1) / dx;
                  let nextY = dy === 0 ? Infinity :
                    ((stepY > 0 ? row + 1 : row) - marker.y1) / dy;
                  const values = [];
                  // Visit each crossed sensor cell once, so repeated samples
                  // cannot bias the line's average temperature.
                  while (true) {
                    values.push(cellAt(column, row));
                    if (column === endColumn && row === endRow) {
                      break;
                    }
                    if (nextX + 1e-12 < nextY) {
                      column += stepX;
                      nextX += deltaX;
                    } else if (nextY + 1e-12 < nextX) {
                      row += stepY;
                      nextY += deltaY;
                    } else {
                      column += stepX;
                      row += stepY;
                      nextX += deltaX;
                      nextY += deltaY;
                    }
                  }
                  return values;
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

              function renderOverlay() {
                const shapes = draft === null ? markers : markers.concat([draft]);
                overlay.innerHTML = shapes
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
                    if (marker.kind === "line") {
                      return `<line x1="${marker.x1}" y1="${marker.y1}"` +
                        ` x2="${marker.x2}" y2="${marker.y2}" stroke="#f9fafb"` +
                        ` stroke-width="1.5" vector-effect="non-scaling-stroke"></line>` +
                        `<text x="${(marker.x1 + marker.x2) / 2 + 0.6}"` +
                        ` y="${(marker.y1 + marker.y2) / 2 - 0.6}">${label}</text>`;
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
                      `<td>${summary.min.toFixed(1)}</td>` +
                      `<td>${summary.max.toFixed(1)}</td>` +
                      `<td>${summary.avg.toFixed(1)}</td></tr>`;
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
                counters.line = 0;
                counters.rect = 0;
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
                  if (pending.kind === "line") {
                    counters.line += 1;
                    pending.id = `L${counters.line}`;
                  } else {
                    counters.rect += 1;
                    pending.id = `R${counters.rect}`;
                  }
                  markers.push(pending);
                }
                renderOverlay();
                renderMeasurements();
              });

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

              refreshCameraStatus();
              refreshRecording();
              refreshTemperatureGrid();
              window.setInterval(() => {
                refreshCameraStatus();
                refreshRecording();
              }, 1000);
              // The sensor publishes at 2 Hz; poll the grid at the same rate so
              // the hover readout follows the picture.
              window.setInterval(refreshTemperatureGrid, 500);
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
