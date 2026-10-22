#!/usr/bin/env python3
"""MLX90640 Raspberry Pi thermal camera web streamer."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from typing import Protocol

import cv2
import numpy as np
from flask import Flask, Response, jsonify

FRAME_WIDTH = 32
FRAME_HEIGHT = 24
DISPLAY_SCALE = 20
MIN_TEMPERATURE_C = 15.0
MAX_TEMPERATURE_C = 45.0
FRAME_READ_ATTEMPTS = 3
FRAME_RETRY_DELAY_SECONDS = 0.1
CAPTURE_FAILURE_DELAY_SECONDS = 0.5
FRAME_READ_ERRORS = (RuntimeError, ValueError, OSError)
LOGGER = logging.getLogger(__name__)


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
    """Render one 32 x 24 temperature frame with the inferno color map."""
    if temperatures.shape != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(
            f"Expected a {FRAME_HEIGHT} x {FRAME_WIDTH} frame, "
            f"received {temperatures.shape}"
        )

    hot_y, hot_x = np.unravel_index(np.argmax(temperatures), temperatures.shape)
    hot_temperature = float(temperatures[hot_y, hot_x])

    normalized = np.clip(
        temperatures,
        MIN_TEMPERATURE_C,
        MAX_TEMPERATURE_C,
    )
    normalized = (normalized - MIN_TEMPERATURE_C) / (
        MAX_TEMPERATURE_C - MIN_TEMPERATURE_C
    )
    grayscale = (normalized * 255).astype(np.uint8)
    display_grayscale = cv2.resize(
        grayscale,
        (FRAME_WIDTH * DISPLAY_SCALE, FRAME_HEIGHT * DISPLAY_SCALE),
        interpolation=cv2.INTER_CUBIC,
    )
    color_image = cv2.applyColorMap(display_grayscale, cv2.COLORMAP_INFERNO)

    label = f"Hot: {hot_temperature:.1f} C"
    label_origin = (12, 32)
    cv2.putText(
        color_image,
        label,
        label_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        color_image,
        label,
        label_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    hot_center = (
        int((hot_x + 0.5) * DISPLAY_SCALE),
        int((hot_y + 0.5) * DISPLAY_SCALE),
    )
    cv2.circle(color_image, hot_center, 10, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.circle(color_image, hot_center, 10, (255, 255, 255), 2, cv2.LINE_AA)

    return color_image


class FrameBroadcaster:
    """Capture once and broadcast each MJPEG frame to every connected viewer."""

    def __init__(
        self,
        camera: Camera,
        *,
        failure_delay: float = CAPTURE_FAILURE_DELAY_SECONDS,
        logger: logging.Logger | None = None,
    ) -> None:
        self._camera = camera
        self._failure_delay = failure_delay
        self._logger = logger if logger is not None else LOGGER
        self._condition = threading.Condition()
        self._capture_thread: threading.Thread | None = None
        self._subscriber_count = 0
        self._stop_requested = False
        self._latest_frame: bytes | None = None
        self._sequence = 0

    @property
    def subscriber_count(self) -> int:
        with self._condition:
            return self._subscriber_count

    def camera_status(self) -> dict[str, str]:
        error = getattr(self._camera, "last_error", None)
        with self._condition:
            has_frame = self._latest_frame is not None
            capture_running = self._capture_thread is not None

        if error:
            return {
                "state": "offline",
                "message": "CAMERA OFFLINE",
                "detail": (
                    "MLX90640 at I2C address 0x33 is not responding. "
                    f"{error}"
                ),
            }
        if has_frame:
            return {
                "state": "online",
                "message": "CAMERA ONLINE",
                "detail": "Live thermal frames are available.",
            }
        return {
            "state": "connecting",
            "message": "CONNECTING TO CAMERA",
            "detail": (
                "Waiting for the first MLX90640 frame."
                if capture_running
                else "The thermal stream has not started yet."
            ),
        }

    def frames(self) -> Iterator[bytes]:
        with self._condition:
            self._subscriber_count += 1
            self._stop_requested = False
            previous_sequence = self._sequence
            if self._capture_thread is None:
                self._capture_thread = threading.Thread(
                    target=self._capture_frames,
                    name="mlx90640-capture",
                    daemon=True,
                )
                self._capture_thread.start()
            self._logger.info(
                "Thermal stream viewer connected; viewers=%d",
                self._subscriber_count,
            )

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

    def _capture_frames(self) -> None:
        self._logger.info("MLX90640 shared capture loop started")
        try:
            while True:
                with self._condition:
                    if self._stop_requested and self._subscriber_count == 0:
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
                        if self._condition.wait_for(
                            lambda: (
                                self._stop_requested
                                and self._subscriber_count == 0
                            ),
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

                multipart_frame = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n"
                )
                with self._condition:
                    self._latest_frame = multipart_frame
                    self._sequence += 1
                    self._condition.notify_all()
        finally:
            with self._condition:
                self._capture_thread = None
                self._condition.notify_all()
            self._logger.info("MLX90640 shared capture loop stopped")


def create_app(camera: Camera | None = None) -> Flask:
    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)
    thermal_camera = (
        camera if camera is not None else Mlx90640Camera(logger=app.logger)
    )
    stream = FrameBroadcaster(thermal_camera, logger=app.logger)
    app.extensions["thermal_stream"] = stream

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
          </head>
          <body>
            <main>
              <h1>Live Thermal Camera Feed</h1>
              <div id="camera-status" class="status connecting"
                   role="alert" aria-live="assertive">
                CONNECTING TO CAMERA
              </div>
              <div class="camera">
                <img src="video_feed" width="640" height="480"
                     alt="Live MLX90640 thermal camera feed">
                <div id="camera-offline" class="camera-offline" hidden>
                  <strong>CAMERA OFFLINE</strong>
                  <span id="camera-error">
                    MLX90640 at I2C address 0x33 is not responding.
                  </span>
                </div>
              </div>
            </main>
            <script>
              const banner = document.getElementById("camera-status");
              const offline = document.getElementById("camera-offline");
              const error = document.getElementById("camera-error");

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
                  offline.hidden = status.state !== "offline";
                  error.textContent = status.detail;
                } catch (requestError) {
                  banner.className = "status offline";
                  banner.textContent =
                    `CAMERA STATUS UNAVAILABLE - ${requestError.message}`;
                  offline.hidden = false;
                  error.textContent =
                    "The application cannot report the MLX90640 status.";
                }
              }

              refreshCameraStatus();
              window.setInterval(refreshCameraStatus, 2000);
            </script>
          </body>
        </html>
        """

    @app.get("/video_feed")
    def video_feed():
        return Response(
            stream.frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/camera_status")
    def camera_status():
        return jsonify(stream.camera_status())

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
