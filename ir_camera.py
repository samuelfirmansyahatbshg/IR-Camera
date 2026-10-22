#!/usr/bin/env python3
"""MLX90640 Raspberry Pi thermal camera web streamer."""

from __future__ import annotations

import os
import threading
import time
from typing import Protocol

import cv2
import numpy as np
from flask import Flask, Response, jsonify

FRAME_WIDTH = 32
FRAME_HEIGHT = 24
DISPLAY_SCALE = 20
MIN_TEMPERATURE_C = 15.0
MAX_TEMPERATURE_C = 45.0


class Camera(Protocol):
    def read_frame(self) -> np.ndarray | None: ...


class Mlx90640Camera:
    """Read the MLX90640 on Raspberry Pi I2C bus 1."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sensor = None
        self._frame = [0.0] * (FRAME_WIDTH * FRAME_HEIGHT)

    def _initialize(self) -> None:
        import adafruit_mlx90640
        import board
        import busio

        print("Opening MLX90640 at I2C address 0x33...", flush=True)
        i2c = busio.I2C(board.SCL, board.SDA, frequency=800_000)
        sensor = adafruit_mlx90640.MLX90640(i2c)
        sensor.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
        self._sensor = sensor
        print("MLX90640 found", flush=True)

    def read_frame(self) -> np.ndarray | None:
        with self._lock:
            if self._sensor is None:
                self._initialize()

            try:
                self._sensor.getFrame(self._frame)
            except ValueError:
                return None

            temperatures = np.asarray(self._frame, dtype=np.float32).reshape(
                (FRAME_HEIGHT, FRAME_WIDTH)
            )
            return np.flipud(temperatures)


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
    color_image = cv2.applyColorMap(grayscale, cv2.COLORMAP_INFERNO)

    cv2.putText(
        color_image,
        f"Hot: {hot_temperature:.1f} C",
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    cv2.circle(color_image, (int(hot_x), int(hot_y)), 3, (255, 255, 255), -1)

    return cv2.resize(
        color_image,
        (FRAME_WIDTH * DISPLAY_SCALE, FRAME_HEIGHT * DISPLAY_SCALE),
        interpolation=cv2.INTER_CUBIC,
    )


def generate_mjpeg(camera: Camera):
    """Yield multipart JPEG frames for a browser video element."""
    while True:
        temperatures = camera.read_frame()
        if temperatures is None:
            time.sleep(0.1)
            continue

        image = render_thermal_frame(temperatures)
        encoded, jpeg = cv2.imencode(".jpg", image)
        if not encoded:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )


def create_app(camera: Camera | None = None) -> Flask:
    app = Flask(__name__)
    thermal_camera = camera if camera is not None else Mlx90640Camera()

    @app.get("/")
    def index():
        return f"""
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>IR Camera</title>
          </head>
          <body>
            <h1>Live Thermal Camera Feed</h1>
            <img src="video_feed" width="640" height="480"
                 alt="Live MLX90640 thermal camera feed">
          </body>
        </html>
        """

    @app.get("/video_feed")
    def video_feed():
        return Response(
            generate_mjpeg(thermal_camera),
            mimetype="multipart/x-mixed-replace; boundary=frame",
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
