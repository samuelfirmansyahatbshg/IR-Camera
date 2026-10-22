from __future__ import annotations

import logging
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from ir_camera import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    FrameBroadcaster,
    Mlx90640Camera,
    create_app,
    render_thermal_frame,
)


class FixedCamera:
    def read_frame(self) -> np.ndarray:
        return np.linspace(
            15.0,
            45.0,
            FRAME_WIDTH * FRAME_HEIGHT,
            dtype=np.float32,
        ).reshape((FRAME_HEIGHT, FRAME_WIDTH))


class ScriptedSensor:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.refresh_rate = None

    def getFrame(self, target):
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        target[:] = outcome


class SharedFrameCamera:
    def __init__(self):
        self.calls = 0
        self.completed_calls = 0
        self.first_read_started = threading.Event()
        self.release_first_read = threading.Event()
        self.release_later_reads = threading.Event()
        self.subscribers_ready = lambda: False

    def read_frame(self):
        self.calls += 1
        if self.calls == 1:
            self.first_read_started.set()
            while not self.subscribers_ready():
                time.sleep(0.001)
            assert self.release_first_read.wait(timeout=1)
            self.completed_calls += 1
            return FixedCamera().read_frame()

        self.release_later_reads.wait(timeout=1)
        return None


def test_home_page_embeds_the_live_feed():
    app = create_app(camera=FixedCamera())
    app.config.update(TESTING=True)

    response = app.test_client().get("/")

    assert response.status_code == 200
    assert b"Live Thermal Camera Feed" in response.data
    assert b'src="video_feed"' in response.data
    assert b'role="alert"' in response.data
    assert b"CAMERA OFFLINE" in response.data
    assert b'fetch("camera_status"' in response.data

def test_home_page_shows_a_capture_rate_readout():
    app = create_app(camera=FixedCamera())
    app.config.update(TESTING=True)

    response = app.test_client().get("/")

    assert b'id="fps-value"' in response.data
    assert b"Capture rate" in response.data
    assert b"fps" in response.data


def test_camera_status_publishes_capture_rate_and_viewer_capacity():
    clock = iter([10.0, 10.5, 11.0, 11.5, 12.0, 12.5]).__next__
    stream = FrameBroadcaster(FixedCamera(), max_viewers=3, clock=clock)
    subscription = stream.frames()

    next(subscription)
    next(subscription)
    next(subscription)
    status = stream.camera_status()
    subscription.close()

    assert status["fps"] == 2.0
    assert status["max_viewers"] == 3
    assert status["state"] == "online"


def test_video_feed_rejects_viewers_beyond_the_cap_without_blocking():
    app = create_app(camera=FixedCamera())
    app.config.update(TESTING=True)
    stream = app.extensions["thermal_stream"]
    client = app.test_client()

    held = []
    for _ in range(stream.max_viewers):
        response = client.get("/video_feed", buffered=False)
        next(response.response)
        held.append(response)

    rejected = client.get("/video_feed")
    status_during_load = client.get("/camera_status")
    for response in held:
        response.close()

    assert rejected.status_code == 503
    assert "already serves" in rejected.get_json()["error"]
    assert status_during_load.status_code == 200
    assert status_during_load.get_json()["viewers"] == stream.max_viewers

def test_camera_status_reports_connecting_before_the_first_frame():
    app = create_app(camera=FixedCamera())
    app.config.update(TESTING=True)

    response = app.test_client().get("/camera_status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["state"] == "connecting"
    assert payload["message"] == "CONNECTING TO CAMERA"
    assert payload["detail"] == "The thermal stream has not started yet."
    assert payload["fps"] is None


def test_camera_status_reports_online_after_a_frame():
    app = create_app(camera=FixedCamera())
    app.config.update(TESTING=True)
    client = app.test_client()

    video_response = client.get("/video_feed", buffered=False)
    next(video_response.response)
    video_response.close()
    response = client.get("/camera_status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["state"] == "online"
    assert payload["message"] == "CAMERA ONLINE"
    assert payload["detail"] == "Live thermal frames are available."


def test_camera_status_loudly_reports_the_latest_sensor_error():
    camera = Mlx90640Camera(
        sensor=ScriptedSensor([OSError("remote I/O error")]),
        read_attempts=1,
    )
    assert camera.read_frame() is None
    app = create_app(camera=camera)
    app.config.update(TESTING=True)

    response = app.test_client().get("/camera_status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["state"] == "offline"
    assert payload["message"] == "CAMERA OFFLINE"
    assert payload["detail"] == (
        "MLX90640 at I2C address 0x33 is not responding. "
        "OSError: remote I/O error"
    )
    assert payload["fps"] is None


def test_video_feed_emits_a_multipart_jpeg_frame():
    app = create_app(camera=FixedCamera())
    app.config.update(TESTING=True)

    response = app.test_client().get("/video_feed", buffered=False)
    first_part = next(response.response)
    response.close()

    assert response.status_code == 200
    assert response.content_type == "multipart/x-mixed-replace; boundary=frame"
    assert first_part.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8")
    assert first_part.endswith(b"\xff\xd9\r\n")


def test_rendered_frame_has_browser_display_dimensions():
    temperatures = FixedCamera().read_frame()

    image = render_thermal_frame(temperatures)

    assert image.shape == (FRAME_HEIGHT * 20, FRAME_WIDTH * 20, 3)
    assert image.dtype == np.uint8

def test_rendered_annotations_stay_small_and_preserve_the_hot_pixel():
    temperatures = np.full((FRAME_HEIGHT, FRAME_WIDTH), 20.0, dtype=np.float32)
    hot_y, hot_x = 12, 16
    temperatures[hot_y, hot_x] = 45.0

    image = render_thermal_frame(temperatures)

    white = np.all(image >= 250, axis=2)
    center_y = hot_y * 20 + 10
    center_x = hot_x * 20 + 10

    assert not white[60:180, :300].any()
    assert not white[center_y, center_x]
    assert white[
        center_y - 12 : center_y + 13,
        center_x - 12 : center_x + 13,
    ].any()
    assert not white[
        center_y - 5 : center_y + 6,
        center_x + 25 : center_x + 40,
    ].any()


def test_liveness_does_not_access_camera_hardware():
    app = create_app(camera=FixedCamera())
    app.config.update(TESTING=True)

    response = app.test_client().get("/healthz")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_camera_initializes_i2c_at_two_hertz(monkeypatch):
    sensor = ScriptedSensor([[20.0] * (FRAME_WIDTH * FRAME_HEIGHT)])
    i2c_arguments = {}
    refresh_rate = object()

    def create_i2c(scl, sda, *, frequency):
        i2c_arguments.update(scl=scl, sda=sda, frequency=frequency)
        return object()

    monkeypatch.setitem(sys.modules, "board", SimpleNamespace(SCL="scl", SDA="sda"))
    monkeypatch.setitem(sys.modules, "busio", SimpleNamespace(I2C=create_i2c))
    monkeypatch.setitem(
        sys.modules,
        "adafruit_mlx90640",
        SimpleNamespace(
            MLX90640=lambda i2c: sensor,
            RefreshRate=SimpleNamespace(REFRESH_2_HZ=refresh_rate),
        ),
    )

    camera = Mlx90640Camera()

    assert camera.read_frame() is not None
    assert i2c_arguments == {"scl": "scl", "sda": "sda", "frequency": 800_000}
    assert sensor.refresh_rate is refresh_rate


def test_camera_retries_transient_frame_failures_then_returns_frame():
    values = [float(index) for index in range(FRAME_WIDTH * FRAME_HEIGHT)]
    sensor = ScriptedSensor(
        [RuntimeError("too many retries"), ValueError("invalid frame"), values]
    )
    delays = []
    camera = Mlx90640Camera(
        sensor=sensor,
        read_attempts=3,
        retry_delay=0.05,
        sleep=delays.append,
    )

    frame = camera.read_frame()

    assert sensor.calls == 3
    assert delays == [0.05, 0.05]
    assert frame.shape == (FRAME_HEIGHT, FRAME_WIDTH)
    assert frame[0, 0] == values[-FRAME_WIDTH]


def test_camera_stops_after_bounded_frame_failures(caplog):
    sensor = ScriptedSensor(
        [
            RuntimeError("too many retries"),
            ValueError("invalid frame"),
            OSError("i2c input/output error"),
        ]
    )
    camera = Mlx90640Camera(
        sensor=sensor,
        read_attempts=3,
        retry_delay=0,
        sleep=lambda delay: None,
    )

    with caplog.at_level(logging.WARNING):
        frame = camera.read_frame()

    assert frame is None
    assert sensor.calls == 3
    assert "attempt 3/3" in caplog.text
    assert "i2c input/output error" in caplog.text


def test_stream_continues_after_an_exhausted_read_cycle():
    values = [20.0] * (FRAME_WIDTH * FRAME_HEIGHT)
    sensor = ScriptedSensor(
        [
            RuntimeError("too many retries"),
            ValueError("invalid frame"),
            OSError("i2c input/output error"),
            values,
        ]
    )
    camera = Mlx90640Camera(
        sensor=sensor,
        read_attempts=3,
        retry_delay=0,
        sleep=lambda delay: None,
    )
    subscription = FrameBroadcaster(camera, failure_delay=0).frames()

    frame = next(subscription)
    subscription.close()

    assert frame.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8")


def test_camera_serializes_i2c_reads():
    class ConcurrentSensor:
        def __init__(self):
            self.active = 0
            self.maximum_active = 0
            self.state_lock = threading.Lock()

        def getFrame(self, target):
            with self.state_lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            time.sleep(0.02)
            target[:] = [20.0] * (FRAME_WIDTH * FRAME_HEIGHT)
            with self.state_lock:
                self.active -= 1

    sensor = ConcurrentSensor()
    camera = Mlx90640Camera(sensor=sensor)
    threads = [threading.Thread(target=camera.read_frame) for _ in range(4)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)

    assert sensor.maximum_active == 1


def test_repeated_viewers_share_one_camera_capture():
    camera = SharedFrameCamera()
    stream = FrameBroadcaster(camera)
    camera.subscribers_ready = lambda: stream.subscriber_count == 2
    subscriptions = [stream.frames(), stream.frames()]
    frames = [None, None]

    def receive(index):
        frames[index] = next(subscriptions[index])

    readers = [
        threading.Thread(target=receive, args=(index,)) for index in range(2)
    ]
    for reader in readers:
        reader.start()

    assert camera.first_read_started.wait(timeout=1)
    camera.release_first_read.set()
    for reader in readers:
        reader.join(timeout=1)

    assert frames[0] == frames[1]
    assert camera.completed_calls == 1

    for subscription in subscriptions:
        subscription.close()
    camera.release_later_reads.set()
