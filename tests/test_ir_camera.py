from __future__ import annotations

import io
import json
import logging
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ir_camera import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    FrameBroadcaster,
    FrameSnapshot,
    Mlx90640Camera,
    Recorder,
    RecordingError,
    create_app,
    render_temperature_scale,
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


class PacedCamera:
    """FixedCamera at a sane rate so recordings stay small in tests."""

    def __init__(self, interval: float = 0.02) -> None:
        self._interval = interval
        self._offset = 0.0

    def read_frame(self) -> np.ndarray:
        time.sleep(self._interval)
        self._offset += 0.1
        return FixedCamera().read_frame() + self._offset


def wait_for_recording(client, *, minimum_frames: int, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    status = client.get("/recording/status").get_json()
    while status["frames"] < minimum_frames and time.monotonic() < deadline:
        time.sleep(0.02)
        status = client.get("/recording/status").get_json()
    return status


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
    assert payload["average_celsius"] == 30.0


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


def test_rendered_frame_keeps_the_sensor_resolution():
    temperatures = FixedCamera().read_frame()

    image = render_thermal_frame(temperatures)

    assert image.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
    assert image.dtype == np.uint8


def test_rendered_frame_maps_temperature_to_the_inferno_ramp_unannotated():
    temperatures = np.full((FRAME_HEIGHT, FRAME_WIDTH), 20.0, dtype=np.float32)
    hot_y, hot_x = 12, 16
    temperatures[hot_y, hot_x] = 45.0

    image = render_thermal_frame(temperatures)

    hottest = cv2.applyColorMap(
        np.full((1, 1), 255, dtype=np.uint8), cv2.COLORMAP_INFERNO
    )
    cool = cv2.applyColorMap(
        np.full((1, 1), round((20.0 - 15.0) / 30.0 * 255), dtype=np.uint8),
        cv2.COLORMAP_INFERNO,
    )
    assert np.array_equal(image[hot_y, hot_x], hottest[0, 0])
    assert np.array_equal(image[0, 0], cool[0, 0])
    # Nothing is painted over the data: every pixel is a color map entry.
    assert not np.all(image >= 250, axis=2).any()


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


def test_recording_status_starts_idle():
    app = create_app(camera=PacedCamera())
    app.config.update(TESTING=True)

    payload = app.test_client().get("/recording/status").get_json()

    assert payload["state"] == "idle"
    assert payload["frames"] == 0
    assert payload["session_id"] is None
    assert payload["limit_reached"] is False


def test_json_recording_downloads_timeframes_and_temperature_grids():
    app = create_app(camera=PacedCamera())
    app.config.update(TESTING=True)
    client = app.test_client()

    started = client.post("/recording/start", json={"format": "json"})
    status = wait_for_recording(client, minimum_frames=2)
    client.post("/recording/stop")
    download = client.get("/recording/download")

    assert started.status_code == 200
    assert started.get_json()["state"] == "recording"
    assert status["frames"] >= 2
    assert download.status_code == 200
    assert download.mimetype == "application/json"
    session_id = started.get_json()["session_id"]
    assert download.headers["Content-Disposition"] == (
        f'attachment; filename="ir-camera-{session_id}.json"'
    )
    payload = json.loads(download.get_data())
    assert payload["frame_count"] == len(payload["frames"])
    assert len(payload["timeframes"]) == payload["frame_count"]
    assert payload["unit"] == "celsius"
    for frame in payload["frames"]:
        assert len(frame) == FRAME_HEIGHT
        assert all(len(row) == FRAME_WIDTH for row in frame)
        assert all(isinstance(value, float) for row in frame for value in row)
    assert payload["frames"][0] != payload["frames"][-1]


@pytest.mark.parametrize("output_format", ["images", "both"])
def test_image_recording_downloads_one_jpeg_per_timeframe(output_format):
    app = create_app(camera=PacedCamera())
    app.config.update(TESTING=True)
    client = app.test_client()

    started = client.post("/recording/start", json={"format": output_format})
    assert started.status_code == 200
    wait_for_recording(client, minimum_frames=2)
    client.post("/recording/stop")
    download = client.get("/recording/download")

    assert download.mimetype == "application/zip"
    assert download.headers["Content-Disposition"].endswith('.zip"')
    archive = zipfile.ZipFile(io.BytesIO(download.get_data()))
    names = archive.namelist()
    jpegs = [name for name in names if name.endswith(".jpg")]
    metadata = json.loads(archive.read("metadata.json"))
    assert jpegs == [f"frame-{index:06d}.jpg" for index in range(1, len(jpegs) + 1)]
    assert len(metadata["timeframes"]) == len(jpegs) == metadata["frame_count"]
    assert archive.read(jpegs[0]).startswith(b"\xff\xd8")
    if output_format == "both":
        temperatures = json.loads(archive.read("frames.json"))
        assert temperatures["timeframes"] == metadata["timeframes"]
        assert temperatures["frame_count"] == len(jpegs)
        assert temperatures["session_id"] == metadata["session_id"]
        assert temperatures["unit"] == "celsius"
        assert len(temperatures["frames"]) == len(jpegs)
        for frame in temperatures["frames"]:
            assert len(frame) == FRAME_HEIGHT
            assert all(len(row) == FRAME_WIDTH for row in frame)
        assert temperatures["frames"][0] != temperatures["frames"][-1]
    else:
        assert "frames.json" not in names


def test_recording_requests_that_do_not_match_the_state_report_conflicts():
    app = create_app(camera=PacedCamera())
    app.config.update(TESTING=True)
    client = app.test_client()

    stop_while_idle = client.post("/recording/stop")
    download_while_idle = client.get("/recording/download")
    unknown_format = client.post("/recording/start", json={"format": "mp4"})
    client.post("/recording/start", json={"format": "json"})
    second_start = client.post("/recording/start", json={"format": "json"})
    client.post("/recording/stop")

    assert stop_while_idle.status_code == 409
    assert stop_while_idle.get_json()["error"] == "No recording is running."
    assert download_while_idle.status_code == 409
    assert download_while_idle.get_json()["error"] == (
        "No finished recording is available to download."
    )
    assert unknown_format.status_code == 409
    assert second_start.status_code == 409
    assert second_start.get_json()["error"] == "A recording is already running."


def test_recording_stops_itself_at_the_frame_limit():
    stream = FrameBroadcaster(PacedCamera())
    recorder = Recorder(stream, max_frames=2)

    recorder.start("json")
    deadline = time.monotonic() + 5.0
    status = recorder.status()
    while status["state"] == "recording" and time.monotonic() < deadline:
        time.sleep(0.02)
        status = recorder.status()

    assert status["state"] == "ready"
    assert status["frames"] == 2
    assert status["limit_reached"] is True
    payload, filename, mimetype = recorder.download()
    assert json.loads(payload)["frame_count"] == 2
    assert filename.endswith(".json")
    assert mimetype == "application/json"


def test_recording_waits_while_the_sensor_is_unresponsive():
    camera = Mlx90640Camera(
        sensor=ScriptedSensor([OSError("remote I/O error")]),
        read_attempts=1,
    )
    stream = FrameBroadcaster(camera, failure_delay=0.01)
    recorder = Recorder(stream, poll_timeout=0.05)

    recorder.start("images")
    time.sleep(0.3)
    during = recorder.status()
    after = recorder.stop()

    assert during["state"] == "recording"
    assert during["frames"] == 0
    assert after["state"] == "ready"
    assert after["frames"] == 0
    with pytest.raises(RecordingError):
        recorder.download()


def test_home_page_offers_recording_controls_without_a_viewer_tile():
    app = create_app(camera=PacedCamera())
    app.config.update(TESTING=True)

    response = app.test_client().get("/")

    assert b'id="start-recording"' in response.data
    assert b'id="stop-recording"' in response.data
    assert b'id="download-recording"' in response.data
    assert b'name="capture-format"' in response.data
    assert b'id="viewers-value"' not in response.data
    assert b"Viewers" not in response.data


def test_home_page_shows_the_temperature_scale_served_as_a_png():
    app = create_app(camera=PacedCamera())
    app.config.update(TESTING=True)
    client = app.test_client()

    page = client.get("/")
    scale = client.get("/temperature_scale.png")

    assert b'src="temperature_scale.png"' in page.data
    assert b"45 C" in page.data and b"15 C" in page.data
    assert scale.status_code == 200
    assert scale.mimetype == "image/png"
    assert scale.get_data().startswith(b"\x89PNG")


def test_temperature_scale_runs_hot_at_the_top_in_the_feed_color_map():
    image = render_temperature_scale(width=4, height=16)

    hottest = cv2.applyColorMap(np.full((1, 1), 255, dtype=np.uint8), cv2.COLORMAP_INFERNO)
    coldest = cv2.applyColorMap(np.zeros((1, 1), dtype=np.uint8), cv2.COLORMAP_INFERNO)
    assert image.shape == (16, 4, 3)
    assert np.array_equal(image[0, 0], hottest[0, 0])
    assert np.array_equal(image[-1, 0], coldest[0, 0])


def test_latest_frame_serves_the_grid_the_hover_readout_reads():
    app = create_app(camera=FixedCamera())
    app.config.update(TESTING=True)
    client = app.test_client()

    before = client.get("/latest_frame.json").get_json()
    video_response = client.get("/video_feed", buffered=False)
    next(video_response.response)
    video_response.close()
    after = client.get("/latest_frame.json").get_json()

    assert before["temperatures"] is None
    assert before["width"] == FRAME_WIDTH and before["height"] == FRAME_HEIGHT
    assert len(after["temperatures"]) == FRAME_HEIGHT
    assert all(len(row) == FRAME_WIDTH for row in after["temperatures"])
    assert after["temperatures"][0][0] == pytest.approx(15.0, abs=0.05)
    assert after["temperatures"][-1][-1] == pytest.approx(45.0, abs=0.05)


def test_restarting_recording_does_not_import_a_previous_sessions_frame():
    class HeldStream:
        def __init__(self):
            self.first_started = threading.Event()
            self.second_started = threading.Event()
            self.release_first = threading.Event()
            self.release_second = threading.Event()
            self.lock = threading.Lock()
            self.calls = 0

        def latest_snapshot(self, timeout):
            with self.lock:
                self.calls += 1
                first = self.calls == 1
            if first:
                self.first_started.set()
                assert self.release_first.wait(5)
                return FrameSnapshot(
                    1, datetime.now(timezone.utc), b"jpeg", FixedCamera().read_frame()
                )
            self.second_started.set()
            self.release_second.wait(5)
            return None

    stream = HeldStream()
    recorder = Recorder(stream, max_frames=1)
    recorder.start("json")
    old_worker = recorder._worker
    try:
        assert stream.first_started.wait(2)
        recorder.stop()
        recorder.start("json")
        assert stream.second_started.wait(2)
        stream.release_first.set()
        old_worker.join(2)
        status = recorder.status()
        assert status["frames"] == 0
        assert status["state"] == "recording"
    finally:
        if recorder.status()["state"] == "recording":
            recorder.stop()
        stream.release_first.set()
        stream.release_second.set()
