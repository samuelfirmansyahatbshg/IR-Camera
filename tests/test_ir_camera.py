from __future__ import annotations

import numpy as np

from ir_camera import FRAME_HEIGHT, FRAME_WIDTH, create_app, render_thermal_frame


class FixedCamera:
    def read_frame(self) -> np.ndarray:
        return np.linspace(
            15.0,
            45.0,
            FRAME_WIDTH * FRAME_HEIGHT,
            dtype=np.float32,
        ).reshape((FRAME_HEIGHT, FRAME_WIDTH))


def test_home_page_embeds_the_live_feed():
    app = create_app(camera=FixedCamera())
    app.config.update(TESTING=True)

    response = app.test_client().get("/")

    assert response.status_code == 200
    assert b"Live Thermal Camera Feed" in response.data
    assert b'/video_feed' in response.data


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


def test_liveness_does_not_access_camera_hardware():
    app = create_app(camera=FixedCamera())
    app.config.update(TESTING=True)

    response = app.test_client().get("/healthz")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
