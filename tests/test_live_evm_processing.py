import base64
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from EVM import app as evm_app, gaussian_pyramid
from rPPG import apply_evm_like_processing


def test_gaussian_pyramid_returns_channel_first_pyramid():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    pyramid = gaussian_pyramid(frame, level=3)

    assert pyramid.shape == (3, 15, 20)


def test_apply_evm_like_processing_returns_bgr_frame():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[:, :, 0] = 40
    frame[:, :, 1] = 80
    frame[:, :, 2] = 120

    processed = apply_evm_like_processing(frame, fps=30.0)

    assert processed.shape == frame.shape
    assert processed.dtype == np.uint8


def test_dashboard_uses_same_origin_processing_endpoint():
    dashboard_path = Path(__file__).resolve().parents[1] / "dashboard.html"
    dashboard_html = dashboard_path.read_text(encoding="utf-8")

    assert "/process_frame" in dashboard_html
    assert "http://127.0.0.1:5000/process_frame" not in dashboard_html


def test_process_frame_accepts_data_url_without_padding():
    evm_app.config["TESTING"] = True
    client = evm_app.test_client()

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", frame)
    payload = base64.b64encode(encoded).decode("ascii").rstrip("=")
    response = client.post(
        "/process_frame",
        json={"image": f"data:image/jpeg;base64,{payload}"},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert "image" in body
    assert body["image"].startswith("data:image/jpeg;base64,")
