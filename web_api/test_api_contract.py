import base64
import io
import os
import sys
from pathlib import Path

os.environ["IDPHOTO_TEST_MODE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from web_api import app as api


class FakeProcessor:
    PRESETS = {"学生证标准": {"bg_color": "blue", "size": "标准一寸"}}
    SIZES = {
        "小一寸": (260, 378),
        "标准一寸": (295, 413),
        "大一寸": (390, 567),
        "小二寸": (413, 531),
        "标准二寸": (413, 579),
        "大二寸": (413, 626),
    }

    def __init__(self):
        self.last_params = None

    def process_single(self, image, params=None):
        self.last_params = params.copy()
        width, height = self.SIZES[params["size"]]
        result = np.full((height, width, 3), (255, 0, 0), dtype=np.uint8)
        return {
            "success": True,
            "image": result,
            "processing_time": 0.012,
            "processing_steps": [{"step": "测试处理", "success": True}],
        }


api.processor = FakeProcessor()
client = TestClient(api.app)


def encoded_image():
    ok, data = cv2.imencode(".jpg", np.zeros((80, 60, 3), dtype=np.uint8))
    assert ok
    return io.BytesIO(data.tobytes())


def test_health_reports_upload_limit():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["max_upload_bytes"] == 10 * 1024 * 1024


def test_large_images_are_downscaled_for_processing():
    image = np.zeros((2400, 3200, 3), dtype=np.uint8)
    processed = api._prepare_for_processing(image)
    assert processed.shape[:2] == (1200, 1600)


def test_rejects_unknown_background_color():
    response = client.post("/process", files={"image": ("photo.jpg", encoded_image(), "image/jpeg")}, data={"bg_color": "green"})
    assert response.status_code == 422


def test_rejects_empty_upload():
    response = client.post("/process", files={"image": ("photo.jpg", b"", "image/jpeg")}, data={"bg_color": "blue"})
    assert response.status_code == 400


def test_process_returns_downloadable_result():
    response = client.post("/process", files={"image": ("photo.jpg", encoded_image(), "image/jpeg")}, data={"bg_color": "red"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["result_size"] == {"width": 295, "height": 413}
    assert payload["result_dpi"] == {"x": 300, "y": 300}
    assert payload["processing_size"] == {"width": 60, "height": 80}
    assert payload["image_data_url"].startswith("data:image/jpeg;base64,")
    encoded_result = base64.b64decode(payload["image_data_url"].split(",", 1)[1])
    with Image.open(io.BytesIO(encoded_result)) as result_image:
        assert result_image.info["dpi"] == (300, 300)


def test_process_defaults_to_standard_one_inch():
    response = client.post(
        "/process",
        files={"image": ("photo.jpg", encoded_image(), "image/jpeg")},
        data={"bg_color": "blue"},
    )
    assert response.status_code == 200
    assert api.processor.last_params["size"] == "标准一寸"
    assert response.json()["result_size"] == {"width": 295, "height": 413}


def test_process_accepts_all_standard_sizes():
    expected = {
        "小一寸": {"width": 260, "height": 378},
        "标准一寸": {"width": 295, "height": 413},
        "大一寸": {"width": 390, "height": 567},
        "小二寸": {"width": 413, "height": 531},
        "标准二寸": {"width": 413, "height": 579},
        "大二寸": {"width": 413, "height": 626},
    }
    for size, dimensions in expected.items():
        response = client.post(
            "/process",
            files={"image": ("photo.jpg", encoded_image(), "image/jpeg")},
            data={"bg_color": "blue", "size": size},
        )
        assert response.status_code == 200
        assert api.processor.last_params["size"] == size
        assert response.json()["result_size"] == dimensions


def test_rejects_unknown_photo_size():
    response = client.post(
        "/process",
        files={"image": ("photo.jpg", encoded_image(), "image/jpeg")},
        data={"bg_color": "blue", "size": "自定义尺寸"},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "证件照尺寸不受支持，请重新选择。"
