import base64
from io import BytesIO
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

APP_ROOT = Path(__file__).resolve().parents[1]
PROCESSOR_ROOT = APP_ROOT / "IDPhotoProcessor"
if str(PROCESSOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSOR_ROOT))

from IDPhotoProcessor import IDPhotoProcessor  # noqa: E402

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
MAX_PROCESSING_DIMENSION = 1600
ALLOWED_BACKGROUND_COLORS = {"blue", "white", "red"}
ALLOWED_PHOTO_SIZES = {
    "小一寸",
    "标准一寸",
    "大一寸",
    "小二寸",
    "标准二寸",
    "大二寸",
}
ENHANCEMENT_MIN = -50
ENHANCEMENT_MAX = 50

app = FastAPI(title="Smart Campus ID Photo API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://eaea499.cn", "https://www.eaea499.cn"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

processor = None if os.getenv("IDPHOTO_TEST_MODE") == "1" else IDPhotoProcessor()
processor_lock = threading.Lock()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "processor_ready": processor is not None,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
    }


def _decode_image(content: bytes):
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 10 MB。")
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=415, detail="无法读取图片，请上传有效的 JPG、PNG 或 WEBP 文件。")
    height, width = image.shape[:2]
    if width * height > MAX_PIXELS:
        raise HTTPException(status_code=413, detail="图片分辨率过大，请上传不超过 2000 万像素的图片。")
    return image


def _prepare_for_processing(image: np.ndarray) -> np.ndarray:
    """Limit expensive model and segmentation work to a practical input size."""
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= MAX_PROCESSING_DIMENSION:
        return image

    scale = MAX_PROCESSING_DIMENSION / longest_side
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def _data_url(image: np.ndarray) -> str:
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with BytesIO() as buffer:
            Image.fromarray(rgb_image).save(buffer, format="JPEG", quality=94, dpi=(300, 300))
            encoded = buffer.getvalue()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="结果图编码失败。") from exc
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def _parse_enhancement_value(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"{label}只能在 -50 到 50 之间。") from exc
    if not ENHANCEMENT_MIN <= parsed <= ENHANCEMENT_MAX:
        raise HTTPException(status_code=422, detail=f"{label}只能在 -50 到 50 之间。")
    return parsed


def _parse_auto_enhance(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise HTTPException(status_code=422, detail="自动增强参数无效。")


def _parse_portrait_denoise(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise HTTPException(status_code=422, detail="人像去噪参数无效。")


@app.post("/process")
async def process(
    image: UploadFile = File(...),
    bg_color: str = Form("blue"),
    size: str = Form("标准一寸"),
    brightness: str = Form("0"),
    contrast: str = Form("10"),
    auto_enhance: str = Form("false"),
    portrait_denoise: str = Form("false"),
):
    if bg_color not in ALLOWED_BACKGROUND_COLORS:
        raise HTTPException(status_code=422, detail="背景颜色只能选择蓝底、白底或红底。")
    if size not in ALLOWED_PHOTO_SIZES:
        raise HTTPException(status_code=422, detail="证件照尺寸不受支持，请重新选择。")
    parsed_brightness = _parse_enhancement_value(brightness, "亮度")
    parsed_contrast = _parse_enhancement_value(contrast, "对比度")
    parsed_auto_enhance = _parse_auto_enhance(auto_enhance)
    parsed_portrait_denoise = _parse_portrait_denoise(portrait_denoise)
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    source = _decode_image(content)
    original_size = {"width": int(source.shape[1]), "height": int(source.shape[0])}
    working_image = _prepare_for_processing(source)
    if processor is None:
        raise HTTPException(status_code=503, detail="图像处理服务尚未就绪，请稍后重试。")

    params = IDPhotoProcessor.PRESETS["学生证标准"].copy()
    params["bg_color"] = bg_color
    params["size"] = size
    params["brightness"] = parsed_brightness
    params["contrast"] = parsed_contrast
    params["auto_enhance"] = parsed_auto_enhance
    params["portrait_denoise"] = parsed_portrait_denoise
    started = time.perf_counter()
    try:
        with processor_lock:
            result = processor.process_single(working_image, params)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="图像处理失败，请更换图片后重试。") from exc
    finally:
        del content

    if not result.get("success") or result.get("image") is None:
        raise HTTPException(status_code=422, detail=result.get("error", "未能从图片中识别人像。"))

    output = result["image"]
    height, width = output.shape[:2]
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    payload = {
        "image_data_url": _data_url(output),
        "original_size": original_size,
        "processing_size": {"width": int(working_image.shape[1]), "height": int(working_image.shape[0])},
        "result_size": {"width": int(width), "height": int(height)},
        "result_dpi": {"x": 300, "y": 300},
        "processing_time_ms": elapsed_ms,
        "processing_steps": [
            {"step": item.get("step", ""), "success": bool(item.get("success", False))}
            for item in result.get("processing_steps", [])
        ],
    }
    del output, source, working_image, result
    return payload
