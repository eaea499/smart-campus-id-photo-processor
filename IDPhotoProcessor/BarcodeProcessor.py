"""
快递包裹条码增强与识别预处理模块
包含去噪、锐化、二值化、形态学优化与条码区域提取
"""

import cv2
import numpy as np


class BarcodeProcessor:
    DEFAULT_PARAMS = {
        "denoise_method": "gaussian",
        "denoise_ksize": 3,
        "sharpen_amount": 1.0,
        "threshold_value": 0,
        "threshold_invert": True,
        "morph_ksize": 5,
        "output_mode": "binary",
    }

    def process_single(self, image, params=None):
        if image is None:
            return {"success": False, "error": "输入图像为空"}

        p = self.DEFAULT_PARAMS.copy()
        if params:
            p.update(params)

        try:
            gray = self._to_gray(image)
            denoised = self._denoise(gray, p["denoise_method"], p["denoise_ksize"])
            sharpened = self._sharpen(denoised, p["sharpen_amount"])
            binary = self._binarize(sharpened, p["threshold_value"], p["threshold_invert"])
            morph = self._morph_open(binary, p["morph_ksize"])

            bbox = self._find_barcode_bbox(morph)
            roi = None
            if bbox is not None:
                x, y, w, h = bbox
                roi = image[y : y + h, x : x + w].copy()

            if p["output_mode"] == "roi" and roi is not None:
                display = roi
            else:
                display = self._to_bgr(morph)

            return {
                "success": True,
                "image": display,
                "gray": gray,
                "binary": binary,
                "morph": morph,
                "bbox": bbox,
                "roi": roi,
                "params": p,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "params": p}

    def _to_gray(self, image):
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _to_bgr(self, image):
        if image is None:
            return None
        if len(image.shape) == 3:
            return image
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def _denoise(self, gray, method, ksize):
        k = int(ksize)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1

        if method == "median":
            return cv2.medianBlur(gray, k)
        if method == "bilateral":
            return cv2.bilateralFilter(gray, 9, 75, 75)
        if method == "none":
            return gray
        return cv2.GaussianBlur(gray, (k, k), 0)

    def _sharpen(self, gray, amount):
        a = float(amount)
        if a <= 0:
            return gray
        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpened = cv2.addWeighted(gray, 1.0 + a, blur, -a, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _binarize(self, gray, threshold_value, invert):
        t = int(threshold_value)
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        if t <= 0:
            _, binary = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
            return binary
        _, binary = cv2.threshold(gray, t, 255, flag)
        return binary

    def _morph_open(self, binary, ksize):
        k = int(ksize)
        if k < 1:
            k = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    def _find_barcode_bbox(self, binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        h, w = binary.shape[:2]
        img_area = float(h * w)

        best = None
        best_score = 0.0

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = float(cw * ch)
            if area < img_area * 0.02:
                continue

            ar = cw / max(ch, 1)
            if ar < 1.5:
                continue

            score = area * ar
            if score > best_score:
                best_score = score
                best = (x, y, cw, ch)

        return best

