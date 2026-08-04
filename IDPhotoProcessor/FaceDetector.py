# FaceDetector.py
"""
人脸检测与定位模块
用于证件照中的人脸检测、关键点定位和裁剪区域计算
使用 YOLOv8 进行人脸检测
"""

import os
import sys
import ctypes


def _preload_torch_dlls():
    """
    用 Windows Kernel32 API 预加载整个 torch\lib 下所有 DLL。
    解决 PyQt5 等库提前调用 SetDefaultDllDirectories 导致后续 DLL 加载失败的问题。
    """
    torch_lib = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
    if not os.path.isdir(torch_lib):
        return

    kernel32 = ctypes.windll.kernel32

    # 移除之前的 DLL 搜索限制（如果有）
    try:
        kernel32.SetDefaultDllDirectories(0x00001000)
    except Exception:
        pass

    # 把 torch\lib 加入 DLL 搜索路径
    kernel32.SetDllDirectoryW(torch_lib)
    try:
        os.add_dll_directory(torch_lib)
    except Exception:
        pass

    # 按字母序加载 torch\lib 下所有 .dll，确保依赖链完整
    failed = []
    for f in sorted(os.listdir(torch_lib)):
        if not f.endswith('.dll'):
            continue
        dll_path = os.path.join(torch_lib, f)
        try:
            kernel32.LoadLibraryExW(dll_path, None, 0x00000008)
        except Exception:
            failed.append(f)

    if failed:
        print(f'[FaceDetector] 预加载 DLL 失败 ({len(failed)}个): {failed[:5]}')


_preload_torch_dlls()

import cv2
import numpy as np


class FaceDetector:
    """人脸检测器类 - 使用 YOLOv8"""
    
    def __init__(self):
        """初始化人脸检测器"""
        self.model = None
        self.using_yolo = False
        self.confidence_threshold = 0.5
        self.last_person_rect = None  # YOLO 检测到的完整人物框
        self._load_yolo_model()
    
    def _load_yolo_model(self):
        """加载 YOLOv8 预训练模型"""
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
            self.using_yolo = True
            print("[FaceDetector]  YOLOv8 (yolov8n.pt) 加载成功，将使用深度学习人脸检测")
        except ImportError:
            print("[FaceDetector]  ultralytics 未安装，将使用 Haar 级联（精度较低）")
            self.model = None
            self.using_yolo = False
        except Exception as e:
            print(f"[FaceDetector]  YOLOv8 加载失败 ({e})，将使用 Haar 级联（精度较低）")
            self.model = None
            self.using_yolo = False
    
    def detect_face(self, image):
        """
        检测图像中的人脸（使用 YOLOv8 检测人体 → 估算人脸区域）

        Args:
            image: numpy数组格式的图像 (BGR格式)

        Returns:
            tuple: (success, face_rect)
                - success: bool, 是否检测到人脸
                - face_rect: tuple (x, y, w, h), 人脸区域坐标
        """
        if image is None:
            return False, None

        if self.model is not None:
            try:
                results = self.model(image, verbose=False, conf=self.confidence_threshold)
                boxes = results[0].boxes

                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()

                    # 优先找 person（COCO class 0），找不到就用置信度最高的
                    if hasattr(boxes, 'cls') and boxes.cls is not None:
                        cls = boxes.cls.cpu().numpy()
                        person_idx = np.where(cls == 0)[0]
                    else:
                        person_idx = []

                    if len(person_idx) > 0:
                        person_boxes = xyxy[person_idx]
                        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in person_boxes]
                        best_idx = person_idx[np.argmax(areas)]
                    else:
                        best_idx = np.argmax(boxes.conf.cpu().numpy())

                    best_box = xyxy[best_idx]
                    x1, y1, x2, y2 = best_box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    person_w = x2 - x1
                    person_h = y2 - y1

                    # 保存完整人物框，供 calculate_crop_region 使用
                    self.last_person_rect = (x1, y1, person_w, person_h)

                    # 从人物框估算人脸区域（头顶到下巴，约占上半身 55~60%）
                    face_x = int(x1 + person_w * 0.1)
                    face_y = int(y1)
                    face_w = int(person_w * 0.8)
                    face_h = int(person_h * 0.58)

                    face_x = max(0, face_x)
                    face_y = max(0, face_y)
                    face_w = min(face_w, image.shape[1] - face_x)
                    face_h = min(face_h, image.shape[0] - face_y)

                    if face_w > 10 and face_h > 10:
                        return True, (face_x, face_y, face_w, face_h)

            except Exception as e:
                print(f"[FaceDetector] YOLOv8 检测失败: {e}，回退到 Haar")

        return self._detect_face_haar(image)
    
    def _detect_face_haar(self, image):
        """使用 Haar 级联分类器检测人脸（备用方案）"""
        if not hasattr(self, 'face_cascade'):
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 逐步放宽检测条件
        for sf in [1.05, 1.1, 1.2]:
            for mn in [3, 5]:
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=sf, minNeighbors=mn,
                    minSize=(60, 60)
                )
                if len(faces) > 0:
                    # 选最靠近图像中心的人脸（而非最大）
                    cx, cy = w // 2, h // 2
                    best_face = min(faces, key=lambda f:
                        (f[0] + f[2]//2 - cx)**2 + (f[1] + f[3]//2 - cy)**2)
                    return True, tuple(best_face)

        return False, None
    
    def detect_face_haar(self, image):
        """
        使用 Haar 级联分类器检测人脸（与detect_face相同，保持兼容性）
        
        Args:
            image: numpy数组格式的图像
            
        Returns:
            tuple: (success, face_rect)
        """
        return self.detect_face(image)
    
    def calculate_crop_region(self, image_shape, face_rect, target_ratio=295/413):
        """
        根据人脸和人物框计算裁剪区域。
        YOLO 模式：用人物框底部=衣领/胸口作为下边界，面部中心水平居中。
        Haar 回退：用固定比例估算。

        Args:
            image_shape: 图像形状 (h, w)
            face_rect: 人脸区域 (x, y, w, h)
            target_ratio: 目标宽高比（默认一寸照295/413）

        Returns:
            tuple: (x, y, w, h) 裁剪区域坐标
        """
        img_h, img_w = image_shape[:2]
        face_x, face_y, face_w, face_h = face_rect

        face_center_x = face_x + face_w // 2

        # ── YOLO 模式：用人物框确定动态裁剪 ──
        if self.last_person_rect is not None:
            px, py, pw, ph = self.last_person_rect
            person_bottom = py + ph
            person_center_x = px + pw // 2

            top = face_y - int(face_h * 0.35)
            bottom = person_bottom

            crop_h = bottom - top
            crop_w = int(crop_h * target_ratio)

            # 确保肩膀左右抵边：如果人物比宽高比比例宽，用人物宽度
            if pw > crop_w:
                crop_w = pw
                crop_h = int(crop_w / target_ratio)
                bottom = top + crop_h

            crop_x = person_center_x - crop_w // 2
            crop_y = top

            # 边界修正
            if crop_x < 0:
                crop_x = 0
            if crop_y < 0:
                crop_y = 0
            if crop_x + crop_w > img_w:
                crop_x = max(0, img_w - crop_w)
            if crop_y + crop_h > img_h:
                crop_y = max(0, img_h - crop_h)

            crop_w = min(crop_w, img_w - crop_x)
            crop_h = min(crop_h, img_h - crop_y)

            self.last_person_rect = None
            return (crop_x, crop_y, crop_w, crop_h)

        # ── Haar 回退：固定比例 ──
        crop_h = int(face_h * 2.5)
        crop_w = int(crop_h * target_ratio)

        crop_x = max(0, face_center_x - crop_w // 2)
        crop_y = max(0, face_y - int(face_h * 0.3))

        if crop_x + crop_w > img_w:
            crop_x = img_w - crop_w
        if crop_y + crop_h > img_h:
            crop_y = img_h - crop_h

        crop_x = max(0, crop_x)
        crop_y = max(0, crop_y)
        crop_w = min(crop_w, img_w - crop_x)
        crop_h = min(crop_h, img_h - crop_y)

        return (crop_x, crop_y, crop_w, crop_h)
    
    def get_face_landmarks(self, image, face_rect):
        """
        获取人脸关键点（眼睛、鼻子、嘴巴等位置）
        
        Args:
            image: numpy数组格式的图像
            face_rect: 人脸区域
            
        Returns:
            dict: 关键点坐标字典
        """
        x, y, w, h = face_rect
        
        landmarks = {
            'face_center': (x + w // 2, y + h // 2),
            'left_eye': (x + w // 3, y + h // 3),
            'right_eye': (x + 2 * w // 3, y + h // 3),
            'nose': (x + w // 2, y + h // 2),
            'mouth': (x + w // 2, y + 2 * h // 3)
        }
        
        return landmarks


class IDPhotoCropper:
    """证件照裁剪器"""
    
    # 标准证件照尺寸定义（300DPI）
    STANDARD_SIZES = {
        '小一寸': (260, 378),
        '标准一寸': (295, 413),
        '大一寸': (390, 567),
        '小二寸': (413, 531),
        '标准二寸': (413, 579),
        '大二寸': (413, 626),
        '身份证': (308, 384),
    }
    
    # 为兼容旧代码保留别名
    STANDARD_SIZES['一寸'] = STANDARD_SIZES['标准一寸']
    STANDARD_SIZES['二寸'] = STANDARD_SIZES['大二寸']
    
    def __init__(self):
        self.face_detector = FaceDetector()
    
    def auto_crop(self, image, size_name='标准一寸', use_face_detection=True, keep_full=False, crop_mode='face'):
        """
        自动裁剪证件照 - 智能人像裁剪 + 覆盖式缩放
        
        处理流程：
        1. 读取原始图片，获取原图宽度、高度像素
        2. 选中目标证件照规格，取出目标宽 W、目标高 H
        3. 人脸检测：若检测到人脸，基于人脸计算合理的人像裁剪区域（头肩胸构图）
        4. 将人像区域调整为目标宽高比
        5. 从原图中裁剪出该区域
        6. 使用双三次插值缩放到目标尺寸，保证清晰度
        
        Args:
            image: numpy数组格式的图像
            size_name: 尺寸名称
            use_face_detection: 是否使用人脸检测辅助裁剪定位
            keep_full: 是否保留完整图像（带填充）
            crop_mode: 裁剪模式 ('face', 'center', 'blue')
            
        Returns:
            numpy数组: 裁剪后的图像
        """
        if image is None:
            return None
        
        target_size = self.STANDARD_SIZES.get(size_name, (295, 413))
        target_w, target_h = target_size
        target_ratio = target_w / target_h
        
        h, w = image.shape[:2]
        
        # 保留完整图像模式：用 padding 适配
        if keep_full:
            return self._fit_with_padding(image, target_size)
        
        # 蓝色区域裁剪模式
        if crop_mode == 'blue':
            cropped = self._crop_blue_region(image, target_ratio)
            if cropped is not None:
                return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        # 人脸检测模式：智能裁剪人像区域
        if use_face_detection and crop_mode == 'face':
            success, face_rect = self.face_detector.detect_face(image)
            if success and face_rect is not None:
                fx, fy, fw, fh = face_rect
                
                # 人脸中心点（浮点精度）
                face_cx = fx + fw / 2.0
                face_cy = fy + fh / 2.0
                
                # 估算人像区域（头肩胸）
                # 头顶：脸高 × 0.1，下巴到胸：脸高 × 0.7
                person_top = fy - int(fh * 0.1)
                person_bottom = fy + fh + int(fh * 0.7)
                # 左右：脸宽 × 1.6
                person_w = int(fw * 1.6)
                person_left = int(face_cx - person_w / 2.0)
                person_right = person_left + person_w
                
                person_h = person_bottom - person_top
                
                # 边界约束
                person_left = max(0, person_left)
                person_right = min(w, person_right)
                person_top = max(0, person_top)
                person_bottom = min(h, person_bottom)
                
                # 重新计算实际尺寸
                person_w = person_right - person_left
                person_h = person_bottom - person_top
                
                if person_w > 20 and person_h > 20:
                    # 从原图裁出人像区域
                    person_crop = image[person_top:person_bottom, person_left:person_right]
                    
                    # 计算人脸在人像区域中的相对位置（浮点）
                    face_rel_x = face_cx - person_left
                    face_rel_y = face_cy - person_top
                    
                    # 覆盖式缩放：取较大比例，保证人像铺满目标画布
                    scale_w = target_w / person_w
                    scale_h = target_h / person_h
                    scale = max(scale_w, scale_h)
                    
                    scaled_w = int(round(person_w * scale))
                    scaled_h = int(round(person_h * scale))
                    scaled = cv2.resize(person_crop, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
                    
                    # 人脸在缩放后图中的位置
                    scaled_face_x = face_rel_x * scale
                    scaled_face_y = face_rel_y * scale
                    
                    # 目标位置：人脸水平 50%，垂直 42%
                    target_face_x = target_w / 2.0
                    target_face_y = target_h * 0.42
                    
                    # 计算裁剪起始位置
                    x1 = int(round(scaled_face_x - target_face_x))
                    y1 = int(round(scaled_face_y - target_face_y))
                    
                    # 边界检查
                    x1 = max(0, min(x1, scaled_w - target_w))
                    y1 = max(0, min(y1, scaled_h - target_h))
                    
                    # 最终裁剪
                    result = scaled[y1:y1+target_h, x1:x1+target_w]
                    
                    if result.shape[0] == target_h and result.shape[1] == target_w:
                        return result
        
        # 中心裁剪模式（无人脸或非face模式）：覆盖式缩放 + 中心裁剪
        scale_w = target_w / w
        scale_h = target_h / h
        scale = max(scale_w, scale_h)
        
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        
        scaled = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
        
        x_offset = (scaled_w - target_w) // 2
        y_offset = (scaled_h - target_h) // 2
        
        result = scaled[y_offset:y_offset+target_h, x_offset:x_offset+target_w]
        
        return result

    def _crop_blue_region(self, image, target_ratio):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([85, 40, 40])
        upper = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        img_h, img_w = image.shape[:2]
        img_area = float(img_h * img_w)

        best = None
        best_score = 0.0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = float(w * h)
            if area < img_area * 0.005:
                continue
            ar = w / max(h, 1)
            if ar < 0.45 or ar > 0.95:
                continue
            score = area / (1.0 + abs(ar - target_ratio))
            if score > best_score:
                best_score = score
                best = (x, y, w, h)

        if best is None:
            return None

        x, y, w, h = best
        pad_x = int(w * 0.12)
        pad_y = int(h * 0.12)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]

    def _fit_with_padding(self, image, target_size):
        tw, th = target_size
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return None

        scale = min(tw / w, th / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        bg_color = self._estimate_border_color(image)
        canvas = np.full((th, tw, 3), bg_color, dtype=np.uint8)

        x = (tw - new_w) // 2
        y = (th - new_h) // 2
        canvas[y : y + new_h, x : x + new_w] = resized
        return canvas

    def _estimate_border_color(self, image):
        h, w = image.shape[:2]
        s = max(2, int(min(h, w) * 0.02))
        patches = [
            image[0:s, 0:s],
            image[0:s, w - s : w],
            image[h - s : h, 0:s],
            image[h - s : h, w - s : w],
        ]
        means = [np.mean(p.reshape(-1, 3), axis=0) for p in patches if p.size > 0]
        if not means:
            return (255, 255, 255)
        m = np.mean(np.stack(means, axis=0), axis=0)
        return tuple(int(x) for x in m)
    
    def _center_crop(self, image, target_ratio):
        """
        中心裁剪图像
        
        Args:
            image: numpy数组格式的图像
            target_ratio: 目标宽高比
            
        Returns:
            numpy数组: 裁剪后的图像
        """
        h, w = image.shape[:2]
        current_ratio = w / h
        
        if current_ratio > target_ratio:
            # 图像太宽，裁剪左右
            new_w = int(h * target_ratio)
            x = (w - new_w) // 2
            return image[:, x:x+new_w]
        else:
            # 图像太高，裁剪上下
            new_h = int(w / target_ratio)
            y = (h - new_h) // 2
            return image[y:y+new_h, :]
    
    def resize_to_standard(self, image, size_name='标准一寸'):
        """
        将图像调整为标准证件照尺寸
        
        Args:
            image: numpy数组格式的图像
            size_name: 尺寸名称
            
        Returns:
            numpy数组: 调整后的图像
        """
        target_size = self.STANDARD_SIZES.get(size_name, (295, 413))
        return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)


# 测试代码
if __name__ == "__main__":
    # 测试人脸检测
    detector = FaceDetector()
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 测试检测
    success, face_rect = detector.detect_face_haar(test_image)
    print(f"Detection success: {success}")
    if success:
        print(f"Face rect: {face_rect}")
