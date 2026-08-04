# IDPhotoProcessor.py
"""
证件照预处理主控模块
整合人脸检测、背景替换、图像增强等功能
"""

import cv2
import numpy as np
import time
import os
from io import BytesIO
from FaceDetector import FaceDetector, IDPhotoCropper
from BackgroundRemover import BackgroundRemover
from ImageEnhancer import ImageEnhancer

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


class IDPhotoProcessor:
    """证件照预处理器主类"""
    
    # 预设处理方案
    PRESETS = {
        '学生证标准': {
            'size': '标准一寸',
            'bg_color': 'blue',
            'bg_method': 'grabcut',
            'feather_radius': 5,
            'brightness': 0,
            'contrast': 10,
            'denoise_method': 'bilateral',
            'keep_full': False,
            'crop_mode': 'face',
            'auto_enhance': False
        },
        '一卡通标准': {
            'size': '标准一寸',
            'bg_color': 'white',
            'bg_method': 'grabcut',
            'feather_radius': 5,
            'brightness': 5,
            'contrast': 5,
            'denoise_method': 'bilateral',
            'keep_full': False,
            'crop_mode': 'face',
            'auto_enhance': False
        },
        '考试报名': {
            'size': '标准二寸',
            'bg_color': 'blue',
            'bg_method': 'grabcut',
            'feather_radius': 5,
            'brightness': 0,
            'contrast': 0,
            'denoise_method': 'median',
            'keep_full': False,
            'crop_mode': 'face',
            'auto_enhance': False
        },
        '护照签证': {
            'size': '标准二寸',
            'bg_color': 'white',
            'bg_method': 'grabcut',
            'feather_radius': 5,
            'brightness': 0,
            'contrast': 0,
            'denoise_method': 'bilateral',
            'keep_full': False,
            'crop_mode': 'face',
            'auto_enhance': False
        }
    }
    
    def __init__(self):
        """初始化证件照处理器"""
        self.face_detector = FaceDetector()
        self.cropper = IDPhotoCropper()
        self.bg_remover = BackgroundRemover()
        self.enhancer = ImageEnhancer()
        
        # 处理历史记录（操作链）
        self.history = []
        self.history_index = -1
        self.original_image_ref = None  # 保存原始图片引用
        
    def set_original_image(self, image):
        """设置原始图片引用（用于完全撤销）"""
        self.original_image_ref = image.copy() if image is not None else None
        
    def _add_to_history(self, image):
        """添加图像到历史记录"""
        # 删除当前位置之后的历史
        self.history = self.history[:self.history_index + 1]
        
        # 添加新图像
        self.history.append(image.copy())
        self.history_index += 1
        
        # 限制历史记录大小（保留更多步骤以支持完全撤销）
        if len(self.history) > 20:
            self.history.pop(0)
            self.history_index -= 1
    
    def undo(self):
        """撤销操作

        工作逻辑:
        - history = [R1, R2, R3], history_index = 2 (显示 R3)
        - 撤销: history_index = 1 → 返回 R2
        - 撤销: history_index = 0 → 返回 R1
        - 撤销: history_index = -1 → 返回 original_image_ref (原始图像)
        - 撤销: 返回 None (无可撤销)
        """
        if self.history_index >= 0:
            # 还有历史记录可以回退
            self.history_index -= 1
            if self.history_index >= 0:
                # 回退到更早的历史记录
                return self.history[self.history_index].copy()
            elif self.original_image_ref is not None:
                # 已经回到第一步之前，返回原始图像
                return self.original_image_ref.copy()
        # history_index 已经是 -1（或无原始图引用）
        return None

    def redo(self):
        """重做操作"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            return self.history[self.history_index].copy()
        return None

    def can_undo(self):
        """检查是否可以撤销: 只要还有历史记录或有原始图引用即可"""
        return self.history_index >= 0 or (
            self.original_image_ref is not None and self.history_index == -1
        )

    def can_redo(self):
        """检查是否可以重做: 历史索引没有到末尾"""
        return self.history_index < len(self.history) - 1
    
    def process_single(self, image, params=None):
        """处理单张证件照
        
        Args:
            image: numpy数组格式的图像
            params: 处理参数字典，如果为None则使用默认参数
            
        Returns:
            dict: 包含处理结果和信息的字典
        """
        if image is None:
            return {'success': False, 'error': '输入图像为空'}
        
        start_time = time.time()
        
        # 使用默认参数
        if params is None:
            params = self.PRESETS['学生证标准'].copy()
        
        result = image.copy()
        face_rect = None
        processing_steps = []
        
        try:
            # 步骤1: 人脸检测
            success, face_rect = self.face_detector.detect_face(result)
            
            processing_steps.append({
                'step': '人脸检测',
                'success': success,
                'data': face_rect
            })
            
            # 步骤2: 尺寸标准化和裁剪
            size_name = params.get('size', '一寸')
            keep_full = params.get('keep_full', False)
            crop_mode = params.get('crop_mode', 'face')
            result = self.cropper.auto_crop(
                result,
                size_name,
                use_face_detection=success,
                keep_full=keep_full,
                crop_mode=crop_mode,
            )
            processing_steps.append({
                'step': '尺寸标准化',
                'success': True,
                'data': {'size': size_name, 'dimensions': result.shape[:2]}
            })

            success2, face_rect2 = self.face_detector.detect_face(result)
            face_rect = face_rect2 if success2 else None
            processing_steps.append({
                'step': '裁剪后人脸检测',
                'success': success2,
                'data': face_rect2
            })
            
            # 步骤3: 背景替换
            bg_color = params.get('bg_color', 'blue')
            bg_method = params.get('bg_method', 'grabcut')
            feather_radius = params.get('feather_radius', 5)

            rect = None
            if bg_method == 'grabcut' and face_rect is not None:
                x, y, w, h = face_rect
                rx = max(0, int(x - w * 0.6))
                ry = max(0, int(y - h * 0.6))
                rw = int(w * 2.2)
                rh = int(h * 3.2)
                rw = min(rw, result.shape[1] - rx)
                rh = min(rh, result.shape[0] - ry)
                if rw > 0 and rh > 0:
                    rect = (rx, ry, rw, rh)

            result = self.bg_remover.auto_remove_and_replace(
                result,
                bg_color,
                bg_method,
                feather_radius=feather_radius,
                rect=rect,
            )
            processing_steps.append({
                'step': '背景替换',
                'success': True,
                'data': {'color': bg_color, 'method': bg_method}
            })
            
            # 步骤4: 图像增强
            if params.get('auto_enhance', True):
                result = self.enhancer.auto_enhance(result, face_rect)
                processing_steps.append({
                    'step': '自动增强',
                    'success': True
                })
            else:
                # 手动调整
                brightness = params.get('brightness', 0)
                contrast = params.get('contrast', 0)
                if brightness != 0 or contrast != 0:
                    result = self.enhancer.adjust_brightness_contrast(
                        result, brightness, contrast
                    )
                    processing_steps.append({
                        'step': '亮度/对比度调整',
                        'success': True,
                        'data': {'brightness': brightness, 'contrast': contrast}
                    })
                
                # 去噪
                denoise_method = params.get('denoise_method', 'bilateral')
                if denoise_method == 'median':
                    result = self.enhancer.denoise_median(result, kernel_size=3)
                elif denoise_method == 'gaussian':
                    result = self.enhancer.denoise_gaussian(result, kernel_size=5)
                elif denoise_method == 'bilateral':
                    result = self.enhancer.denoise_bilateral(result)
                
                if denoise_method:
                    processing_steps.append({
                        'step': '去噪',
                        'success': True,
                        'data': {'method': denoise_method}
                    })
            
            elapsed_time = time.time() - start_time
            
            # 保存到历史记录
            self._add_to_history(result)
            
            return {
                'success': True,
                'image': result,
                'original': image,
                'face_rect': face_rect,
                'processing_steps': processing_steps,
                'processing_time': elapsed_time,
                'params': params
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_steps': processing_steps
            }
    
    def process_batch(self, images, params=None, progress_callback=None):
        """
        批量处理证件照
        
        Args:
            images: 图像列表
            params: 处理参数
            progress_callback: 进度回调函数(current, total)
            
        Returns:
            list: 处理结果列表
        """
        results = []
        total = len(images)
        
        for i, image in enumerate(images):
            result = self.process_single(image, params)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def quick_process(self, image, preset_name='学生证标准'):
        """
        使用预设快速处理
        
        Args:
            image: numpy数组格式的图像
            preset_name: 预设名称
            
        Returns:
            dict: 处理结果
        """
        params = self.PRESETS.get(preset_name, self.PRESETS['学生证标准'])
        return self.process_single(image, params)
    
    def get_processing_info(self):
        """获取处理信息"""
        return {
            'history_size': len(self.history),
            'current_index': self.history_index,
            'can_undo': self.can_undo(),
            'can_redo': self.can_redo()
        }


class IDPhotoUtils:
    """证件照工具类"""
    
    @staticmethod
    def load_image(path):
        """加载图像"""
        try:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                return None, f"无法加载图像: {path}"
            return image, None
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def save_image(image, path, quality=95, dpi=300):
        """
        保存图像 - 设置 300DPI 元信息
        
        Args:
            image: numpy数组格式的图像 (BGR)
            path: 保存路径
            quality: JPEG 压缩质量 (0-100)
            dpi: DPI 值，默认300
            
        Returns:
            tuple: (success, error_message)
        """
        try:
            ext = os.path.splitext(path)[1].lower()
            
            # 使用 PIL 保存以设置 DPI 元信息
            if _PIL_AVAILABLE:
                # BGR -> RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_img = PILImage.fromarray(rgb_image)
                
                # 设置 DPI
                dpi_tuple = (dpi, dpi)
                
                if ext in ('.jpg', '.jpeg'):
                    pil_img.save(path, 'JPEG', quality=quality, dpi=dpi_tuple)
                elif ext == '.png':
                    # PNG 压缩等级: 0-9，从 quality 转换
                    compress_level = max(0, min(9, int((100 - quality) / 11)))
                    pil_img.save(path, 'PNG', compress_level=compress_level, dpi=dpi_tuple)
                elif ext in ('.bmp', '.dib'):
                    pil_img.save(path, 'BMP', dpi=dpi_tuple)
                else:
                    pil_img.save(path, dpi=dpi_tuple)
            else:
                # 回退到 OpenCV 保存（无 DPI 元信息）
                if ext in ('.jpg', '.jpeg'):
                    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
                elif ext == '.png':
                    compress_level = max(0, min(9, int((100 - quality) / 11)))
                    cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
                else:
                    cv2.imwrite(path, image)
            
            return True, None
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def set_image_dpi(image, dpi=300):
        """
        返回带 DPI 元信息的 PIL 图像（用于需要 DPI 的场景）
        
        Args:
            image: numpy数组格式的图像 (BGR)
            dpi: DPI 值
            
        Returns:
            PIL.Image: 带 DPI 元信息的 PIL 图像
        """
        if not _PIL_AVAILABLE:
            return None
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb_image)
        pil_img.info['dpi'] = (dpi, dpi)
        return pil_img
    
    @staticmethod
    def get_image_info(image):
        """获取图像信息"""
        if image is None:
            return None
        
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        return {
            'width': w,
            'height': h,
            'channels': channels,
            'aspect_ratio': w / h,
            'size': f"{w}×{h}"
        }
    
    @staticmethod
    def create_comparison(original, processed, orientation='horizontal'):
        """创建对比图"""
        if original is None or processed is None:
            return None
        
        # 确保两图尺寸相同
        h1, w1 = original.shape[:2]
        h2, w2 = processed.shape[:2]
        
        if (h1, w1) != (h2, w2):
            processed = cv2.resize(processed, (w1, h1))
        
        if orientation == 'horizontal':
            # 水平拼接
            comparison = np.hstack((original, processed))
        else:
            # 垂直拼接
            comparison = np.vstack((original, processed))
        
        return comparison
    
    @staticmethod
    def add_watermark(image, text="证件照", position='bottom-right', 
                      font_scale=0.5, color=(128, 128, 128)):
        """添加水印"""
        if image is None:
            return None
        
        result = image.copy()
        h, w = result.shape[:2]
        
        # 计算位置
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                    font_scale, 1)[0]
        
        if position == 'bottom-right':
            x = w - text_size[0] - 10
            y = h - 10
        elif position == 'bottom-left':
            x = 10
            y = h - 10
        elif position == 'top-right':
            x = w - text_size[0] - 10
            y = text_size[1] + 10
        else:  # top-left
            x = 10
            y = text_size[1] + 10
        
        # 添加文字
        cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 1, cv2.LINE_AA)
        
        return result


# 测试代码
if __name__ == "__main__":
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 创建处理器
    processor = IDPhotoProcessor()
    
    # 测试处理
    result = processor.quick_process(test_image, '学生证标准')
    
    print(f"处理成功: {result['success']}")
    if result['success']:
        print(f"处理时间: {result['processing_time']:.3f}s")
        print(f"处理步骤: {len(result['processing_steps'])}")
        for step in result['processing_steps']:
            print(f"  - {step['step']}: {'成功' if step['success'] else '失败'}")
