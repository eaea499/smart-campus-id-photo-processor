# ImageEnhancer.py
"""
图像增强模块
用于证件照的亮度、对比度调节和人脸去噪
"""

import cv2
import numpy as np


class ImageEnhancer:
    """图像增强器类"""
    
    def __init__(self):
        """初始化图像增强器"""
        pass
    
    def adjust_brightness_contrast(self, image, brightness=0, contrast=0):
        """
        调整图像亮度和对比度
        
        Args:
            image: numpy数组格式的图像
            brightness: 亮度调整值 (-100 到 100)
            contrast: 对比度调整值 (-100 到 100)
            
        Returns:
            numpy数组: 调整后的图像
        """
        if image is None:
            return None
        
        # 转换到float32进行处理
        result = image.astype(np.float32)
        
        # 调整对比度
        if contrast != 0:
            # 对比度因子: 1.0 表示无变化
            contrast_factor = (100.0 + contrast) / 100.0
            result = (result - 128) * contrast_factor + 128
        
        # 调整亮度
        if brightness != 0:
            result = result + brightness
        
        # 裁剪到有效范围
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def adjust_gamma(self, image, gamma=1.0):
        """
        Gamma校正
        
        Args:
            image: numpy数组格式的图像
            gamma: Gamma值 (<1变亮, >1变暗)
            
        Returns:
            numpy数组: 校正后的图像
        """
        if image is None:
            return None
        
        # 创建查找表
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]).astype("uint8")
        
        # 应用查找表
        return cv2.LUT(image, table)
    
    def auto_adjust_brightness(self, image, target_mean=128):
        """
        自动调整亮度到目标均值
        
        Args:
            image: numpy数组格式的图像
            target_mean: 目标亮度均值
            
        Returns:
            numpy数组: 调整后的图像
        """
        if image is None:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_mean = np.mean(gray)
        
        # 计算需要调整的亮度值
        brightness = target_mean - current_mean
        
        return self.adjust_brightness_contrast(image, brightness=brightness)
    
    def histogram_equalization(self, image):
        """
        直方图均衡化
        
        Args:
            image: numpy数组格式的图像
            
        Returns:
            numpy数组: 均衡化后的图像
        """
        if image is None:
            return None
        
        if len(image.shape) == 3:
            # 彩色图像：转换到YCrCb，只对Y通道均衡化
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            # 灰度图像
            return cv2.equalizeHist(image)
    
    def clahe(self, image, clip_limit=2.0, tile_size=8):
        """
        自适应直方图均衡化（CLAHE）
        
        Args:
            image: numpy数组格式的图像
            clip_limit: 对比度限制
            tile_size: 网格大小
            
        Returns:
            numpy数组: 处理后的图像
        """
        if image is None:
            return None
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                                tileGridSize=(tile_size, tile_size))
        
        if len(image.shape) == 3:
            # 彩色图像
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # 灰度图像
            return clahe.apply(image)
    
    def denoise_median(self, image, kernel_size=3):
        """
        中值滤波去噪
        
        Args:
            image: numpy数组格式的图像
            kernel_size: 滤波核大小（奇数）
            
        Returns:
            numpy数组: 去噪后的图像
        """
        if image is None:
            return None
        
        return cv2.medianBlur(image, kernel_size)
    
    def denoise_gaussian(self, image, kernel_size=5, sigma=0):
        """
        高斯滤波去噪
        
        Args:
            image: numpy数组格式的图像
            kernel_size: 滤波核大小（奇数）
            sigma: 标准差（0表示自动计算）
            
        Returns:
            numpy数组: 去噪后的图像
        """
        if image is None:
            return None
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def denoise_bilateral(self, image, d=9, sigma_color=75, sigma_space=75):
        """
        双边滤波去噪（保留边缘）
        
        Args:
            image: numpy数组格式的图像
            d: 滤波直径
            sigma_color: 颜色空间标准差
            sigma_space: 坐标空间标准差
            
        Returns:
            numpy数组: 去噪后的图像
        """
        if image is None or image.size == 0:
            return None
        
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def denoise_nlmeans(self, image, h=10, h_color=10, template_size=7, search_size=21):
        """
        非局部均值去噪（高质量但较慢）
        
        Args:
            image: numpy数组格式的图像
            h: 亮度滤波强度
            h_color: 颜色滤波强度
            template_size: 模板大小
            search_size: 搜索窗口大小
            
        Returns:
            numpy数组: 去噪后的图像
        """
        if image is None:
            return None
        
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                image, None, h, h_color, template_size, search_size
            )
        else:
            return cv2.fastNlMeansDenoising(
                image, None, h, template_size, search_size
            )
    
    def sharpen(self, image, amount=1.0):
        """
        图像锐化
        
        Args:
            image: numpy数组格式的图像
            amount: 锐化强度
            
        Returns:
            numpy数组: 锐化后的图像
        """
        if image is None:
            return None
        
        # 使用Unsharp Masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(image, 1 + amount, gaussian, -amount, 0)
        
        return sharpened
    
    def auto_enhance(self, image, face_region=None):
        """
        自动增强图像（针对证件照优化）
        
        Args:
            image: numpy数组格式的图像
            face_region: 人脸区域 (x, y, w, h)，用于针对性处理
            
        Returns:
            numpy数组: 增强后的图像
        """
        if image is None:
            return None
        
        result = image.copy()
        
        # 1. 自动亮度调整
        result = self.auto_adjust_brightness(result, target_mean=135)
        
        # 2. 自适应直方图均衡化（轻度）
        result = self.clahe(result, clip_limit=1.5, tile_size=8)
        
        # 3. 双边滤波去噪（保留边缘）
        result = self.denoise_bilateral(result, d=9, sigma_color=75, sigma_space=75)
        
        # 4. 轻度锐化
        result = self.sharpen(result, amount=0.5)
        
        # 5. 如果提供了人脸区域，针对性增强人脸
        if face_region is not None:
            result = self._enhance_face_region(result, face_region)
        
        return result
    
    def _enhance_face_region(self, image, face_region):
        """
        针对性增强人脸区域
        
        Args:
            image: numpy数组格式的图像
            face_region: 人脸区域 (x, y, w, h)
            
        Returns:
            numpy数组: 增强后的图像
        """
        if image is None or image.size == 0 or face_region is None:
            return image

        x, y, w, h = face_region
        if w <= 0 or h <= 0:
            return image
        if x >= image.shape[1] or y >= image.shape[0]:
            return image
        
        # 扩展人脸区域（包含部分颈部和肩膀）
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.3)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)

        if x1 >= x2 or y1 >= y2:
            return image
        
        # 提取人脸区域
        face_roi = image[y1:y2, x1:x2]
        if face_roi is None or face_roi.size == 0:
            return image
        
        # 对人脸区域进行轻度增强
        denoised_roi = self.denoise_bilateral(face_roi, d=5, sigma_color=50, sigma_space=50)
        if denoised_roi is None or denoised_roi.size == 0:
            return image
        face_roi = self.sharpen(denoised_roi, amount=0.3)
        
        # 放回原图
        result = image.copy()
        result[y1:y2, x1:x2] = face_roi
        
        return result
    
    def white_balance(self, image):
        """
        自动白平衡
        
        Args:
            image: numpy数组格式的图像
            
        Returns:
            numpy数组: 白平衡后的图像
        """
        if image is None:
            return None
        
        # 使用灰度世界假设
        result = image.astype(np.float32)
        
        # 计算各通道均值
        mean_b = np.mean(result[:, :, 0])
        mean_g = np.mean(result[:, :, 1])
        mean_r = np.mean(result[:, :, 2])
        
        # 计算灰度均值
        mean_gray = (mean_b + mean_g + mean_r) / 3.0
        mean_b = max(mean_b, 1e-6)
        mean_g = max(mean_g, 1e-6)
        mean_r = max(mean_r, 1e-6)
        
        # 调整各通道
        result[:, :, 0] = result[:, :, 0] * (mean_gray / mean_b)
        result[:, :, 1] = result[:, :, 1] * (mean_gray / mean_g)
        result[:, :, 2] = result[:, :, 2] * (mean_gray / mean_r)
        
        # 裁剪到有效范围
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def skin_smoothing(self, image, face_region, strength=0.5):
        """
        皮肤平滑处理（美颜效果）
        
        Args:
            image: numpy数组格式的图像
            face_region: 人脸区域
            strength: 平滑强度 (0-1)
            
        Returns:
            numpy数组: 处理后的图像
        """
        if image is None or image.size == 0 or face_region is None:
            return image
        
        x, y, w, h = face_region
        if w <= 0 or h <= 0:
            return image
        
        # 提取人脸区域
        face_roi = image[y:y+h, x:x+w]
        if face_roi is None or face_roi.size == 0:
            return image
        
        # 双边滤波平滑皮肤
        smoothed = cv2.bilateralFilter(face_roi, d=9, 
                                       sigmaColor=75*strength, 
                                       sigmaSpace=75*strength)
        
        # 混合原图和平滑图
        result = image.copy()
        result[y:y+h, x:x+w] = cv2.addWeighted(face_roi, 1-strength, 
                                                smoothed, strength, 0)
        
        return result


# 测试代码
if __name__ == "__main__":
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    enhancer = ImageEnhancer()
    
    # 测试自动增强
    result = enhancer.auto_enhance(test_image)
    print(f"Result shape: {result.shape if result is not None else 'None'}")
    
    # 测试亮度调整
    result2 = enhancer.adjust_brightness_contrast(test_image, brightness=20, contrast=10)
    print(f"Brightness adjusted shape: {result2.shape if result2 is not None else 'None'}")
