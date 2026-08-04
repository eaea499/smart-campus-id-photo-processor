# Adjust.py
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap, qRgb
from PyQt5.QtCore import Qt

class ImageAdjuster:
    @staticmethod
    def imadjust(qimage, low_in=0.0, high_in=1.0, low_out=0.0, high_out=1.0, gamma=1.0):
        """
        实现类似MATLAB的imadjust函数功能
        :param qimage: 输入的QImage对象
        :param low_in: 输入图像的下限(0-1)
        :param high_in: 输入图像的上限(0-1)
        :param low_out: 输出图像的下限(0-1)
        :param high_out: 输出图像的上限(0-1)
        :param gamma: 伽马校正值(>1变暗，<1变亮)
        :return: 调整后的QPixmap对象
        """
        if qimage.isNull():
            return None
        
        try:
            # 将QImage转换为OpenCV Mat格式
            cv_img = ImageAdjuster.qimage_to_cvmat(qimage)
            
            # 归一化到0-1范围
            img_normalized = cv_img.astype(np.float32) / 255.0
            
            # 应用imadjust算法
            if gamma == 1:
                # 线性映射
                adjusted_img = np.clip((img_normalized - low_in) / (high_in - low_in), 0, 1)
                adjusted_img = adjusted_img * (high_out - low_out) + low_out
            else:
                # 非线性gamma校正
                adjusted_img = np.power(np.clip((img_normalized - low_in) / (high_in - low_in), 0, 1), gamma)
                adjusted_img = adjusted_img * (high_out - low_out) + low_out
            
            # 转换回0-255范围
            adjusted_img = np.clip(adjusted_img * 255, 0, 255).astype(np.uint8)
            
            # 将OpenCV图像转换回QPixmap
            return ImageAdjuster.cvmat_to_qpixmap(adjusted_img)
        except Exception as e:
            print(f"对比度调整错误: {str(e)}")
            return None

    @staticmethod
    def auto_adjust(qimage, gamma=1.0):
        """
        自动调整对比度(类似stretchlim+imadjust)
        :param qimage: 输入的QImage对象
        :param gamma: 伽马校正值
        :return: 调整后的QPixmap对象
        """
        if qimage.isNull():
            return None
            
        try:
            # 将QImage转换为OpenCV Mat格式
            cv_img = ImageAdjuster.qimage_to_cvmat(qimage)
            
            # 计算自动调整的输入范围(类似stretchlim)
            if len(cv_img.shape) == 2:  # 灰度图像
                low, high = np.percentile(cv_img, (2, 98))
            else:  # 彩色图像
                low, high = np.percentile(cv_img, (2, 98), axis=(0,1))
            
            # 归一化到0-1范围
            low_in = np.array(low) / 255.0
            high_in = np.array(high) / 255.0
            
            # 调用imadjust
            return ImageAdjuster.imadjust(qimage, 
                                        low_in=low_in.min(), 
                                        high_in=high_in.max(), 
                                        gamma=gamma)
        except Exception as e:
            print(f"自动对比度调整错误: {str(e)}")
            return None

    @staticmethod
    def qimage_to_cvmat(qimage):
        """
        将QImage转换为OpenCV Mat格式
        :param qimage: QImage对象
        :return: OpenCV Mat对象
        """
        width = qimage.width()
        height = qimage.height()
        
        # 根据图像格式进行不同处理
        if qimage.format() in (QImage.Format_RGB32, QImage.Format_ARGB32, 
                              QImage.Format_ARGB32_Premultiplied):
            ptr = qimage.bits()
            ptr.setsize(qimage.byteCount())
            arr = np.array(ptr).reshape(height, width, 4)
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif qimage.format() == QImage.Format_RGB888:
            ptr = qimage.bits()
            ptr.setsize(qimage.byteCount())
            arr = np.array(ptr).reshape(height, width, 3)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif qimage.format() == QImage.Format_Indexed8:
            ptr = qimage.bits()
            ptr.setsize(qimage.byteCount())
            return np.array(ptr).reshape(height, width)
        else:
            # 不支持的格式转换为RGB888再处理
            qimage = qimage.convertToFormat(QImage.Format_RGB888)
            return ImageAdjuster.qimage_to_cvmat(qimage)

    @staticmethod
    def cvmat_to_qpixmap(cv_img):
        """
        将OpenCV Mat转换为QPixmap
        :param cv_img: OpenCV Mat对象
        :return: QPixmap对象
        """
        # 确保输入是numpy数组
        if not isinstance(cv_img, np.ndarray):
            cv_img = np.array(cv_img)
            
        # 处理灰度图像
        if len(cv_img.shape) == 2:
            height, width = cv_img.shape
            bytes_per_line = width
            q_img = QImage(cv_img.data, width, height, 
                         bytes_per_line, QImage.Format_Grayscale8)
        # 处理彩色图像
        else:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_img.shape
            bytes_per_line = channels * width
            q_img = QImage(rgb_img.data, width, height, 
                         bytes_per_line, QImage.Format_RGB888)
        
        return QPixmap.fromImage(q_img)