# MedianFiltering.py
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class MedianFilter:
    @staticmethod
    def apply_median_filter(qimage, kernel_size=3):
        """
        对QImage应用中值滤波
        :param qimage: 输入的QImage对象
        :param kernel_size: 滤波器大小(奇数)
        :return: 滤波后的QPixmap对象
        """
        if qimage.isNull():
            return None
        
        # 将QImage转换为OpenCV Mat格式
        cv_img = MedianFilter.qimage_to_cvmat(qimage)
        
        # 应用中值滤波
        if len(cv_img.shape) == 3:  # 如果是彩色图像
            # 对每个通道分别应用中值滤波
            filtered_channels = []
            for i in range(3):
                channel = cv_img[:, :, i]
                filtered_channel = cv2.medianBlur(channel, kernel_size)
                filtered_channels.append(filtered_channel)
            filtered_img = cv2.merge(filtered_channels)
        else:  # 如果是灰度图像
            filtered_img = cv2.medianBlur(cv_img, kernel_size)
        
        # 将OpenCV图像转换回QPixmap
        return MedianFilter.cvmat_to_qpixmap(filtered_img)

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
            return MedianFilter.qimage_to_cvmat(qimage)

    @staticmethod
    def cvmat_to_qpixmap(cv_img):
        """
        将OpenCV Mat转换为QPixmap
        :param cv_img: OpenCV Mat对象
        :return: QPixmap对象
        """
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