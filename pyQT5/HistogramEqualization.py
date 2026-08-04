# HistogramEqualization.py
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class HistogramEqualizer:
    @staticmethod
    def apply_histogram_equalization(qimage):
        """
        对QImage应用直方图均衡化
        :param qimage: 输入的QImage对象
        :return: 均衡化后的QPixmap对象
        """
        if qimage.isNull():
            return None
        
        try:
            # 将QImage转换为OpenCV Mat格式
            # 确保转换为可写的numpy数组
            cv_img = np.array(HistogramEqualizer.qimage_to_cvmat(qimage))
            
            # 处理彩色图像和灰度图像的不同情况
            if len(cv_img.shape) == 3:  # 彩色图像
                # 转换为YCrCb色彩空间，只对亮度通道(Y)进行均衡化
                ycrcb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YCrCb)
                channels = list(cv2.split(ycrcb))  # 确保转换为列表
                channels[0] = cv2.equalizeHist(channels[0])
                ycrcb = cv2.merge(channels)
                equalized_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            else:  # 灰度图像
                equalized_img = cv2.equalizeHist(cv_img)
            
            # 将OpenCV图像转换回QPixmap
            return HistogramEqualizer.cvmat_to_qpixmap(equalized_img)
        except Exception as e:
            print(f"直方图均衡化错误: {str(e)}")
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
            return HistogramEqualizer.qimage_to_cvmat(qimage)

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