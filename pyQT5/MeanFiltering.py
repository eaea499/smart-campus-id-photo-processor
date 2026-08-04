import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap

class MeanFilter:
    """
    均值滤波器类，封装了QImage的均值滤波处理功能
    """
    def __init__(self):
        pass

    def apply_mean_filter(self, qimage, kernel_size=3):
        """
        应用均值滤波到QImage对象
        
        参数:
            qimage: QImage对象
            kernel_size: 滤波核大小(奇数)，默认为3
            
        返回:
            QPixmap: 滤波后的图像，如果出错返回None
        """
        try:
            # 将QImage转换为numpy数组
            img = self._qimage_to_numpy(qimage)
            
            # 应用OpenCV的均值滤波
            filtered_img = cv2.blur(img, (kernel_size, kernel_size))
            
            # 将numpy数组转换回QPixmap
            return self._numpy_to_qpixmap(filtered_img)
        
        except Exception as e:
            print(f"均值滤波错误: {str(e)}")
            return None

    def _qimage_to_numpy(self, qimage):
        """将QImage转换为numpy数组(内部方法)"""
        qimage = qimage.convertToFormat(QImage.Format_RGB32)
        width = qimage.width()
        height = qimage.height()
        
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    def _numpy_to_qpixmap(self, img):
        """将numpy数组转换为QPixmap(内部方法)"""
        if len(img.shape) == 2:  # 灰度图
            qimage = QImage(img.data, img.shape[1], img.shape[0], 
                          img.strides[0], QImage.Format_Grayscale8)
        else:  # 彩色图
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimage = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], 
                          img_rgb.strides[0], QImage.Format_RGB888)
        return QPixmap.fromImage(qimage)