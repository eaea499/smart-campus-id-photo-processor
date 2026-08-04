import sys
import os
import numpy as np
import cv2
import time
from PyQt5.QtWidgets import (QMainWindow, QApplication, QFileDialog, QMessageBox,
                             QLabel, QAction, QMenu, QSizePolicy, QVBoxLayout,
                             QHBoxLayout, QWidget, QDialog, QPushButton)
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import Qt, QPoint, QSize, QEvent, QObject
from untitled import Ui_MainWindow
from PyQt5.QtMultimedia import QCamera, QCameraImageCapture, QCameraInfo, QMultimedia
from PyQt5.QtMultimediaWidgets import QCameraViewfinder
from PyQt5.QtMultimedia import QImageEncoderSettings

#导入自己的函数
from gray import ImageConverter
from MedianFiltering import MedianFilter
from GaussianFiltering import GaussianFilter
from HistogramEqualization import HistogramEqualizer
from Adjust import ImageAdjuster
from MeanFiltering import MeanFilter


class CameraDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("摄像头预览")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        # 创建布局
        layout = QVBoxLayout(self)

        # 创建取景器
        self.viewfinder = QCameraViewfinder()
        self.viewfinder.setMinimumSize(640, 480)
        layout.addWidget(self.viewfinder)

        # 创建按钮
        self.captureButton = QPushButton("拍摄照片")
        self.cancelButton = QPushButton("取消")

        # 按钮布局
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.captureButton)
        buttonLayout.addWidget(self.cancelButton)
        layout.addLayout(buttonLayout)


class _SmartPaintFilter(QObject):
    """事件过滤器：拦截 label 的 paintEvent，用原始 pixmap 按当前尺寸缩放绘制。
    优先使用 smart_original_pixmap（处理后的原图），回退到 label.pixmap()。
    这样避免外部反复 setPixmap(scaled) 造成的多帧重绘抖动。"""
    def __init__(self, label):
        super().__init__(label)
        self._label = label
        label._smart_painter_ref = self

    def eventFilter(self, obj, event):
        if obj is self._label and event.type() == QEvent.Paint:
            pix = self._label.property('smart_original_pixmap')
            if pix is None or pix.isNull():
                # 回退：如果 smart 版本不存在，用 label.pixmap() 也可以（原始图）
                pix = self._label.pixmap()
            if pix is not None and not pix.isNull():
                from PyQt5.QtGui import QPainter
                painter = QPainter(self._label)
                painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
                lw = max(1, self._label.width() - 6)
                lh = max(1, self._label.height() - 6)
                pw, ph = pix.width(), pix.height()
                if pw > 0 and ph > 0:
                    scale = min(lw / pw, lh / ph)
                    dw, dh = int(pw * scale), int(ph * scale)
                    dx = (self._label.width() - dw) // 2
                    dy = (self._label.height() - dh) // 2
                    painter.drawPixmap(dx, dy, dw, dh, pix)
                painter.end()
                return True
        return super().eventFilter(obj, event)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        #按钮连接函数
        # 存储当前显示的图像
        self.current_image = None
        # 连接灰度转换按钮
        self.pushButton_Gray.clicked.connect(self.convert_to_grayscale)
        # 连接中值滤波按钮
        self.pushButton_MedianFiltering.clicked.connect(self.apply_median_filter)
        # 连接高斯滤波按钮
        self.pushButton_GaussianFiltering.clicked.connect(self.apply_gaussian_filter)
        # 连接均值滤波按钮
        self.pushButton_MeanFiltering.clicked.connect(self.apply_mean_filter)
        # 连接直方图均衡化按钮
        self.pushButton_HistogramEqualization.clicked.connect(self.apply_histogram_equalization)
        # 连接对比度调整按钮
        self.pushButton_Adjust.clicked.connect(self.adjust_image_contrast)
        # 连接提取轮廓按钮
        self.pushButton_ExtractOutline.clicked.connect(self.extract_outline)
        # 连接提取主体按钮
        self.pushButton_ExtractSubject.clicked.connect(self.extract_subject)
         # 连接处理方法2按钮
        self.pushButton_Dispose2.clicked.connect(self.apply_dispose2)
        # 连接计算尺寸按钮
        self.pushButton_Caculate1.clicked.connect(self.calculate_dimensions)

        # 初始化摄像头相关变量
        self.camera = None
        self.imageCapture = None
        self.current_camera_label = None
        self.camera_dialog = None

        # 调整布局保持UI设计器效果
        self.adjust_layouts()

        # 设置按钮图标
        self.setup_icons()

        # 设置界面样式
        self.setup_styles()

        # 连接信号与槽
        self.connect_signals()

    #灰度
    def convert_to_grayscale(self):
        pixmap = self.label_Pretreatment_Image.pixmap()
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(self, "警告", "请先加载图像!")
            return
        try:
            qimage = pixmap.toImage()
            gray_pixmap = ImageConverter.convert_to_grayscale(qimage)
            if gray_pixmap and not gray_pixmap.isNull():
                # 只设置原始 pixmap — 缩放交给 _SmartPaintFilter 在 paintEvent 中做
                self._set_label_image(self.label_Pretreatment_Image, gray_pixmap)
                self.current_image = gray_pixmap
            else:
                QMessageBox.critical(self, "错误", "灰度转换失败!")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"灰度转换时发生错误: {str(e)}")

    #中值滤波
    def apply_median_filter(self):
        pixmap = self.label_Pretreatment_Image.pixmap()
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(self, "警告", "请先加载图像!")
            return
        try:
            qimage = pixmap.toImage()
            filtered_pixmap = MedianFilter.apply_median_filter(qimage, kernel_size=3)
            if filtered_pixmap and not filtered_pixmap.isNull():
                self._set_label_image(self.label_Pretreatment_Image, filtered_pixmap)
                self.current_image = filtered_pixmap
            else:
                QMessageBox.critical(self, "错误", "中值滤波失败!")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"中值滤波时发生错误: {str(e)}")

    #高斯滤波
    def apply_gaussian_filter(self):
        pixmap = self.label_Pretreatment_Image.pixmap()
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(self, "警告", "请先加载图像!")
            return
        try:
            qimage = pixmap.toImage()
            filtered_pixmap = GaussianFilter.apply_gaussian_filter(qimage, kernel_size=5, sigma_x=0, sigma_y=0)
            if filtered_pixmap and not filtered_pixmap.isNull():
                self._set_label_image(self.label_Pretreatment_Image, filtered_pixmap)
                self.current_image = filtered_pixmap
            else:
                QMessageBox.critical(self, "错误", "高斯滤波失败!")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"高斯滤波时发生错误: {str(e)}")

    # 均值滤波
    def apply_mean_filter(self):
        pixmap = self.label_Pretreatment_Image.pixmap()
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(self, "警告", "请先加载图像!")
            return
        try:
            qimage = pixmap.toImage()
            mean_filter = MeanFilter()
            filtered_pixmap = mean_filter.apply_mean_filter(qimage, kernel_size=3)
            if filtered_pixmap and not filtered_pixmap.isNull():
                self._set_label_image(self.label_Pretreatment_Image, filtered_pixmap)
                self.current_image = filtered_pixmap
            else:
                QMessageBox.critical(self, "错误", "均值滤波失败!")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"均值滤波时发生错误: {str(e)}")

    #直方图均衡化
    def apply_histogram_equalization(self):
        pixmap = self.label_Pretreatment_Image.pixmap()
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(self, "警告", "请先加载图像!")
            return
        try:
            qimage = pixmap.toImage()
            equalized_pixmap = HistogramEqualizer.apply_histogram_equalization(qimage)
            if equalized_pixmap and not equalized_pixmap.isNull():
                self._set_label_image(self.label_Pretreatment_Image, equalized_pixmap)
                self.current_image = equalized_pixmap
            else:
                QMessageBox.critical(self, "错误", "直方图均衡化失败!")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"直方图均衡化时发生错误: {str(e)}")


    #调整对比度
    def adjust_image_contrast(self):
        pixmap = self.label_Pretreatment_Image.pixmap()
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(self, "警告", "请先加载图像!")
            return
        try:
            qimage = pixmap.toImage()
            adjusted_pixmap = ImageAdjuster.auto_adjust(qimage, gamma=0.5)
            if adjusted_pixmap and not adjusted_pixmap.isNull():
                self._set_label_image(self.label_Pretreatment_Image, adjusted_pixmap)
                self.current_image = adjusted_pixmap
            else:
                QMessageBox.critical(self, "错误", "对比度调整失败!")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"对比度调整时发生错误: {str(e)}")

    #提取轮廓
    def extract_outline(self):
        """提取图像轮廓"""
        start_time = time.time()
        pixmap = self.label_Pretreatment_Image.pixmap()
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(self, "警告", "请先加载图像!")
            return
        try:
            threshold_text = self.textEdit_Input.toPlainText().strip()
            low_th = 0.001
            high_th = 0.21
            if threshold_text:
                try:
                    thresholds = threshold_text.split(';')
                    if len(thresholds) == 2:
                        low_th = float(thresholds[0].strip())
                        high_th = float(thresholds[1].strip())
                        QMessageBox.information(self, "成功", "阈值修改成功!")
                    else:
                        QMessageBox.warning(self, "警告", "请输入两个用分号分隔的阈值!")
                        return
                except ValueError:
                    QMessageBox.warning(self, "警告", "请输入有效的数字阈值!")
                    return
            qimage = pixmap.toImage()
            img = self.qimage_to_numpy(qimage)
            from ExtractOutline import process_image
            edge, contour = process_image(img, low_th=low_th, high_th=high_th)
            edge_pixmap = self.numpy_to_qpixmap(edge)
            contour_pixmap = self.numpy_to_qpixmap(contour)
            # 直接设置原始 pixmap — 缩放交给事件过滤器
            self._set_label_image(self.label_Edge, edge_pixmap)
            self._set_label_image(self.label_Contour, contour_pixmap)
            elapsed = time.time() - start_time
            self.statusBar().showMessage(f"处理方法1耗时: {elapsed:.3f}秒", 5000)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"提取轮廓时发生错误: {str(e)}")
        
        

    def qimage_to_numpy(self, qimage):
        """将QImage转换为numpy数组"""
        qimage = qimage.convertToFormat(QImage.Format_RGB32)
        width = qimage.width()
        height = qimage.height()
        
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    def numpy_to_qpixmap(self, img):
        """将numpy数组转换为QPixmap，支持多种图像格式"""
        # 确保图像是numpy数组
        if not isinstance(img, np.ndarray):
            raise ValueError("输入必须是numpy数组")
        
        # 处理单通道图像（二值或灰度）
        if len(img.shape) == 2:
            # 检查是否是二值图像（0和1）
            if img.dtype == np.bool_ or (img.max() <= 1 and img.min() >= 0):
                img = (img * 255).astype(np.uint8)
            # 确保是8位灰度
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # 将单通道图像转换为3通道伪RGB图像
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # 处理多通道图像（RGB/RGBA等）
        elif len(img.shape) == 3:
            # 确保是8位图像
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # 根据通道数选择格式
            channels = img.shape[2]
            if channels == 3:
                # RGB图像需要交换R和B通道（OpenCV使用BGR）
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif channels == 4:
                # RGBA图像
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            else:
                # 其他通道数，转换为RGB
                if channels > 4:
                    img = img[:, :, :3]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        else:
            raise ValueError("不支持的图像维度: {}".format(len(img.shape)))
        
        height, width = img.shape[:2]
        bytes_per_line = 3 * width  # 3通道
        qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage)

    #处理方法2
    def apply_dispose2(self):
        """调用Dispose2.py的处理流程（直接传递图像数据）"""
        start_time = time.time()
        pixmap = self.label_Pretreatment_Image.pixmap()
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(self, "警告", "请先加载图像!")
            return
        try:
            qimage = pixmap.toImage()
            img = self.qimage_to_numpy(qimage)
            from Dispose2 import image_processing_pipeline
            BW, filtered_img = image_processing_pipeline(img)
            if BW is not None:
                BW_pixmap = self.numpy_to_qpixmap(BW.astype(np.uint8) * 255)
                self._set_label_image(self.label_Dispose21, BW_pixmap)
            if filtered_img is not None:
                filtered_pixmap = self.numpy_to_qpixmap(filtered_img)
                self._set_label_image(self.label_Dispose2, filtered_pixmap)
            if BW is None or filtered_img is None:
                QMessageBox.warning(self, "警告", "图像处理失败!")
            elapsed = time.time() - start_time
            self.statusBar().showMessage(f"处理方法2耗时: {elapsed:.3f}秒", 5000)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"图像处理时发生错误: {str(e)}")

    #提取主体
    def extract_subject(self):
        """提取主体区域"""
        start_time = time.time()
        if self.radioButton_Dispose1.isChecked():
            source_label = self.label_Contour
        elif self.radioButton_Dispose2.isChecked():
            source_label = self.label_Dispose2
        else:
            source_label = self.label_Contour
        pixmap = source_label.pixmap()
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(self, "警告", "请先加载图像!")
            return
        try:
            param_text = self.textEdit_Input2.toPlainText().strip()
            close_size = 0
            if param_text:
                try:
                    params = param_text.split(';')
                    if len(params) >= 1:
                        close_size = int(params[0].strip())
                        QMessageBox.information(self, "成功", "参数修改成功!")
                    else:
                        QMessageBox.warning(self, "警告", "请输入有效的参数!")
                        return
                except ValueError:
                    QMessageBox.warning(self, "警告", "请输入有效的数字参数!")
                    return
            qimage = pixmap.toImage()
            if qimage.format() != QImage.Format_Grayscale8:
                qimage = qimage.convertToFormat(QImage.Format_Grayscale8)
            width = qimage.width()
            height = qimage.height()
            bytes_per_line = qimage.bytesPerLine()
            ptr = qimage.bits()
            ptr.setsize(qimage.byteCount())
            if bytes_per_line == width:
                arr = np.frombuffer(ptr, np.uint8).reshape((height, width))
            else:
                arr = np.zeros((height, width), dtype=np.uint8)
                for row in range(height):
                    start = row * bytes_per_line
                    end = start + width
                    arr[row, :] = np.frombuffer(ptr, np.uint8, width, start)
            from Subject import RegionAnalyzer
            processor = RegionAnalyzer(arr)
            processed_img = processor.solve(close_size=close_size)
            if processed_img is not None:
                processed_pixmap = self.numpy_to_qpixmap(processed_img)
                self._set_label_image(self.label_Subject, processed_pixmap)
            else:
                QMessageBox.warning(self, "警告", "主体提取失败!")
            elapsed = time.time() - start_time
            self.statusBar().showMessage(f"图像填充耗时: {elapsed:.3f}秒", 5000)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"主体提取时发生错误: {str(e)}")

    #计算尺寸
    def calculate_dimensions(self):
        """计算杯子与硬币尺寸"""
        start_time = time.time()
        pixmap = self.label_Subject.pixmap()
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(self, "警告", "请先提取主体图像!")
            return
        try:
            qimage = pixmap.toImage()
            img = self.qimage_to_numpy(qimage)
            from Caculate import process_image
            annotated_img, cup_width, cup_height, coin_diameter = process_image(img)
            if cup_width is None or cup_height is None or coin_diameter is None:
                QMessageBox.warning(self, "警告", "尺寸计算失败，请检查图像质量!")
                return
            if annotated_img is not None:
                annotated_pixmap = self.numpy_to_qpixmap(annotated_img)
                self._set_label_image(self.label_Caculate, annotated_pixmap)
            result_text = f"杯子宽度：{cup_width:.2f}mm\n杯子高度：{cup_height:.2f}mm\n标定物直径：{coin_diameter:.2f}mm"
            self.label_CaculateResult.setText(result_text)
            elapsed = time.time() - start_time
            self.statusBar().showMessage(f"计算尺寸耗时: {elapsed:.3f}秒", 5000)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"尺寸计算时发生错误: {str(e)}")


    def adjust_layouts(self):
        """调整布局以保持UI设计器效果同时实现自适应"""
        # 1. 确保中央部件设置正确
        if not self.centralwidget.layout():
            central_layout = QVBoxLayout(self.centralwidget)
            central_layout.setContentsMargins(0, 0, 0, 0)
        else:
            central_layout = self.centralwidget.layout()

        # 2. 设置主窗口布局策略
        self.centralwidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 3. 调整stackedWidget的布局策略
        self.stackedWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 4. 调整各页面控件的布局策略
        self.adjust_page_layouts()

    def adjust_page_layouts(self):
        """调整各页面控件的布局策略"""
        # 首页
        self.label_Home.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 预处理页
        self.label_Pretreatment.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.label_Pretreatment_Image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frame_Pretreatment_1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        # 图像处理页
        self.label_2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        # self.label_Edge.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.label_Contour.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frame_Dispose.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        # 尺寸计算页
        self.label_3.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.label_Caculate.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_CaculateResult.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # 固定按钮大小
        self.pushButton_Caculate1.setFixedSize(60, 60)

    def setup_icons(self):
        # 设置按钮图标
        self.pushButton_Home.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/首页-copy.png"))
        self.pushButton_Pretreatment.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/预处理.png"))
        self.pushButton_Dispose.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/图像处理.png"))
        self.pushButton_Caculate.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/cube.png"))
        self.pushButton_Close.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/退出.png"))
        self.pushButton_Gray.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/灰度-灰.png"))
        self.pushButton_MedianFiltering.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/ocean中值滤波.png"))
        self.pushButton_GaussianFiltering.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/高斯滤波.png"))
        self.pushButton_HistogramEqualization.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/直方图均衡化.png"))
        self.pushButton_Adjust.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/对比度.png"))
        self.pushButton_ExtractOutline.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/生成轮廓.png"))
        self.pushButton_ExtractSubject.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/主要信息.png"))
        self.pushButton_MeanFiltering.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/trend.png"))
        self.pushButton_Hole.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/填色.png"))
        self.pushButton_Dispose2.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/列表-处理方法.png"))

        self.pushButton_Caculate1.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/ruler-pencil.png"))
        self.pushButton_Caculate1.setIconSize(QSize(55, 55))

        

        # 菜单栏图标
        self.action_Open.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/导入.png"))
        self.action_Save.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/保存.png"))
        self.action_Camera.setIcon(QIcon("D:/homework/ima/pythonProject2/图标/摄像头.png"))

        # 主窗口图标
        self.setWindowIcon(QIcon("D:/homework/ima/pythonProject2/图标/量水杯-01.png"))
        self.setWindowTitle("水杯尺寸测量")

    def setup_styles(self):
        # 设置Frame背景颜色
        self.frame_Pretreatment.setStyleSheet("background-color: rgb(173, 216, 230);")
        self.frame_Pretreatment_1.setStyleSheet("background-color:#FFF87D;")
        self.frame_Dispose.setStyleSheet("background-color:#FFF87D;")

        # StackWidget背景颜色
        self.stackedWidget.setStyleSheet("background-color:#DEFFDE;")

        # 输出框背景颜色
        self.label_CaculateResult.setStyleSheet("background-color:#D4D4D4;")

        # 窗口背景颜色
        self.setStyleSheet("background-color: #D2CCFF;")

        # 图片label背景颜色
        self.label_Pretreatment_Image.setStyleSheet("background-color:#E2EDF8;")
        self.label_Edge.setStyleSheet("background-color:#E2EDF8;")
        self.label_Contour.setStyleSheet("background-color:#E2EDF8;")
        self.label_Caculate.setStyleSheet("background-color:#E2EDF8;")
        self.label_Subject.setStyleSheet("background-color:#E2EDF8;")
        self.label_Dispose2.setStyleSheet("background-color:#E2EDF8;")
        self.label_Dispose21.setStyleSheet("background-color:#E2EDF8;")


        # 设置右键菜单
        self.label_Pretreatment_Image.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_Edge.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_Contour.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_Caculate.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_Subject.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_Dispose2.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_Dispose21.setContextMenuPolicy(Qt.CustomContextMenu)

    def connect_signals(self):
        # 关闭界面
        self.pushButton_Close.clicked.connect(self.close)

        # 连接摄像头
        self.action_Camera.triggered.connect(self.openCameraAndCapture)

        # 右键菜单
        self.label_Pretreatment_Image.customContextMenuRequested.connect(self.showImageContextMenu)
        self.label_Edge.customContextMenuRequested.connect(self.showImageContextMenu)
        self.label_Contour.customContextMenuRequested.connect(self.showImageContextMenu)
        self.label_Caculate.customContextMenuRequested.connect(self.showImageContextMenu)
        self.label_Subject.customContextMenuRequested.connect(self.showImageContextMenu)
        self.label_Dispose2.customContextMenuRequested.connect(self.showImageContextMenu)
        self.label_Dispose21.customContextMenuRequested.connect(self.showImageContextMenu)

        # 菜单栏
        self.action_Open.triggered.connect(self.openImageAction)
        self.action_Save.triggered.connect(self.saveCurrentImage)

        # 页面切换按钮
        self.pushButton_Home.clicked.connect(lambda: self.switch_page(0))
        self.pushButton_Pretreatment.clicked.connect(lambda: self.switch_page(1))
        self.pushButton_Dispose.clicked.connect(lambda: self.switch_page(2))
        self.pushButton_Hole.clicked.connect(lambda: self.switch_page(3))
        self.pushButton_Caculate.clicked.connect(lambda: self.switch_page(4))

    def switch_page(self, index):
        """切换页面 — 不再调用 processEvents，不再手动刷新每个 label，
        因为我们已经通过 _SmartPaintFilter 在 paintEvent 中按需缩放了原始 pixmap"""
        # 直接切换，不做额外刷新
        self.stackedWidget.setCurrentIndex(index)

    def _install_smart_painter(self, label):
        """给指定 label 安装智能绘制过滤器：保存原始 pixmap，在 paintEvent 中按需缩放"""
        if hasattr(label, '_smart_painter_installed'):
            return
        filt = _SmartPaintFilter(label)
        label.installEventFilter(filt)
        label._smart_painter_installed = True

    def _set_label_image(self, label, qpixmap):
        """统一入口：给 label 设置原始 pixmap，由事件过滤器在 paintEvent 中按当前尺寸缩放。
        同时调用 setPixmap() 保持 label.pixmap() 兼容。"""
        if label is None:
            return
        self._install_smart_painter(label)
        label.setScaledContents(False)
        if qpixmap is not None and not qpixmap.isNull():
            label.setProperty('smart_original_pixmap', qpixmap.copy())
            # 调用 setPixmap(原始) — 让 label.pixmap() 能返回它；绘制逻辑由 eventFilter 覆盖
            label.setPixmap(qpixmap.copy())
        else:
            label.setProperty('smart_original_pixmap', None)
            label.setPixmap(QPixmap())
        label.setText("")
        label.setAlignment(Qt.AlignCenter)
        label.update()

    def resizeEvent(self, event):
        """窗口尺寸变化 — label 的 paintEvent 会被自动调用，不需要额外刷新"""
        super().resizeEvent(event)

    def showImageContextMenu(self, pos):
        label = self.sender()  # 获取触发右键菜单的标签
        if not isinstance(label, QLabel):
            return

        menu = QMenu(self)

        # 添加菜单项
        saveAction = menu.addAction("保存图片")
        openAction = menu.addAction("打开图片")
        cameraAction = menu.addAction("摄像头导入")

        # 根据当前标签是否有图片设置菜单项状态
        saveAction.setEnabled(label.pixmap() is not None and not label.pixmap().isNull())

        # 显示菜单并处理选择
        action = menu.exec_(label.mapToGlobal(pos))
        if action == saveAction:
            self.saveImage(label)
        elif action == openAction:
            self.openImage(label)
        elif action == cameraAction:
            self.openCameraAndCapture_label(label)

    def openImageAction(self):
        fileNames, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Image Files",
            "e:\\temp",
            "Images (*.png *.jpg *.tif *.jpeg *.bmp);;All Files (*)"
        )
        if fileNames:
            for fileName in fileNames:
                pixmap = QPixmap(fileName)
                if not pixmap.isNull():
                    # 保存原始 pixmap 引用，由 _SmartPaintFilter 在 paintEvent 中按当前尺寸缩放
                    self._set_label_image(self.label_Pretreatment_Image, pixmap)
                    self.original_pixmap = pixmap.copy()
                else:
                    QMessageBox.warning(self, "Error", f"Failed to load image: {fileName}")

    
    def saveCurrentImage(self):
        self.saveImage(self.label_Pretreatment_Image)

    def saveImage(self, label):
        pixmap = label.pixmap()
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(self, "Error", "No image to save!")
            return

        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "e:\\temp\\untitled.png",
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif);;BMP (*.bmp);;All Files (*)"
        )

        if fileName:
            if not pixmap.save(fileName):
                QMessageBox.critical(self, "Error", "Failed to save image!")
            else:
                QMessageBox.information(self, "Success", f"Image saved to:\n{fileName}")

    def openImage(self, label):
        fileNames, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Image Files",
            "e:\\temp",
            "Images (*.png *.jpg *.tif *.jpeg *.bmp);;All Files (*)"
        )
        if fileNames:
            names = "\n".join(fileNames)
            for fileName in fileNames:
                pixmap = QPixmap(fileName)
                if not pixmap.isNull():
                    self._set_label_image(label, pixmap)
                    if label == self.label_Pretreatment_Image:
                        self.original_pixmap = pixmap.copy()
                else:
                    QMessageBox.warning(self, "Error", f"Failed to load image: {fileName}")
            QMessageBox.information(self, "Files Opened", f"Successfully opened:\n{names}")

    def openCameraAndCapture(self):
        self.openCameraAndCapture_label(self.label_Pretreatment_Image)



    def openCameraAndCapture_label(self, targetLabel):
        # 记录当前显示图像的标签
        self.current_camera_label = targetLabel

        # 检查可用摄像头
        cameras = QCameraInfo.availableCameras()
        if not cameras:
            QMessageBox.warning(self, "错误", "没有检测到可用摄像头")
            return

        # 清理之前的摄像头资源
        self.cleanupCamera()

        try:
            # 创建摄像头对话框
            self.camera_dialog = CameraDialog(self)
            self.camera_dialog.setModal(True)

            # 创建并配置摄像头
            self.camera = QCamera(cameras[0])

            # 设置取景器
            self.camera.setViewfinder(self.camera_dialog.viewfinder)

            # 创建图像捕获对象
            self.imageCapture = QCameraImageCapture(self.camera)

            # 配置图像设置
            settings = QImageEncoderSettings()
            settings.setCodec("image/jpeg")
            settings.setQuality(QMultimedia.VeryHighQuality)
            self.imageCapture.setEncodingSettings(settings)

            # 设置捕获模式为静态图像
            self.camera.setCaptureMode(QCamera.CaptureStillImage)

            # 连接信号
            self.imageCapture.imageCaptured.connect(self.handleImageCaptured)
            self.imageCapture.error.connect(self.handleCaptureError)
            self.camera.errorOccurred.connect(self.handleCameraError)

            # 连接对话框按钮信号
            self.camera_dialog.captureButton.clicked.connect(self.captureImage)
            self.camera_dialog.cancelButton.clicked.connect(self.cleanupCamera)

            # 启动摄像头
            self.camera.start()

            # 显示对话框
            self.camera_dialog.exec_()

        except Exception as e:
            QMessageBox.critical(self, "摄像头错误", f"初始化摄像头失败: {str(e)}")
            self.cleanupCamera()

    def captureImage(self):
        """执行拍照操作"""
        if self.imageCapture and self.imageCapture.isReadyForCapture():
            # 不指定文件路径，直接捕获到内存
            self.imageCapture.capture()
        else:
            QMessageBox.warning(self, "警告", "摄像头尚未准备好拍照")

    def handleImageCaptured(self, id, image):
        """处理捕获的图像"""
        try:
            if self.current_camera_label and not image.isNull():
                pixmap = QPixmap.fromImage(image)
                if not pixmap.isNull():
                    self._set_label_image(self.current_camera_label, pixmap)
                    if self.current_camera_label == self.label_Pretreatment_Image:
                        self.original_pixmap = pixmap.copy()
                    if self.camera_dialog:
                        self.camera_dialog.accept()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理图像时出错: {str(e)}")
        finally:
            self.cleanupCamera()


    def handleCameraError(self, error):
        QMessageBox.critical(self, "摄像头错误", self.camera.errorString())
        self.cleanupCamera()

    def handleCaptureError(self, id, error, errorString):
        """处理捕获错误"""
        # 忽略"could not save image to file"错误，因为我们不需要保存文件
        if "could not save image to file" not in errorString.lower():
            QMessageBox.critical(self, "捕获错误", errorString)
        self.cleanupCamera()

    def cleanupCamera(self):
        """清理摄像头资源"""
        if self.camera:
            self.camera.stop()
            self.camera.unload()
        if self.camera_dialog:
            self.camera_dialog.close()
            self.camera_dialog.deleteLater()
        self.camera = None
        self.imageCapture = None
        self.camera_dialog = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())