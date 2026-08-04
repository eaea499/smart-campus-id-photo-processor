"""
智慧快递包裹条码增强与识别预处理系统 - GUI主界面
基于PyQt5开发
"""

import sys
import os
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from BarcodeProcessor import BarcodeProcessor
from IDPhotoProcessor import IDPhotoUtils


class ImageLabel(QLabel):
    imageClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setFrameShape(QFrame.Box)
        self.setStyleSheet("background-color: #E8E8E8; border: 2px solid #CCCCCC;")
        self.setMinimumSize(320, 420)
        self.setText("暂无图像\n\n点击导入或拖拽图像到此处")
        self.setWordWrap(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.imageClicked.emit()


class BarcodeMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.processor = BarcodeProcessor()
        self.image_paths = []
        self.current_index = -1
        self.original_image = None
        self.processed_image = None

        self.current_params = {
            "denoise_method": "gaussian",
            "denoise_ksize": 3,
            "sharpen_amount": 1.0,
            "threshold_value": 0,
            "threshold_invert": True,
            "morph_ksize": 5,
            "output_mode": "binary",
        }

        self.init_ui()
        self.setup_styles()

    def init_ui(self):
        self.setWindowTitle("智慧快递包裹条码增强与识别预处理系统")
        self.setMinimumSize(1400, 900)
        self.resize(1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.create_navigation_panel()
        main_layout.addWidget(self.nav_panel, 0)

        self.create_display_area()
        main_layout.addWidget(self.display_area, 1)

        self.create_control_panel()
        main_layout.addWidget(self.control_panel, 0)

        self.create_status_bar()
        self.create_menu_bar()

    def create_navigation_panel(self):
        self.nav_panel = QFrame()
        self.nav_panel.setFixedWidth(170)
        self.nav_panel.setStyleSheet("background-color: #2C3E50;")

        nav_layout = QVBoxLayout(self.nav_panel)
        nav_layout.setSpacing(5)
        nav_layout.setContentsMargins(10, 20, 10, 20)

        nav_buttons = [
            ("🏠 首页", self.show_home_panel),
            ("📁 打开图片", self.open_images),
            ("🧩 参数调节", self.show_param_panel),
            ("⚙️ 一键预处理", self.one_click_process),
            ("💾 保存结果", self.save_current),
            ("📦 批量保存", self.save_batch),
            ("❌ 退出", self.close),
        ]

        for text, callback in nav_buttons:
            btn = QPushButton(text)
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: transparent;
                    color: white;
                    border: none;
                    padding: 15px 10px;
                    text-align: left;
                    font-size: 13px;
                }
                QPushButton:hover { background-color: #34495E; }
                QPushButton:pressed { background-color: #3498DB; }
                """
            )
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(callback)
            nav_layout.addWidget(btn)

        nav_layout.addStretch()

    def create_display_area(self):
        self.display_area = QFrame()
        self.display_area.setStyleSheet("background-color: #ECF0F1;")

        display_layout = QVBoxLayout(self.display_area)
        display_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("智慧快递包裹条码增强与识别预处理系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(
            """
            font-size: 20px;
            font-weight: bold;
            color: #2C3E50;
            padding: 10px;
            """
        )
        display_layout.addWidget(title_label)

        images_layout = QHBoxLayout()

        original_group = QGroupBox("原始图像")
        original_layout = QVBoxLayout(original_group)
        self.original_label = ImageLabel()
        self.original_label.imageClicked.connect(self.open_images)
        original_layout.addWidget(self.original_label)
        images_layout.addWidget(original_group)

        arrow_label = QLabel("➜")
        arrow_label.setAlignment(Qt.AlignCenter)
        arrow_label.setStyleSheet("font-size: 30px; color: #3498DB;")
        images_layout.addWidget(arrow_label)

        processed_group = QGroupBox("处理后图像")
        processed_layout = QVBoxLayout(processed_group)
        self.processed_label = ImageLabel()
        self.processed_label.imageClicked.connect(self.save_current)
        processed_layout.addWidget(self.processed_label)
        images_layout.addWidget(processed_group)

        display_layout.addLayout(images_layout)

        self.info_label = QLabel("未加载图像")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #7F8C8D; padding: 10px;")
        display_layout.addWidget(self.info_label)

    def create_control_panel(self):
        self.control_panel = QFrame()
        self.control_panel.setFixedWidth(330)
        self.control_panel.setStyleSheet("background-color: #FFFFFF; border-left: 1px solid #BDC3C7;")

        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_layout.setContentsMargins(15, 20, 15, 20)
        self.control_layout.setSpacing(15)

        self.show_home_panel()

    def create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        self.size_label = QLabel("图像尺寸: -")
        self.params_label = QLabel("阈值: 自动 | 锐化: 1.0")
        self.status_bar.addPermanentWidget(self.size_label)
        self.status_bar.addPermanentWidget(self.params_label)

    def create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件(F)")
        open_action = QAction("打开图片...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_images)
        file_menu.addAction(open_action)

        save_action = QAction("保存当前...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_current)
        file_menu.addAction(save_action)

        file_menu.addSeparator()
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        process_menu = menubar.addMenu("处理(P)")
        run_action = QAction("一键预处理", self)
        run_action.setShortcut("F5")
        run_action.triggered.connect(self.one_click_process)
        process_menu.addAction(run_action)

        help_menu = menubar.addMenu("帮助(H)")
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_styles(self):
        self.setStyleSheet(
            """
            QMainWindow { background-color: #ECF0F1; }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #BDC3C7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #2C3E50;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2980B9; }
            QPushButton:pressed { background-color: #21618C; }
            QPushButton:disabled { background-color: #BDC3C7; }
            QSlider::groove:horizontal {
                height: 8px;
                background: #BDC3C7;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                width: 18px;
                background: #3498DB;
                border-radius: 9px;
                margin: -5px 0;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #BDC3C7;
                border-radius: 4px;
                background: white;
            }
            QLabel { color: #2C3E50; }
            """
        )

    def clear_control_panel(self):
        while self.control_layout.count():
            item = self.control_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def show_home_panel(self):
        self.clear_control_panel()

        title = QLabel("⚡ 快捷操作")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2C3E50;")
        self.control_layout.addWidget(title)

        btn_open = QPushButton("📁 打开图片（可多选）")
        btn_open.setMinimumHeight(45)
        btn_open.clicked.connect(self.open_images)
        self.control_layout.addWidget(btn_open)

        btn_run = QPushButton("⚙️ 一键预处理")
        btn_run.setMinimumHeight(45)
        btn_run.clicked.connect(self.one_click_process)
        self.control_layout.addWidget(btn_run)

        self.control_layout.addSpacing(15)

        list_group = QGroupBox("📋 多图对比（选择切换）")
        list_layout = QVBoxLayout(list_group)
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_list_row_changed)
        list_layout.addWidget(self.list_widget)
        self.control_layout.addWidget(list_group)

        tips = QLabel("提示：阈值=0 为自动（Otsu），F5 可快速预处理。")
        tips.setWordWrap(True)
        tips.setStyleSheet("color: #7F8C8D;")
        self.control_layout.addWidget(tips)

        self.control_layout.addStretch()

    def show_param_panel(self):
        self.clear_control_panel()

        title = QLabel("🧩 参数调节")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2C3E50;")
        self.control_layout.addWidget(title)

        denoise_group = QGroupBox("滤波去噪")
        denoise_layout = QVBoxLayout(denoise_group)

        self.denoise_combo = QComboBox()
        self.denoise_combo.addItems(["gaussian", "median", "bilateral", "none"])
        self.denoise_combo.setCurrentText(self.current_params["denoise_method"])
        self.denoise_combo.currentTextChanged.connect(self.on_param_changed)
        denoise_layout.addWidget(QLabel("去噪方法"))
        denoise_layout.addWidget(self.denoise_combo)

        self.denoise_slider = QSlider(Qt.Horizontal)
        self.denoise_slider.setRange(1, 11)
        self.denoise_slider.setValue(int(self.current_params["denoise_ksize"]))
        self.denoise_slider.valueChanged.connect(self.on_param_changed)
        denoise_layout.addWidget(QLabel("滤波核大小（建议奇数）"))
        denoise_layout.addWidget(self.denoise_slider)

        self.control_layout.addWidget(denoise_group)

        sharp_group = QGroupBox("图像锐化")
        sharp_layout = QVBoxLayout(sharp_group)
        self.sharp_slider = QSlider(Qt.Horizontal)
        self.sharp_slider.setRange(0, 300)
        self.sharp_slider.setValue(int(float(self.current_params["sharpen_amount"]) * 100))
        self.sharp_slider.valueChanged.connect(self.on_param_changed)
        sharp_layout.addWidget(QLabel("锐化强度"))
        sharp_layout.addWidget(self.sharp_slider)
        self.control_layout.addWidget(sharp_group)

        thresh_group = QGroupBox("二值化阈值")
        thresh_layout = QVBoxLayout(thresh_group)
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(0, 255)
        self.thresh_slider.setValue(int(self.current_params["threshold_value"]))
        self.thresh_slider.valueChanged.connect(self.on_param_changed)
        thresh_layout.addWidget(QLabel("阈值（0=自动）"))
        thresh_layout.addWidget(self.thresh_slider)

        self.invert_checkbox = QCheckBox("反色（黑条码/白底常用）")
        self.invert_checkbox.setChecked(bool(self.current_params["threshold_invert"]))
        self.invert_checkbox.stateChanged.connect(self.on_param_changed)
        thresh_layout.addWidget(self.invert_checkbox)
        self.control_layout.addWidget(thresh_group)

        morph_group = QGroupBox("形态学开运算")
        morph_layout = QVBoxLayout(morph_group)
        self.morph_slider = QSlider(Qt.Horizontal)
        self.morph_slider.setRange(1, 25)
        self.morph_slider.setValue(int(self.current_params["morph_ksize"]))
        self.morph_slider.valueChanged.connect(self.on_param_changed)
        morph_layout.addWidget(QLabel("开运算核大小"))
        morph_layout.addWidget(self.morph_slider)
        self.control_layout.addWidget(morph_group)

        out_group = QGroupBox("输出显示")
        out_layout = QVBoxLayout(out_group)
        self.output_combo = QComboBox()
        self.output_combo.addItems(["binary", "roi"])
        self.output_combo.setCurrentText(self.current_params["output_mode"])
        self.output_combo.currentTextChanged.connect(self.on_param_changed)
        out_layout.addWidget(QLabel("处理图显示模式"))
        out_layout.addWidget(self.output_combo)
        self.control_layout.addWidget(out_group)

        run_btn = QPushButton("应用参数并预处理")
        run_btn.setMinimumHeight(42)
        run_btn.clicked.connect(self.one_click_process)
        self.control_layout.addWidget(run_btn)

        self.auto_checkbox = QCheckBox("参数变化自动更新")
        self.auto_checkbox.setChecked(True)
        self.control_layout.addWidget(self.auto_checkbox)

        self.control_layout.addStretch()

    def on_param_changed(self, *args):
        if hasattr(self, "denoise_combo"):
            self.current_params["denoise_method"] = self.denoise_combo.currentText()
        if hasattr(self, "denoise_slider"):
            self.current_params["denoise_ksize"] = int(self.denoise_slider.value())
        if hasattr(self, "sharp_slider"):
            self.current_params["sharpen_amount"] = float(self.sharp_slider.value()) / 100.0
        if hasattr(self, "thresh_slider"):
            self.current_params["threshold_value"] = int(self.thresh_slider.value())
        if hasattr(self, "invert_checkbox"):
            self.current_params["threshold_invert"] = bool(self.invert_checkbox.isChecked())
        if hasattr(self, "morph_slider"):
            self.current_params["morph_ksize"] = int(self.morph_slider.value())
        if hasattr(self, "output_combo"):
            self.current_params["output_mode"] = self.output_combo.currentText()

        if getattr(self, "auto_checkbox", None) is not None and self.auto_checkbox.isChecked():
            self.one_click_process()

    def open_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择快递包裹图像",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not file_paths:
            return

        if not hasattr(self, "list_widget") or self.list_widget is None or self.list_widget.parent() is None:
            self.show_home_panel()

        self.image_paths = file_paths
        self.list_widget.clear()
        for p in self.image_paths:
            self.list_widget.addItem(os.path.basename(p))

        self.list_widget.setCurrentRow(0)
        self.status_bar.showMessage(f"已加载 {len(self.image_paths)} 张图像")

    def on_list_row_changed(self, row):
        if row < 0 or row >= len(self.image_paths):
            return
        self.current_index = row
        self.load_current_image()
        self.one_click_process()

    def load_current_image(self):
        if self.current_index < 0 or self.current_index >= len(self.image_paths):
            return
        path = self.image_paths[self.current_index]
        image, err = IDPhotoUtils.load_image(path)
        if err:
            QMessageBox.warning(self, "加载失败", err)
            return

        self.original_image = image
        self.update_image_display(self.original_label, self.original_image)

        info = IDPhotoUtils.get_image_info(self.original_image)
        if info:
            self.size_label.setText(f"图像尺寸: {info['size']}")
        self.info_label.setText(f"当前文件：{os.path.basename(path)}")

    def one_click_process(self):
        if self.original_image is None:
            return

        result = self.processor.process_single(self.original_image, self.current_params)
        if not result.get("success"):
            QMessageBox.warning(self, "处理失败", result.get("error", "未知错误"))
            return

        self.processed_image = result["image"]
        self.update_image_display(self.processed_label, self.processed_image)

        tv = self.current_params["threshold_value"]
        tv_text = "自动" if tv <= 0 else str(tv)
        self.params_label.setText(f"阈值: {tv_text} | 锐化: {self.current_params['sharpen_amount']:.2f}")
        self.status_bar.showMessage("预处理完成")

    def save_current(self):
        if self.processed_image is None:
            return

        default_name = "barcode_processed.png"
        if self.current_index >= 0 and self.current_index < len(self.image_paths):
            base = os.path.splitext(os.path.basename(self.image_paths[self.current_index]))[0]
            default_name = f"{base}_processed.png"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存处理结果",
            default_name,
            "PNG Files (*.png);;JPG Files (*.jpg *.jpeg);;BMP Files (*.bmp)",
        )
        if not save_path:
            return

        ok, err = IDPhotoUtils.save_image(self.processed_image, save_path)
        if not ok:
            QMessageBox.warning(self, "保存失败", err or "未知错误")
            return
        self.status_bar.showMessage(f"已保存：{save_path}")

    def save_batch(self):
        if not self.image_paths:
            return

        out_dir = QFileDialog.getExistingDirectory(self, "选择批量保存目录")
        if not out_dir:
            return

        saved = 0
        for i, path in enumerate(self.image_paths):
            image, err = IDPhotoUtils.load_image(path)
            if err or image is None:
                continue
            result = self.processor.process_single(image, self.current_params)
            if not result.get("success"):
                continue
            base = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(out_dir, f"{base}_processed.png")
            ok, _ = IDPhotoUtils.save_image(result["image"], out_path)
            if ok:
                saved += 1

        self.status_bar.showMessage(f"批量保存完成：{saved}/{len(self.image_paths)}")

    def update_image_display(self, label, image):
        if image is None:
            label.setText("暂无图像")
            return

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_image is not None:
            self.update_image_display(self.original_label, self.original_image)
        if self.processed_image is not None:
            self.update_image_display(self.processed_label, self.processed_image)

    def show_about(self):
        QMessageBox.information(
            self,
            "关于",
            "智慧快递包裹条码增强与识别预处理系统\n\n"
            "功能：去噪、锐化、二值化、形态学优化与条码区域提取。\n"
            "快捷键：Ctrl+O 打开，F5 一键预处理，Ctrl+S 保存。",
        )


def main():
    app = QApplication(sys.argv)
    window = BarcodeMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
