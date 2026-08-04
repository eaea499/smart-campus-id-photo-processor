# IDPhotoGUI.py
"""
证件照标准化预处理系统 - GUI主界面
左侧导航 + 中间图像 + 右侧操作区布局
"""

import os as _os, sys as _sys, ctypes as _ctypes
_P = _os.path.join(_sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
if _os.path.isdir(_P):
    _K = _ctypes.windll.kernel32
    try: _K.SetDefaultDllDirectories(0x00001000)
    except: pass
    _K.SetDllDirectoryW(_P)
    try: _os.add_dll_directory(_P)
    except: pass
    for _f in sorted(_os.listdir(_P)):
        if _f.endswith('.dll'):
            try: _K.LoadLibraryExW(_os.path.join(_P, _f), None, 0x00000008)
            except: pass

import sys, os, cv2, numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from IDPhotoProcessor import IDPhotoProcessor, IDPhotoUtils


class ImageLabel(QLabel):
    """图像显示标签 — 保存原始 pixmap，在 paintEvent 中自动缩放，避免切换页面时因外部多次 setPixmap 产生抖动"""
    imageClicked = pyqtSignal()

    def __init__(self, parent=None, allow_open=True):
        super().__init__(parent)
        self.allow_open = allow_open
        self.setAlignment(Qt.AlignCenter)
        self.setFrameShape(QFrame.Box)
        self.setStyleSheet("background-color: #E8E8E8; border: 2px solid #CCCCCC;")
        self.setMinimumSize(260, 350)
        self._empty_text = "暂无图像\n\n点击导入或拖拽图像到此处"
        self.setText(self._empty_text)
        self.setWordWrap(True)
        # 保存原始未缩放的 pixmap（None 表示无图）
        self._original_pixmap = None
        # 标记是否为已处理图像（用于右键保存判断）
        self._is_processed = False

    def set_image(self, qpixmap, is_processed=False):
        """设置原始 pixmap，不触发多次缩放；真正的缩放在 paintEvent 中完成"""
        self._original_pixmap = qpixmap
        self._is_processed = is_processed
        # 仅触发一次 repaint，所有绘制在同一 paint 帧完成
        self.setText("")
        self.update()

    def clear_image(self):
        """清空图像显示"""
        self._original_pixmap = None
        self._is_processed = False
        self.setText(self._empty_text)
        self.update()

    def has_image(self):
        return self._original_pixmap is not None and not self._original_pixmap.isNull()

    def paintEvent(self, event):
        # 无图：显示默认文字（交给 QLabel.paintEvent）
        if self._original_pixmap is None or self._original_pixmap.isNull():
            super().paintEvent(event)
            return
        # 有图：在 paintEvent 中直接按当前尺寸等比缩放绘制
        # 关键：不再通过 setPixmap→repaint→setPixmap 这种多帧路径
        from PyQt5.QtGui import QPainter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        # 计算缩放后的目标矩形（保持宽高比，居中）
        label_w = max(1, self.width() - 8)
        label_h = max(1, self.height() - 8)
        pw = self._original_pixmap.width()
        ph = self._original_pixmap.height()
        if pw <= 0 or ph <= 0:
            super().paintEvent(event)
            return
        scale = min(label_w / pw, label_h / ph)
        dw = int(pw * scale)
        dh = int(ph * scale)
        dx = (self.width() - dw) // 2
        dy = (self.height() - dh) // 2
        painter.drawPixmap(dx, dy, dw, dh, self._original_pixmap)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.imageClicked.emit()


class NavButton(QPushButton):
    """左侧导航按钮"""
    def __init__(self, icon_text, label_text, parent=None):
        super().__init__(parent)
        self.setFixedSize(70, 70)
        self.setText(f"{icon_text}\n{label_text}")
        self.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #BDC3C7;
                border: none;
                border-radius: 8px;
                font-size: 11px;
                font-weight: bold;
                padding: 4px;
            }
            QPushButton:hover {
                background-color: #34495E;
                color: #ECF0F1;
            }
        """)
        self._active = False

    def set_active(self, active):
        self._active = active
        if active:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #3498DB;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 11px;
                    font-weight: bold;
                    padding: 4px;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #BDC3C7;
                    border: none;
                    border-radius: 8px;
                    font-size: 11px;
                    font-weight: bold;
                    padding: 4px;
                }
                QPushButton:hover {
                    background-color: #34495E;
                    color: #ECF0F1;
                }
            """)


class IDPhotoMainWindow(QMainWindow):
    """证件照处理主窗口 - 左导航 + 中图像 + 右操作区"""

    def __init__(self):
        super().__init__()
        self.processor = IDPhotoProcessor()
        self.cropper = self.processor.cropper
        self.bg_remover = self.processor.bg_remover
        self.enhancer = self.processor.enhancer
        self.current_image = None
        self.original_image = None
        self.processed_image = None
        self.current_file_path = None
        self._worker_thread = None
        self._progress_dialog = None

        self.bg_color = 'blue'
        self.nav_buttons = []
        self.pages = {}

        self.init_ui()
        self.setup_styles()

    def init_ui(self):
        self.setWindowTitle("证件照标准化预处理系统")
        self.setMinimumSize(950, 680)
        self.resize(1050, 720)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 左侧导航栏
        nav_bar = self._build_nav_bar()
        main_layout.addWidget(nav_bar)

        # 右侧内容区
        self.content_stack = QStackedWidget()
        main_layout.addWidget(self.content_stack, 1)

        # 构建各页面
        self._build_oneclick_page()
        self._build_resize_page()
        self._build_background_page()
        self._build_brightness_page()
        self._build_denoise_page()

        # 默认显示一键处理页面
        self.switch_page(0)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 - 请导入图片后开始处理")
        self.time_label = QLabel("")
        self.status_bar.addPermanentWidget(self.time_label)

    def _build_nav_bar(self):
        """构建左侧导航栏"""
        nav_frame = QFrame()
        nav_frame.setFixedWidth(90)
        nav_frame.setStyleSheet("background-color: #2C3E50;")
        nav_layout = QVBoxLayout(nav_frame)
        nav_layout.setContentsMargins(8, 15, 8, 15)
        nav_layout.setSpacing(6)

        # 标题
        title = QLabel("功能导航")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ECF0F1; font-size: 12px; font-weight: bold; margin-bottom: 8px;")
        nav_layout.addWidget(title)

        # 导航按钮
        nav_items = [
            ("⚡", "一键处理"),
            ("📐", "尺寸标准化"),
            ("🎨", "背景替换"),
            ("🔆", "亮度对比度"),
            ("✨", "人脸去噪"),
        ]

        for icon, label in nav_items:
            btn = NavButton(icon, label)
            idx = len(self.nav_buttons)
            btn.clicked.connect(lambda _, i=idx: self.switch_page(i))
            nav_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        nav_layout.addStretch()

        # 底部撤销按钮
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #34495E;")
        nav_layout.addWidget(sep)

        undo_btn = NavButton("↩", "撤销")
        undo_btn.setFixedSize(70, 50)
        undo_btn.clicked.connect(self.step_undo)
        nav_layout.addWidget(undo_btn)

        return nav_frame

    def switch_page(self, index):
        """切换右侧内容页面 — 不触发 processEvents，避免中间帧抖动"""
        # 只更新按钮高亮状态 + 切换页面
        for i, btn in enumerate(self.nav_buttons):
            btn.set_active(i == index)
        self.content_stack.setCurrentIndex(index)
        # 注意：不再强行刷新 — ImageLabel 自身在 paintEvent 中按当前尺寸缩放原始 pixmap，
        # 页面切换时布局系统会自动触发 repaint，ImageLabel 自己绘制即可，
        # 这样就不会出现"先清空再重绘"的中间帧

    # ==================== 页面构建 ====================

    def _build_page_layout(self, title_text, desc_text):
        """构建通用页面布局：中间图像区 + 右侧操作区"""
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # 中间：图像显示区
        mid_frame = QFrame()
        mid_frame.setStyleSheet("background-color: #ECF0F1; border-radius: 6px;")
        mid_layout = QVBoxLayout(mid_frame)
        mid_layout.setContentsMargins(15, 10, 15, 10)

        title = QLabel(title_text)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2C3E50;")
        mid_layout.addWidget(title)

        if desc_text:
            desc = QLabel(desc_text)
            desc.setAlignment(Qt.AlignCenter)
            desc.setStyleSheet("color: #7F8C8D; font-size: 12px;")
            mid_layout.addWidget(desc)

        cols = QHBoxLayout()
        cols.setSpacing(10)

        g1 = QGroupBox("原始图像")
        l1 = QVBoxLayout(g1)
        orig_label = ImageLabel(allow_open=True)
        orig_label.imageClicked.connect(self.open_image)
        l1.addWidget(orig_label)
        cols.addWidget(g1)

        arrow = QLabel("➜")
        arrow.setAlignment(Qt.AlignCenter)
        arrow.setStyleSheet("font-size: 24px; color: #3498DB;")
        cols.addWidget(arrow)

        g2 = QGroupBox("处理后图像")
        l2 = QVBoxLayout(g2)
        proc_label = ImageLabel(allow_open=False)
        proc_label.imageClicked.connect(self.save_image)
        proc_label.setText("暂无结果\n\n处理完成后可点击保存")
        l2.addWidget(proc_label)
        cols.addWidget(g2)

        mid_layout.addLayout(cols)

        info_label = QLabel("未加载图像")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #7F8C8D;")
        mid_layout.addWidget(info_label)

        layout.addWidget(mid_frame, 1)

        # 保存引用
        page._orig_label = orig_label
        page._proc_label = proc_label
        page._info_label = info_label
        page._mid_layout = mid_layout

        return page, layout

    def _build_oneclick_page(self):
        """一键处理页面（初始界面）"""
        page, layout = self._build_page_layout(
            "证件照标准化预处理系统",
            "导入图片，选择背景颜色，一键完成证件照处理"
        )

        # 右侧操作区
        right_panel = QFrame()
        right_panel.setFixedWidth(220)
        right_panel.setStyleSheet("background-color: #FFFFFF; border: 1px solid #CCC; border-radius: 6px;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 20, 15, 20)
        right_layout.setSpacing(15)

        # 背景颜色选择
        color_lbl = QLabel("选择背景颜色")
        color_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #2C3E50;")
        right_layout.addWidget(color_lbl)

        color_col = QVBoxLayout()
        colors = [
            ("蓝底", "blue", "#438EDB"),
            ("白底", "white", "#FFFFFF"),
            ("红底", "red", "#CE0000"),
        ]
        self.oneclick_color_btns = []
        for name, val, hex_c in colors:
            b = QPushButton(name)
            b.setCheckable(True)
            b.setMinimumHeight(40)
            txt_c = 'black' if val == 'white' else 'white'
            b.setStyleSheet(f"""
                QPushButton {{
                    background-color: {hex_c};
                    color: {txt_c};
                    border: 2px solid #AAA;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 14px;
                }}
                QPushButton:checked {{
                    border: 3px solid #27AE60;
                }}
            """)
            b.clicked.connect(lambda _, v=val: self._pick_bg_color(v))
            if val == self.bg_color:
                b.setChecked(True)
            color_col.addWidget(b)
            self.oneclick_color_btns.append(b)
        right_layout.addLayout(color_col)

        right_layout.addStretch()

        # 一键处理按钮
        btn_process = QPushButton("⚡ 一键处理")
        btn_process.setMinimumHeight(52)
        btn_process.setStyleSheet("""
            QPushButton {
                background-color: #27AE60; color: white; border: none;
                border-radius: 8px; font-size: 16px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2ECC71; }
            QPushButton:pressed { background-color: #219B50; }
        """)
        btn_process.clicked.connect(self.one_click_process)
        right_layout.addWidget(btn_process)

        layout.addWidget(right_panel)

        self.pages['oneclick'] = page
        self.content_stack.addWidget(page)

    def _build_resize_page(self):
        """尺寸标准化页面"""
        page, layout = self._build_page_layout(
            "图像尺寸标准化",
            "将证件照裁剪并缩放为标准尺寸，支持人脸定位自动裁剪"
        )

        # 右侧操作区
        right_panel = QFrame()
        right_panel.setFixedWidth(220)
        right_panel.setStyleSheet("background-color: #FFFFFF; border: 1px solid #CCC; border-radius: 6px;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 20, 15, 20)
        right_layout.setSpacing(15)

        size_lbl = QLabel("目标尺寸")
        size_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #2C3E50;")
        right_layout.addWidget(size_lbl)

        self.size_combo = QComboBox()
        self.size_combo.addItems([
            "小一寸 (260×378)",
            "标准一寸 (295×413)",
            "大一寸 (390×567)",
            "小二寸 (413×531)",
            "标准二寸 (413×579)",
            "大二寸 (413×626)",
            "身份证 (308×384)",
        ])
        self.size_combo.setMinimumHeight(36)
        right_layout.addWidget(self.size_combo)

        right_layout.addStretch()

        btn_resize = QPushButton("📐 执行尺寸标准化")
        btn_resize.setMinimumHeight(52)
        btn_resize.setStyleSheet("""
            QPushButton { background-color: #27AE60; color: white; border: none;
                          border-radius: 8px; font-size: 14px; font-weight: bold; }
            QPushButton:hover { background-color: #2ECC71; }
        """)
        btn_resize.clicked.connect(self.step_resize)
        right_layout.addWidget(btn_resize)

        layout.addWidget(right_panel)

        self.pages['resize'] = page
        self.content_stack.addWidget(page)

    def _build_background_page(self):
        """背景替换页面"""
        page, layout = self._build_page_layout(
            "背景色替换",
            "使用深度学习或传统算法替换证件照背景色"
        )

        # 右侧操作区
        right_panel = QFrame()
        right_panel.setFixedWidth(220)
        right_panel.setStyleSheet("background-color: #FFFFFF; border: 1px solid #CCC; border-radius: 6px;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 20, 15, 20)
        right_layout.setSpacing(15)

        # 背景颜色
        color_lbl = QLabel("目标背景色")
        color_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #2C3E50;")
        right_layout.addWidget(color_lbl)

        color_col = QVBoxLayout()
        colors = [("蓝底", "blue", "#438EDB"), ("白底", "white", "#FFFFFF"), ("红底", "red", "#CE0000")]
        self.bg_color_btns = []
        for name, val, hex_c in colors:
            b = QPushButton(name)
            b.setCheckable(True)
            b.setMinimumHeight(38)
            txt_c = 'black' if val == 'white' else 'white'
            b.setStyleSheet(f"""
                QPushButton {{ background-color: {hex_c}; color: {txt_c};
                    border: 2px solid #AAA; border-radius: 4px; font-weight: bold; }}
                QPushButton:checked {{ border: 3px solid #27AE60; }}
            """)
            b.clicked.connect(lambda _, v=val: self._pick_bg_color(v))
            if val == self.bg_color:
                b.setChecked(True)
            color_col.addWidget(b)
            self.bg_color_btns.append(b)
        right_layout.addLayout(color_col)

        # 分割方法
        method_lbl = QLabel("分割方法")
        method_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #2C3E50;")
        right_layout.addWidget(method_lbl)

        self.bg_method_combo = QComboBox()
        self.bg_method_combo.addItems(["PP-HumanSeg (推荐)", "GrabCut", "颜色分割"])
        self.bg_method_combo.setMinimumHeight(36)
        right_layout.addWidget(self.bg_method_combo)

        right_layout.addStretch()

        btn_bg = QPushButton("🎨 执行背景替换")
        btn_bg.setMinimumHeight(52)
        btn_bg.setStyleSheet("""
            QPushButton { background-color: #27AE60; color: white; border: none;
                          border-radius: 8px; font-size: 14px; font-weight: bold; }
            QPushButton:hover { background-color: #2ECC71; }
        """)
        btn_bg.clicked.connect(self.step_background)
        right_layout.addWidget(btn_bg)

        layout.addWidget(right_panel)

        self.pages['background'] = page
        self.content_stack.addWidget(page)

    def _build_brightness_page(self):
        """亮度/对比度调节页面"""
        page, layout = self._build_page_layout(
            "亮度 / 对比度调节",
            "手动调节证件照的亮度和对比度参数"
        )

        # 右侧操作区
        right_panel = QFrame()
        right_panel.setFixedWidth(220)
        right_panel.setStyleSheet("background-color: #FFFFFF; border: 1px solid #CCC; border-radius: 6px;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 20, 15, 20)
        right_layout.setSpacing(15)

        # 亮度
        brightness_lbl = QLabel("亮度")
        brightness_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #2C3E50;")
        right_layout.addWidget(brightness_lbl)

        brightness_val_layout = QHBoxLayout()
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-50, 50)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setTickPosition(QSlider.TicksBelow)
        self.brightness_slider.setTickInterval(10)
        self.brightness_slider.valueChanged.connect(self._on_brightness_changed)
        brightness_val_layout.addWidget(self.brightness_slider)
        self.brightness_label = QLabel("0")
        self.brightness_label.setFixedWidth(30)
        self.brightness_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        brightness_val_layout.addWidget(self.brightness_label)
        right_layout.addLayout(brightness_val_layout)

        # 对比度
        contrast_lbl = QLabel("对比度")
        contrast_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #2C3E50;")
        right_layout.addWidget(contrast_lbl)

        contrast_val_layout = QHBoxLayout()
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-50, 50)
        self.contrast_slider.setValue(0)
        self.contrast_slider.setTickPosition(QSlider.TicksBelow)
        self.contrast_slider.setTickInterval(10)
        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        contrast_val_layout.addWidget(self.contrast_slider)
        self.contrast_label = QLabel("0")
        self.contrast_label.setFixedWidth(30)
        self.contrast_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        contrast_val_layout.addWidget(self.contrast_label)
        right_layout.addLayout(contrast_val_layout)

        right_layout.addStretch()

        btn_bc = QPushButton("🔆 应用")
        btn_bc.setMinimumHeight(52)
        btn_bc.setStyleSheet("""
            QPushButton { background-color: #27AE60; color: white; border: none;
                          border-radius: 8px; font-size: 14px; font-weight: bold; }
            QPushButton:hover { background-color: #2ECC71; }
        """)
        btn_bc.clicked.connect(self.step_brightness_contrast)
        right_layout.addWidget(btn_bc)

        layout.addWidget(right_panel)

        self.pages['brightness'] = page
        self.content_stack.addWidget(page)

    def _build_denoise_page(self):
        """人脸去噪页面"""
        page, layout = self._build_page_layout(
            "人脸去噪",
            "对证件照进行去噪处理，支持多种滤波方法"
        )

        # 右侧操作区
        right_panel = QFrame()
        right_panel.setFixedWidth(220)
        right_panel.setStyleSheet("background-color: #FFFFFF; border: 1px solid #CCC; border-radius: 6px;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 20, 15, 20)
        right_layout.setSpacing(15)

        method_lbl = QLabel("去噪方法")
        method_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #2C3E50;")
        right_layout.addWidget(method_lbl)

        self.denoise_combo = QComboBox()
        self.denoise_combo.addItems(["双边滤波 (推荐)", "中值滤波", "高斯滤波", "非局部均值"])
        self.denoise_combo.setMinimumHeight(36)
        right_layout.addWidget(self.denoise_combo)

        right_layout.addStretch()

        btn_denoise = QPushButton("✨ 执行去噪")
        btn_denoise.setMinimumHeight(52)
        btn_denoise.setStyleSheet("""
            QPushButton { background-color: #27AE60; color: white; border: none;
                          border-radius: 8px; font-size: 14px; font-weight: bold; }
            QPushButton:hover { background-color: #2ECC71; }
        """)
        btn_denoise.clicked.connect(self.step_denoise)
        right_layout.addWidget(btn_denoise)

        layout.addWidget(right_panel)

        self.pages['denoise'] = page
        self.content_stack.addWidget(page)

    def _refresh_all_displays(self):
        """刷新所有页面的图像显示 — 不再使用 processEvents，减少中间帧抖动"""
        # 用 setUpdatesEnabled(False) 包住所有 label 的更新，让它们在同一 paint 帧内完成
        try:
            for i in range(self.content_stack.count()):
                page = self.content_stack.widget(i)
                if page is None or not hasattr(page, '_orig_label'):
                    continue
                page.setUpdatesEnabled(False)
                try:
                    # 原始图像
                    if self.original_image is not None:
                        self.display_image(page._orig_label, self.original_image, is_processed=False)
                    else:
                        if hasattr(page._orig_label, 'clear_image'):
                            page._orig_label.clear_image()

                    # 处理后图像
                    if self.processed_image is not None:
                        self.display_image(page._proc_label, self.processed_image, is_processed=True)
                    else:
                        if hasattr(page._proc_label, 'clear_image'):
                            page._proc_label.clear_image()

                    # 尺寸信息
                    if self.current_image is not None:
                        info = IDPhotoUtils.get_image_info(self.current_image)
                        page._info_label.setText(f"尺寸: {info['size']}")
                    else:
                        page._info_label.setText("未加载图像")
                finally:
                    page.setUpdatesEnabled(True)
        except Exception:
            pass

    # ==================== 公共方法 ====================

    def _pick_bg_color(self, value):
        self.bg_color = value

    def setup_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #F5F5F5; }
            QGroupBox { font-weight: bold; border: 2px solid #CCC; border-radius: 4px;
                        margin-top: 8px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #2C3E50; }
            QPushButton { background-color: #3498DB; color: white; border: none;
                          padding: 10px 20px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #2980B9; }
            QPushButton:pressed { background-color: #21618C; }
            QPushButton:disabled { background-color: #BDC3C7; }
            QLabel { color: #2C3E50; }
            QComboBox { padding: 6px 10px; border: 1px solid #CCC; border-radius: 4px; }
        """)

    def _set_busy(self, busy, message=None):
        if message: self.status_bar.showMessage(message)
        if busy:
            if self._progress_dialog is None:
                self._progress_dialog = QProgressDialog("正在处理...", "", 0, 0, self)
                self._progress_dialog.setWindowModality(Qt.WindowModal)
                self._progress_dialog.setCancelButton(None)
                self._progress_dialog.setMinimumDuration(0)
            self._progress_dialog.show()
        else:
            if self._progress_dialog is not None:
                self._progress_dialog.hide()

    def _run_async(self, fn, on_success, on_error, busy_message="正在处理..."):
        if self._worker_thread is not None and self._worker_thread.isRunning():
            return

        class _Worker(QObject):
            finished = pyqtSignal(object)
            failed = pyqtSignal(str)
            def __init__(self, _fn):
                super().__init__()
                self._fn = _fn
            def run(self):
                try: self.finished.emit(self._fn())
                except Exception as e: self.failed.emit(str(e))

        self._set_busy(True, busy_message)
        self._worker_thread = QThread()
        worker = _Worker(fn)
        worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(worker.run)

        def _cleanup():
            self._set_busy(False)
            worker.deleteLater()
            self._worker_thread.quit()
            self._worker_thread.wait()
            self._worker_thread.deleteLater()
            self._worker_thread = None

        worker.finished.connect(lambda payload: (on_success(payload), _cleanup()))
        worker.failed.connect(lambda err: (on_error(err), _cleanup()))
        self._worker_thread.start()

    # ==================== 核心功能 ====================

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择证件照", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif);;所有文件 (*.*)"
        )
        if file_path:
            self.load_image_from_path(file_path)

    def load_image_from_path(self, file_path):
        image, error = IDPhotoUtils.load_image(file_path)
        if image is None:
            QMessageBox.critical(self, "错误", f"无法加载图像: {error}")
            return
        self.current_image = image
        self.original_image = image.copy()
        self.processed_image = None
        self.current_file_path = file_path
        self.processor.set_original_image(image)
        self.processor.history.clear()
        self.processor.history_index = -1
        self._refresh_all_displays()
        info = IDPhotoUtils.get_image_info(image)
        self.status_bar.showMessage(f"已加载: {os.path.basename(file_path)} | 尺寸: {info['size']}")

    def display_image(self, label, image, is_processed=False):
        """将 OpenCV 图像显示到 ImageLabel — 不预缩放，由 ImageLabel.paintEvent 按需缩放"""
        if image is None or label is None:
            return
        # 颜色空间转换 + 构造原始尺寸 QPixmap
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.copy().data, w, h, ch * w, QImage.Format_RGB888)
        if qt_img.isNull():
            return
        pix = QPixmap.fromImage(qt_img)
        # 调用 ImageLabel 自己的接口 — pixmap 内部保存，paintEvent 中按需缩放
        if hasattr(label, 'set_image'):
            label.set_image(pix, is_processed=is_processed)
        else:
            # 容错：对普通 QLabel 回退到 setPixmap
            label.setPixmap(pix)
            label.setText("")

    def save_image(self):
        if self.processed_image is None:
            QMessageBox.warning(self, "警告", "请先处理图像")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存证件照", "",
            "JPEG (*.jpg);;PNG (*.png);;BMP (*.bmp)"
        )
        if file_path:
            success, error = IDPhotoUtils.save_image(self.processed_image, file_path)
            if success:
                self.status_bar.showMessage(f"已保存: {os.path.basename(file_path)}")
            else:
                QMessageBox.critical(self, "错误", f"保存失败: {error}")

    def one_click_process(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先导入图像")
            return

        bg_color = self.bg_color
        base_image = self.current_image.copy()

        def _fn():
            import time
            t0 = time.time()
            
            # 步骤1：使用覆盖式缩放+中心裁剪进行尺寸标准化（双三次插值）
            result = self.cropper.auto_crop(
                base_image,
                size_name='标准一寸',
                use_face_detection=True,
                keep_full=False,
                crop_mode='face'
            )

            # 步骤2：检测裁剪后人脸，用于背景分割定位
            ok2, fr2 = self.processor.face_detector.detect_face(result)
            rect = None
            if ok2 and fr2 is not None:
                x, y, w, h = fr2
                rx = max(0, int(x - w * 0.6))
                ry = max(0, int(y - h * 0.6))
                rw, rh = int(w * 2.2), int(h * 3.2)
                rw = min(rw, result.shape[1] - rx)
                rh = min(rh, result.shape[0] - ry)
                if rw > 0 and rh > 0:
                    rect = (rx, ry, rw, rh)
            
            # 步骤3：背景替换
            result = self.bg_remover.auto_remove_and_replace(
                result, bg_color, 'humanseg', feather_radius=5, rect=rect)

            t1 = time.time()
            return result, t1 - t0

        def _ok(payload):
            result, elapsed = payload
            self.processed_image = result
            self.processor._add_to_history(result)
            self._refresh_all_displays()
            self.time_label.setText(f"处理时间: {elapsed:.2f}s")
            self.status_bar.showMessage("一键处理完成")

        def _err(err):
            QMessageBox.critical(self, "错误", f"处理失败: {err}")
            self.status_bar.showMessage("处理失败")

        self._run_async(_fn, _ok, _err, busy_message="正在一键处理...")

    # ==================== 分步骤处理 ====================

    def _get_work_image(self):
        if self.processed_image is not None:
            return self.processed_image.copy()
        elif self.current_image is not None:
            return self.current_image.copy()
        return None

    def _apply_step_result(self, result, message="处理完成"):
        if result is not None:
            self.processed_image = result
            self.processor._add_to_history(result)
            self._refresh_all_displays()
            self.status_bar.showMessage(message)

    def _on_brightness_changed(self, value):
        self.brightness_label.setText(str(value))

    def _on_contrast_changed(self, value):
        self.contrast_label.setText(str(value))

    def step_resize(self):
        image = self._get_work_image()
        if image is None:
            QMessageBox.warning(self, "警告", "请先导入图像")
            return

        size_text = self.size_combo.currentText()
        size_map = {
            "小一寸 (260×378)": "小一寸",
            "标准一寸 (295×413)": "标准一寸",
            "大一寸 (390×567)": "大一寸",
            "小二寸 (413×531)": "小二寸",
            "标准二寸 (413×579)": "标准二寸",
            "大二寸 (413×626)": "大二寸",
            "身份证 (308×384)": "身份证",
        }
        size_name = size_map.get(size_text, "标准一寸")

        def _fn():
            return self.cropper.auto_crop(
                image, size_name, use_face_detection=True, keep_full=False, crop_mode='face'
            )

        def _ok(result):
            self._apply_step_result(result, f"尺寸标准化完成: {size_name}")

        def _err(err):
            QMessageBox.critical(self, "错误", f"尺寸标准化失败: {err}")

        self._run_async(_fn, _ok, _err, busy_message="正在标准化尺寸...")

    def step_background(self):
        image = self._get_work_image()
        if image is None:
            QMessageBox.warning(self, "警告", "请先导入图像")
            return

        bg_color = self.bg_color
        method_text = self.bg_method_combo.currentText()
        method_map = {
            "PP-HumanSeg (推荐)": "humanseg",
            "GrabCut": "grabcut",
            "颜色分割": "color",
        }
        method = method_map.get(method_text, "humanseg")

        def _fn():
            ok, face_rect = self.processor.face_detector.detect_face(image)
            rect = None
            if ok and face_rect is not None:
                x, y, w, h = face_rect
                rx = max(0, int(x - w * 0.6))
                ry = max(0, int(y - h * 0.6))
                rw, rh = int(w * 2.2), int(h * 3.2)
                rw = min(rw, image.shape[1] - rx)
                rh = min(rh, image.shape[0] - ry)
                if rw > 0 and rh > 0:
                    rect = (rx, ry, rw, rh)
            return self.bg_remover.auto_remove_and_replace(
                image, bg_color, method, feather_radius=5, rect=rect
            )

        def _ok(result):
            self._apply_step_result(result, f"背景替换完成: {bg_color}")

        def _err(err):
            QMessageBox.critical(self, "错误", f"背景替换失败: {err}")

        self._run_async(_fn, _ok, _err, busy_message="正在替换背景...")

    def step_brightness_contrast(self):
        image = self._get_work_image()
        if image is None:
            QMessageBox.warning(self, "警告", "请先导入图像")
            return

        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value()

        if brightness == 0 and contrast == 0:
            QMessageBox.information(self, "提示", "亮度和对比度均为0，无需调整")
            return

        def _fn():
            return self.enhancer.adjust_brightness_contrast(image, brightness, contrast)

        def _ok(result):
            self._apply_step_result(result, f"亮度/对比度调节完成: 亮度={brightness}, 对比度={contrast}")

        def _err(err):
            QMessageBox.critical(self, "错误", f"调节失败: {err}")

        self._run_async(_fn, _ok, _err, busy_message="正在调节亮度/对比度...")

    def step_denoise(self):
        image = self._get_work_image()
        if image is None:
            QMessageBox.warning(self, "警告", "请先导入图像")
            return

        method_text = self.denoise_combo.currentText()

        def _fn():
            ok, face_rect = self.processor.face_detector.detect_face(image)

            if "双边滤波" in method_text:
                result = self.enhancer.denoise_bilateral(image, d=9, sigma_color=75, sigma_space=75)
            elif "中值滤波" in method_text:
                result = self.enhancer.denoise_median(image, kernel_size=3)
            elif "高斯滤波" in method_text:
                result = self.enhancer.denoise_gaussian(image, kernel_size=5)
            elif "非局部均值" in method_text:
                result = self.enhancer.denoise_nlmeans(image, h=10, h_color=10)
            else:
                result = self.enhancer.denoise_bilateral(image)

            if ok and face_rect is not None:
                result = self.enhancer._enhance_face_region(result, face_rect)

            return result

        def _ok(result):
            self._apply_step_result(result, f"去噪完成: {method_text}")

        def _err(err):
            QMessageBox.critical(self, "错误", f"去噪失败: {err}")

        self._run_async(_fn, _ok, _err, busy_message="正在人脸去噪...")

    def step_undo(self):
        """撤销：回退到上一步处理结果

        撤销流程（history = [R1, R2, R3] 为例）:
        - 撤销 1 → history_index=1，显示 R2
        - 撤销 2 → history_index=0，显示 R1
        - 撤销 3 → history_index=-1，processed_image = original_image，界面显示原图（过渡台阶）
        - 撤销 4 → history_index=-1，processed_image = None，显示"暂无结果"
        """
        result = self.processor.undo()
        if result is not None:
            if self.processor.history_index == -1:
                # 刚刚撤销到原始图像 — 先显示原图，让用户看到自己回到了起点，
                # 再撤销一次才清空（两步过渡，不会突兀）
                self.processed_image = self.original_image
                self._refresh_all_displays()
                self.status_bar.showMessage("已回到原图，再撤销一次会清空")
            else:
                # 撤销到中间处理结果
                self.processed_image = result
                self._refresh_all_displays()
                self.status_bar.showMessage("已撤销到上一步")
        else:
            # history_index 已经是 -1 再撤销，清空显示
            if self.original_image is not None:
                self.processed_image = None
                self._refresh_all_displays()
                self.status_bar.showMessage("已恢复到原始图像")
            else:
                QMessageBox.information(self, "提示", "没有可撤销的操作")

    def resizeEvent(self, event):
        """窗口尺寸变化 — ImageLabel 自己会在 paintEvent 中按新尺寸缩放，
        这里只需调用父类 resizeEvent，不需要额外刷新，避免抖动"""
        super().resizeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = IDPhotoMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
