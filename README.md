# 智慧校园证件照标准化预处理系统

一个基于 Python、OpenCV 和 PyQt5 的校园证件照标准化预处理工具，支持人脸检测、智能裁剪、背景替换和图像增强，并提供图形界面操作。

## 主要功能

- 证件照导入、预览和保存
- 人脸检测与证件照比例裁剪
- 蓝色、白色、红色背景替换
- GrabCut、PP-HumanSeg 等背景处理方式
- 亮度、对比度、去噪、锐化和自动增强
- 学生证、一卡通、考试报名和护照签证等预设方案
- 条码图像增强与识别辅助工具

## 项目结构

```text
.
├── IDPhotoProcessor/                                      # 证件照处理主程序
│   ├── IDPhotoGUI.py                                     # 证件照 GUI 入口
│   ├── IDPhotoProcessor.py                               # 核心处理流程
│   ├── FaceDetector.py                                   # 人脸检测与裁剪
│   ├── BackgroundRemover.py                              # 背景替换
│   ├── ImageEnhancer.py                                  # 图像增强
│   ├── BarcodeGUI.py                                     # 条码处理 GUI 入口
│   ├── BarcodeProcessor.py                               # 条码处理模块
│   └── yolov8n.pt                                        # YOLOv8 人脸检测模型
├── human_pp_humansegv2_lite_192x192_inference_model.../   # 人像分割模型
├── portrait_pp_humansegv1_lite_398x224_inference_model.../ # 人像分割模型
├── pyQT5/                                                # 图像处理课程实验 GUI
├── requirements.txt                                      # Python 依赖
└── 智慧校园证件照标准化预处理系统_*.md                   # 需求与界面设计文档
```

## 环境要求

- Windows
- Python 3.12（推荐；其他 Python 3.8+ 版本可能需要调整依赖版本）
- 可选：支持摄像头的设备

建议使用虚拟环境安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> 项目目录中的本地虚拟环境不会提交到 GitHub。依赖应通过 `requirements.txt` 重新安装。

## 启动证件照处理程序

```powershell
cd IDPhotoProcessor
python IDPhotoGUI.py
```

首次启动时，程序会尝试加载项目根目录中的 PP-HumanSeg 模型和 `IDPhotoProcessor/yolov8n.pt`。如果 YOLOv8 依赖或模型加载失败，人脸检测模块会尝试回退到 OpenCV Haar 级联检测；背景分割也可以回退到传统 GrabCut 方法。

## 启动条码处理程序

```powershell
cd IDPhotoProcessor
python BarcodeGUI.py
```

## 启动图像处理实验程序

```powershell
cd pyQT5
python main.py
```

## 隐私与使用说明

- 个人证件照属于敏感个人信息，请仅使用经过授权的图片进行测试。
- 仓库中的测试证件照和生成图片已通过 `.gitignore` 排除，不应上传到公开仓库。
- 本项目主要用于学习、研究和教学演示，证件照是否符合具体业务或证件办理规范仍需人工确认。
- 项目包含第三方模型文件，使用和再分发时请遵守相应模型及框架的许可协议。

## 文档

- [需求分析](智慧校园证件照标准化预处理系统_需求分析.md)
- [界面设计](智慧校园证件照标准化预处理系统_界面设计.md)
- [证件照模块说明](IDPhotoProcessor/README.md)
