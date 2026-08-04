import cv2
import numpy as np
from collections import defaultdict
from typing import Optional, Tuple

def calculate_size(thresh: np.ndarray, coin_real_diameter: float = 25.0) -> Tuple[float, float, float, Tuple[int, int, int, int], Tuple[int, int, int, int], Tuple[float, float]]:
    """
    计算杯子和硬币的尺寸
    
    参数:
        thresh: 二值化图像
        coin_real_diameter: 硬币实际直径(mm)
    
    返回:
        cup_width_real: 杯子实际宽度(mm)
        cup_height_real: 杯子实际高度(mm)
        coin_diameter_real: 硬币实际直径(mm)
        cup_coords: 杯子边界坐标 (min_x, max_x, min_y, max_y)
        coin_coords: 硬币边界坐标 (min_x, max_x, min_y, max_y)
        coin_center: 硬币中心坐标
    """
    # 寻找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 2:
        return None, None, None, None, None, None
    
    # 按面积排序轮廓，假设最大的两个区域是杯子和硬币
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    
    # 确定哪个是杯子，哪个是硬币（假设杯子面积大于硬币）
    cup_contour = contours[0]
    coin_contour = contours[1]
    
    # 获取杯子和硬币的边界矩形
    cup_x, cup_y, cup_w, cup_h = cv2.boundingRect(cup_contour)
    coin_x, coin_y, coin_w, coin_h = cv2.boundingRect(coin_contour)
    
    # 计算硬币直径（像素）
    coin_diameter_px = max(coin_w, coin_h)
    
    # 计算比例因子（像素到毫米）
    scale_factor = coin_real_diameter / coin_diameter_px
    
    # 计算实际尺寸
    cup_width_real = cup_w * scale_factor
    cup_height_real = cup_h * scale_factor
    coin_diameter_real = coin_diameter_px * scale_factor
    
    # 计算硬币中心
    coin_center = (coin_x + coin_w/2, coin_y + coin_h/2)
    
    # 返回结果
    cup_coords = (cup_x, cup_x + cup_w, cup_y, cup_y + cup_h)
    coin_coords = (coin_x, coin_x + coin_w, coin_y, coin_y + coin_h)
    
    return cup_width_real, cup_height_real, coin_diameter_real, cup_coords, coin_coords, coin_center

def annotate_image(img, cup_coords, coin_coords, coin_center, cup_width, cup_height, coin_diameter):
    """
    在图像上标注杯子和硬币的尺寸信息
    
    参数:
        img: 输入的灰度图像
        cup_coords: 杯子边界坐标 (min_x, max_x, min_y, max_y)
        coin_coords: 硬币边界坐标 (min_x, max_x, min_y, max_y)
        coin_center: 硬币中心坐标
        cup_width: 杯子宽度 (mm)
        cup_height: 杯子高度 (mm)
        coin_diameter: 硬币直径 (mm)
        
    返回:
        标注后的彩色图像
    """
    # 将灰度图像转换为BGR彩色图像
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 定义颜色 - 使用BGR格式 (OpenCV默认)
    red = (0, 0, 255)    # 红色
    green = (0, 255, 0)  # 绿色
    blue = (255, 0, 0)   # 蓝色
    purple = (255, 0, 255)  # 紫色
    
    cup_min_x, cup_max_x, cup_min_y, cup_max_y = cup_coords
    coin_min_x, coin_max_x, coin_min_y, coin_max_y = coin_coords
    
    # 计算中点坐标
    cup_mid_y_width = (cup_min_y + cup_max_y) // 2
    cup_mid_x_height = (cup_min_x + cup_max_x) // 2
    
    coin_mid_y = (coin_min_y + coin_max_y) // 2
    coin_mid_x = (coin_min_x + coin_max_x) // 2
    
    # 绘制杯子边界框
    cv2.rectangle(color_img, (cup_min_x, cup_min_y), (cup_max_x, cup_max_y), green, 2)
    
    # 绘制硬币边界框
    cv2.rectangle(color_img, (coin_min_x, coin_min_y), (coin_max_x, coin_max_y), blue, 2)
    
    # 绘制硬币中心标记
    cv2.circle(color_img, (int(coin_center[0]), int(coin_center[1])), 5, purple, -1)
    
    # ---- 绘制杯子宽度标注 ----
    cv2.line(color_img, (cup_min_x, cup_mid_y_width-10), (cup_min_x, cup_mid_y_width+10), red, 2)
    cv2.line(color_img, (cup_max_x, cup_mid_y_width-10), (cup_max_x, cup_mid_y_width+10), red, 2)
    cv2.line(color_img, (cup_min_x, cup_mid_y_width), (cup_max_x, cup_mid_y_width), red, 2)
    cv2.line(color_img, (cup_max_x, cup_mid_y_width), (cup_max_x + 40, cup_mid_y_width), red, 2)
    cv2.putText(color_img, f"{cup_width:.1f} mm", (cup_max_x + 45, cup_mid_y_width + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)
    
    # ---- 绘制杯子高度标注 ----
    cv2.line(color_img, (cup_mid_x_height-10, cup_min_y), (cup_mid_x_height+10, cup_min_y), red, 2)
    cv2.line(color_img, (cup_mid_x_height-10, cup_max_y), (cup_mid_x_height+10, cup_max_y), red, 2)
    cv2.line(color_img, (cup_mid_x_height, cup_min_y), (cup_mid_x_height, cup_max_y), red, 2)
    cv2.line(color_img, (cup_mid_x_height, cup_max_y), (cup_mid_x_height, cup_max_y + 40), red, 2)
    cv2.putText(color_img, f"{cup_height:.1f} mm", (cup_mid_x_height + 5, cup_max_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)
    
    # ---- 绘制硬币直径标注 ----
    # 水平直径
    cv2.line(color_img, (coin_min_x, coin_mid_y-5), (coin_min_x, coin_mid_y+5), purple, 2)
    cv2.line(color_img, (coin_max_x, coin_mid_y-5), (coin_max_x, coin_mid_y+5), purple, 2)
    cv2.line(color_img, (coin_min_x, coin_mid_y), (coin_max_x, coin_mid_y), purple, 2)
    cv2.line(color_img, (coin_max_x, coin_mid_y), (coin_max_x + 40, coin_mid_y), purple, 2)
    cv2.putText(color_img, f"{coin_diameter:.1f} mm", (coin_max_x + 45, coin_mid_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, purple, 2)
    
    # 垂直直径
    cv2.line(color_img, (coin_mid_x-5, coin_min_y), (coin_mid_x+5, coin_min_y), purple, 2)
    cv2.line(color_img, (coin_mid_x-5, coin_max_y), (coin_mid_x+5, coin_max_y), purple, 2)
    cv2.line(color_img, (coin_mid_x, coin_min_y), (coin_mid_x, coin_max_y), purple, 2)
    cv2.line(color_img, (coin_mid_x, coin_max_y), (coin_mid_x, coin_max_y + 40), purple, 2)
    cv2.putText(color_img, f"{coin_diameter:.1f} mm", (coin_mid_x + 5, coin_max_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, purple, 2)
    
    return color_img

def process_image(input_img: np.ndarray, output_dir: Optional[str] = None) -> Tuple[np.ndarray, float, float, float]:
    """
    处理图像并返回标注后的图像及杯子和硬币的尺寸
    
    参数:
        input_img: 输入图像
        output_dir: 输出图像目录，用于调试，默认为 None（不保存）
    
    返回:
        标注后的图像、杯子宽度、杯子高度、硬币直径
    """
    # 转换为灰度图
    if len(input_img.shape) == 3:
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = input_img
    
    # 二值化图像
    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    
    # 计算尺寸
    cup_width, cup_height, coin_diameter, cup_coords, coin_coords, coin_center = calculate_size(thresh)
    
    if cup_width is None or cup_height is None or coin_diameter is None:
        print("无法计算尺寸，请检查图像质量")
        return gray_img, None, None, None
    
    # 标注图像
    annotated_img = annotate_image(gray_img, cup_coords, coin_coords, coin_center, cup_width, cup_height, coin_diameter)
    
    if output_dir is not None:
        # 保存图像
        output_path = f"{output_dir}/annotated_image.png"
        cv2.imwrite(output_path, annotated_img)
        print(f"已将标注后的图像保存至: {output_path}")
    
    return annotated_img, cup_width, cup_height, coin_diameter