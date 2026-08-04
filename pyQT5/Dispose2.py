import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology, measure

def prewitt_edge_detection(img: np.ndarray) -> np.ndarray:
    """
    使用Prewitt算子进行边缘检测
    :param img: 输入的灰度图像，类型为uint8
    :return: 边缘检测结果，类型为uint8
    """
    # 定义Prewitt卷积核
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float32)
    
    prewitt_y = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]], dtype=np.float32)
    
    rows, cols = img.shape
    
    # 将输入图像转换为float32类型以避免溢出
    img_float = img.astype(np.float32)
    
    # 初始化梯度矩阵
    grad_x = np.zeros((rows-2, cols-2), dtype=np.float32)
    grad_y = np.zeros((rows-2, cols-2), dtype=np.float32)
    
    # 执行卷积运算
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            # 提取3x3邻域（修正切片范围）
            region = img_float[i-1:i+2, j-1:j+2]  # 注意这里是i+2和j+2
            
            # 计算梯度
            gx = np.sum(region * prewitt_x)
            gy = np.sum(region * prewitt_y)
            
            grad_x[i-1, j-1] = gx
            grad_y[i-1, j-1] = gy
    
    # 计算梯度幅值
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # 对比度增强
    contrast_factor = 0.5
    grad_mag = grad_mag**contrast_factor
    
    # 归一化到0-255范围
    max_val = np.max(grad_mag)
    if max_val > 0:
        grad_mag_normalized = (grad_mag / max_val * 255).astype(np.uint8)
    else:
        grad_mag_normalized = np.zeros_like(grad_mag, dtype=np.uint8)
    
    # 添加边界填充
    edge_img = np.zeros((rows, cols), dtype=np.uint8)
    edge_img[1:-1, 1:-1] = grad_mag_normalized
    return edge_img
def label_regions(img: np.ndarray) -> tuple:
    """
    区域标记（4连通）
    :param img: 输入的二值图像
    :return: 标记结果和标记数量
    """
    rows, cols = img.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    current_label = 1
    queue = []
    
    for i in range(rows):
        for j in range(cols):
            if img[i, j] and labels[i, j] == 0:  # 直接使用布尔值
                queue.append((i, j))
                labels[i, j] = current_label
                
                while queue:
                    x, y = queue.pop(0)
                    
                    # 检查4邻域
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and img[nx, ny] and labels[nx, ny] == 0:
                            labels[nx, ny] = current_label
                            queue.append((nx, ny))
                
                current_label += 1
    
    return labels, current_label - 1

def filter_regions(labels: np.ndarray, num_labels: int, min_area: int = 100) -> np.ndarray:
    """
    计算区域属性并筛选
    :param labels: 标记结果
    :param num_labels: 标记数量
    :param min_area: 最小面积阈值
    :return: 筛选后的二值图像
    """
    rows, cols = labels.shape
    areas = np.zeros(num_labels, dtype=np.int32)
    
    # 计算每个区域的面积
    for i in range(rows):
        for j in range(cols):
            label = labels[i, j]
            if label > 0:
                areas[label - 1] += 1
    
    # 创建筛选后的图像
    filtered = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            label = labels[i, j]
            if label > 0 and areas[label - 1] >= min_area:
                filtered[i, j] = 255
    
    return filtered

def strel_ellipse(radius: tuple) -> np.ndarray:
    """
    生成椭圆结构元素
    :param radius: 椭圆半径 (a, b)
    :return: 椭圆结构元素
    """
    a, b = radius
    kernel = np.zeros((2*a + 1, 2*b + 1), dtype=np.uint8)
    cx, cy = a + 1, b + 1
    
    for i in range(1, 2*a + 2):
        for j in range(1, 2*b + 2):
            dx = i - cx
            dy = j - cy
            if (dx**2)/(a**2) + (dy**2)/(b**2) <= 1 + 1e-6:
                kernel[i-1, j-1] = 1
    
    return kernel

def my_imdilate(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    自定义膨胀操作
    :param img: 输入图像
    :param se: 结构元素
    :return: 膨胀后的图像
    """
    # 获取结构元素尺寸和锚点（中心位置）
    kh, kw = se.shape
    anchor_y, anchor_x = (kh + 1) // 2, (kw + 1) // 2
    
    # 计算填充尺寸
    pad_top = anchor_y - 1
    pad_bottom = kh - anchor_y
    pad_left = anchor_x - 1
    pad_right = kw - anchor_x
    
    # 边界填充
    rows, cols = img.shape
    padded_img = np.zeros((rows + pad_top + pad_bottom, cols + pad_left + pad_right), dtype=np.uint8)
    padded_img[pad_top:pad_top+rows, pad_left:pad_left+cols] = img
    
    # 初始化输出图像
    dilated_img = np.zeros((rows, cols), dtype=np.uint8)
    
    # 遍历每个像素进行膨胀操作
    for i in range(rows):
        for j in range(cols):
            # 计算结构元素在填充图像中的对应区域
            y_start = i + pad_top - anchor_y + 1
            y_end = y_start + kh
            x_start = j + pad_left - anchor_x + 1
            x_end = x_start + kw
            
            # 提取邻域区域并与结构元素进行逻辑或
            neighborhood = padded_img[y_start:y_end, x_start:x_end]
            # 若邻域中存在任意前景像素（非0），则当前像素置为前景
            if np.any(neighborhood > 0):
                dilated_img[i, j] = 255  # 前景设为255
    
    return dilated_img

def my_imerode(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    自定义腐蚀操作
    :param img: 输入图像
    :param se: 结构元素
    :return: 腐蚀后的图像
    """
    kh, kw = se.shape
    anchor_y, anchor_x = (kh + 1) // 2, (kw + 1) // 2
    pad_top = anchor_y - 1
    pad_bottom = kh - anchor_y
    pad_left = anchor_x - 1
    pad_right = kw - anchor_x

    rows, cols = img.shape
    padded_img = np.zeros((rows + pad_top + pad_bottom, cols + pad_left + pad_right), dtype=np.uint8)
    padded_img[pad_top:pad_top+rows, pad_left:pad_left+cols] = img

    eroded_img = np.zeros((rows, cols), dtype=np.uint8)
    mask = se > 0  # 创建结构元素掩码

    for i in range(rows):
        for j in range(cols):
            y_start = i + pad_top - anchor_y + 1
            y_end = y_start + kh
            x_start = j + pad_left - anchor_x + 1
            x_end = x_start + kw

            neighborhood = padded_img[y_start:y_end, x_start:x_end]
            # 仅检查结构元素中为1的位置对应的像素
            if np.all(neighborhood[mask] > 0):
                eroded_img[i, j] = 255
    
    return eroded_img

def remove_boundary_connected(processed_img: np.ndarray) -> np.ndarray:
    """
    移除与边界相连的区域
    :param processed_img: 输入图像
    :return: 处理后的图像
    """
    height, width = processed_img.shape
    visited = np.zeros((height, width), dtype=bool)
    queue = []
    
    # 初始化左右边界种子点
    for y in range(height):
        queue.append((y, 0))
        queue.append((y, width - 1))
        visited[y, 0] = True
        visited[y, width - 1] = True
    
    # 定义上下左右偏移
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while queue:
        y, x = queue.pop(0)
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx]:
                if processed_img[ny, nx] != 255:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
    
    # 将与边界相连的区域设为背景
    result = processed_img.copy()
    for y in range(height):
        for x in range(width):
            if visited[y, x]:
                result[y, x] = 255
    
    return result

def invert_gray(img: np.ndarray) -> np.ndarray:
    """
    灰度图反色
    :param img: 输入图像
    :return: 反色后的图像
    """
    return 255 - img

def image_processing_pipeline(img: np.ndarray) -> np.ndarray:
    """
    完整图像处理流程封装函数（直接处理图像数据）
    :param img: 输入的numpy数组图像数据
    :return: 处理后的二值图像 (0-255 uint8)
    """
    
    if len(img.shape) > 2:
        img = img[:,:,0]  # 取第一通道（兼容伪灰度图）
    
    # 确保uint8类型
    img = img.astype(np.uint8) if img.max() > 1 else (img*255).astype(np.uint8)
    

    gray_img1 = img
    
    # Prewitt边缘检测
    BW2 = prewitt_edge_detection(gray_img1)
    
    # 创建椭圆结构元素
    SE = strel_ellipse((17, 17))
    
    # 膨胀操作
    dilated_edge = my_imdilate(BW2, SE)
    
    # 阈值处理
    threshold = 75
    BW = BW2 > threshold
    
    # 区域标记
    labels, num_labels = label_regions(BW)
    
    # 区域筛选
    filtered_img = filter_regions(labels, num_labels, 200)
    SE1 = strel_ellipse((13, 13))
    
    # 再次膨胀和腐蚀
    dilated_edge1 = my_imdilate(filtered_img, SE1)
    dilated_edge2 = my_imerode(dilated_edge1, SE1)
    
    # 移除与边界相连的区域
    result = remove_boundary_connected(dilated_edge2)
    
    # 反色
    #return invert_gray(result)
    BW2=invert_gray(BW2)
    return BW2,filtered_img


