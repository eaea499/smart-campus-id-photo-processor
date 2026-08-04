import cv2
import numpy as np
from scipy import ndimage
from typing import Tuple, Optional
from collections import deque


class ImageProcessor:
    def __init__(self, img: np.ndarray, output_dir: Optional[str] = None, low_th: float = 0.001, high_th: float = 0.21):
        """
        初始化 ImageProcessor 类

        参数:
        img (np.ndarray): 输入图像（BGR 格式或灰度图，shape 为 (H, W) 或 (H, W, 3)）
        output_dir (str): 输出图像目录，用于调试，默认为 None（不保存）
        low_th (float): Canny 边缘检测低阈值，默认值为 0.001
        high_th (float): Canny 边缘检测高阈值，默认值为 0.21
        """
        self.img = img
        self.gray_img = None
        self.gray_normalized = None
        self.edge_result = None
        self.contour_result = None
        self.output_dir = output_dir
        self.low_th = low_th
        self.high_th = high_th

    def preprocess(self):
        """
        图像预处理：转为灰度图并归一化
        """
        if len(self.img.shape) == 3:
            self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_img = self.img.copy()
        self.gray_normalized = self.gray_img.astype(np.float64) / 255.0  # 归一化到 [0, 1]
        
        if self.output_dir:
            cv2.imwrite(f"{self.output_dir}/1_gray_img.png", self.gray_img)

    def canny_manual(self, low_th: Optional[float] = None, high_th: Optional[float] = None):
        """
        手动实现 Canny 边缘检测（含高斯平滑、非极大值抑制、双阈值跟踪）

        参数:
        low_th (float): 低阈值，默认使用类初始化时的值
        high_th (float): 高阈值，默认使用类初始化时的值

        返回:
        edge (np.ndarray): Canny 边缘检测结果（二值图，0/1）
        """
        low_th = self.low_th if low_th is None else low_th
        high_th = self.high_th if high_th is None else high_th
        
        # 高斯平滑
        kernel_size = 9
        sigma = 0.8
        smoothed = ndimage.gaussian_filter(self.gray_normalized, sigma=sigma)
        
        if self.output_dir:
            smoothed_vis = (smoothed * 255).astype(np.uint8)
            cv2.imwrite(f"{self.output_dir}/2_smoothed.png", smoothed_vis)

        # Sobel 梯度计算
        gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        grad_x = ndimage.convolve(smoothed, gx)
        grad_y = ndimage.convolve(smoothed, gy)
        grad_mag = np.hypot(grad_x, grad_y)
        grad_mag = grad_mag / (np.max(grad_mag) + 1e-10)  # 归一化
        
        if self.output_dir:
            grad_mag_vis = (grad_mag * 255).astype(np.uint8)
            cv2.imwrite(f"{self.output_dir}/3_grad_mag.png", grad_mag_vis)

        # 非极大值抑制
        h, w = grad_mag.shape
        suppressed = np.zeros((h, w), dtype=bool)
        grad_dir = np.arctan2(grad_y, grad_x) * 180 / np.pi  # 角度转换为度数

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                theta = grad_dir[i, j]
                if (-22.5 <= theta < 22.5) or (157.5 <= theta <= 180) or (-180 <= theta < -157.5):
                    ne1, ne2 = grad_mag[i, j + 1], grad_mag[i, j - 1]
                elif (22.5 <= theta < 67.5) or (-157.5 <= theta < -112.5):
                    ne1, ne2 = grad_mag[i + 1, j + 1], grad_mag[i - 1, j - 1]
                elif (67.5 <= theta < 112.5) or (-112.5 <= theta < -67.5):
                    ne1, ne2 = grad_mag[i + 1, j], grad_mag[i - 1, j]
                else:
                    ne1, ne2 = grad_mag[i + 1, j - 1], grad_mag[i - 1, j + 1]
                if grad_mag[i, j] >= ne1 and grad_mag[i, j] >= ne2:
                    suppressed[i, j] = True
        
        if self.output_dir:
            suppressed_vis = (suppressed * 255).astype(np.uint8)
            cv2.imwrite(f"{self.output_dir}/4_suppressed.png", suppressed_vis)

        # 双阈值检测与滞后跟踪
        lower = suppressed & (grad_mag >= low_th)
        upper = suppressed & (grad_mag >= high_th)
        edge = np.zeros_like(suppressed, dtype=bool)
        edge[upper] = True

        queue = deque(np.argwhere(upper))
        while queue:
            i, j = queue.popleft()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w and lower[ni, nj] and not edge[ni, nj]:
                        edge[ni, nj] = True
                        queue.append((ni, nj))
        
        if self.output_dir:
            edge_vis = (edge * 255).astype(np.uint8)
            cv2.imwrite(f"{self.output_dir}/5_edge.png", edge_vis)
            
        return edge

    def get_ellipse_kernel(self, length=5, width=3):
        """
        生成椭圆结构元素（OpenCV 格式）

        参数:
        length (int): 椭圆长轴长度，默认值为 5
        width (int): 椭圆短轴长度，默认值为 3

        返回:
        kernel (np.ndarray): 椭圆结构元素
        """
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, length))

    def dilation(self, im, kernel):
        """
        二值图像膨胀

        参数:
        im (np.ndarray): 输入二值图像
        kernel (np.ndarray): 结构元素

        返回:
        dilated_im (np.ndarray): 膨胀后的二值图像
        """
        result = cv2.dilate(im.astype(np.uint8), kernel, iterations=1).astype(bool)
        
        # 保存膨胀后的图像
        if self.output_dir:
            result_vis = (result * 255).astype(np.uint8)
            cv2.imwrite(f"{self.output_dir}/6_dilated.png", result_vis)
            
        return result

    def erosion(self, im, kernel):
        """
        二值图像腐蚀

        参数:
        im (np.ndarray): 输入二值图像
        kernel (np.ndarray): 结构元素

        返回:
        eroded_im (np.ndarray): 腐蚀后的二值图像
        """
        result = cv2.erode(im.astype(np.uint8), kernel, iterations=1).astype(bool)
        
        # 保存腐蚀后的图像
        if self.output_dir:
            result_vis = (result * 255).astype(np.uint8)
            cv2.imwrite(f"{self.output_dir}/7_eroded.png", result_vis)
            
        return result

    def closing(self, im, kernel):
        """
        二值图像闭运算

        参数:
        im (np.ndarray): 输入二值图像
        kernel (np.ndarray): 结构元素

        返回:
        closed_im (np.ndarray): 闭运算后的二值图像
        """
        result = cv2.morphologyEx(im.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        
        # 保存闭运算后的图像
        if self.output_dir:
            result_vis = (result * 255).astype(np.uint8)
            cv2.imwrite(f"{self.output_dir}/8_closed.png", result_vis)
            
        return result


    def extract_largest_contour(self, im):
        """
        提取最大连通区域
        
        参数:
        im (np.ndarray): 输入二值图像
        
        返回:
        largest_contour (np.ndarray): 最大连通区域的二值图像
        """
        # 兼容低版本OpenCV的findContours函数
        contours = None
        hierarchy = None
        
        if cv2.__version__.startswith('4') or cv2.__version__.startswith('3'):
            # OpenCV 4.x/3.x 返回值: contours, hierarchy
            contours, hierarchy = cv2.findContours(
                im.astype(np.uint8),  # 确保输入是uint8类型
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
        else:
            # OpenCV 2.x 返回值: image, contours, hierarchy
            _, contours, hierarchy = cv2.findContours(
                im.astype(np.uint8),  # 确保输入是uint8类型
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
        
        if not contours:
            return im.astype(np.uint8)  # 返回uint8类型
        
        largest = max(contours, key=lambda c: cv2.contourArea(c))
        result = np.zeros_like(im, dtype=np.uint8)  # 明确指定为uint8类型
        cv2.drawContours(result, [largest], -1, 1, thickness=cv2.FILLED)
        ######cv2.drawContours(result, [largest], -1, 1, thickness=1)  # 将FILLED改为1
        # 保存最大轮廓图像
        if self.output_dir:
            cv2.imwrite(f"{self.output_dir}/9_largest_contour.png", result * 255)
            
        return result
   
   
    def process(self):
        """
        图像处理主流程

        执行 Canny 边缘检测和形态学处理，生成边缘检测结果和轮廓提取结果
        """
        self.preprocess()
        self.edge_result = self.canny_manual()

        ellipse_kernel = self.get_ellipse_kernel()  # 5x3 椭圆核

        dilated = self.dilation(self.edge_result, ellipse_kernel)
        denoised = self.closing(dilated, ellipse_kernel)
        largest_contour = self.extract_largest_contour(denoised)
        self.contour_result = self.closing(largest_contour, ellipse_kernel)  # 平滑轮廓
        
        # 保存最终轮廓结果
        if self.output_dir:
            contour_vis = (self.contour_result * 255).astype(np.uint8)
            cv2.imwrite(f"{self.output_dir}/10_final_contour.png", contour_vis)

    def get_results(self):
        """
        获取处理结果

        返回:
        edge_result (np.ndarray): Canny 边缘检测结果（二值图，np.uint8 类型，0=黑色，1=白色）
        contour_result (np.ndarray): 最大轮廓提取结果（经过形态学平滑处理的二值图）
        """
        if self.edge_result is None or self.contour_result is None:
            raise ValueError("请先调用 process 方法进行图像处理")
        return self.edge_result.astype(np.uint8), self.contour_result.astype(np.uint8)

def process_image(
    input_img: np.ndarray, 
    output_dir: Optional[str] = None, 
    low_th: float = 0.001, 
    high_th: float = 0.21
) -> Tuple[np.ndarray, np.ndarray]:
    """
    处理图像并返回边缘和轮廓结果，支持自定义 Canny 阈值
    
    参数:
    input_img (np.ndarray): 输入图像
    output_dir (str): 输出图像目录，用于调试，默认为 None（不保存）
    low_th (float): Canny 边缘检测低阈值，默认值为 0.001
    high_th (float): Canny 边缘检测高阈值，默认值为 0.21
    
    返回:
    Tuple[np.ndarray, np.ndarray]: 边缘检测结果和轮廓提取结果
    """
    processor = ImageProcessor(input_img, output_dir, low_th, high_th)
    processor.process()
    edge, contour = processor.get_results()
    return edge, contour