# BackgroundRemover.py
"""
背景替换模块
用于证件照的人像分割和背景颜色替换
使用 PP-HumanSeg 深度学习模型进行人像分割
"""

import os
import cv2
import numpy as np


class BackgroundRemover:
    """背景移除器类 — 核心使用 PP-HumanSeg 深度学习人像分割"""

    BACKGROUND_COLORS = {
        'blue': (219, 142, 67),
        'white': (255, 255, 255),
        'red': (0, 0, 206),
    }

    _humanseg_predictor = None
    _humanseg_input_name = None
    _humanseg_output_name = None
    _humanseg_input_size = None  # (w, h)，根据模型类型自动设置

    _MODEL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 优先级：通用人像分割 > 面部人像
    _MODEL_CANDIDATES = [
        ('human_pp_humansegv2_lite_192x192_inference_model_with_softmax', (192, 192)),
        ('human_pp_humansegv1_lite_192x192_inference_model_with_softmax', (192, 192)),
        ('human_pp_humansegv1_server_512x512_inference_model_with_softmax', (512, 512)),
        ('portrait_pp_humansegv1_lite_398x224_inference_model_with_softmax', (398, 224)),
    ]

    def __init__(self):
        pass

    @classmethod
    def _ensure_humanseg_loaded(cls):
        if cls._humanseg_predictor is not None:
            return True

        for model_dir_name, input_size in cls._MODEL_CANDIDATES:
            model_dir = os.path.join(cls._MODEL_BASE, model_dir_name)
            model_file = os.path.join(model_dir, 'model.pdmodel')
            params_file = os.path.join(model_dir, 'model.pdiparams')
            if not os.path.isfile(model_file) or not os.path.isfile(params_file):
                continue
            try:
                import paddle
                from paddle.inference import Config as PConfig, create_predictor
                config = PConfig(model_file, params_file)
                config.disable_gpu()
                config.set_cpu_math_library_num_threads(4)
                config.switch_ir_optim(False)
                predictor = create_predictor(config)
                cls._humanseg_predictor = predictor
                cls._humanseg_input_name = predictor.get_input_names()[0]
                cls._humanseg_output_name = predictor.get_output_names()[0]
                cls._humanseg_input_size = input_size
                model_type = '通用人像' if 'human_' in model_dir_name else '面部人像'
                print(f"[BG] PP-HumanSeg ({model_type}, {input_size[0]}x{input_size[1]}) 加载成功")
                return True
            except Exception as e:
                print(f"[BG] {model_dir_name} 加载失败: {e}")
                continue

        return False

    def _humanseg_predict(self, image):
        """
        使用 PP-HumanSeg 进行人像分割，返回 255=前景 0=背景 的掩码
        """
        if not self._ensure_humanseg_loaded():
            return None

        h, w = image.shape[:2]
        iw, ih = self._humanseg_input_size
        inp = cv2.resize(image, (iw, ih))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp = inp.astype(np.float32) / 255.0
        inp = (inp - 0.5) / 0.5
        inp = inp.transpose((2, 0, 1))
        inp = np.expand_dims(inp, 0)

        input_handle = self._humanseg_predictor.get_input_handle(self._humanseg_input_name)
        input_handle.copy_from_cpu(inp)
        self._humanseg_predictor.run()
        output_handle = self._humanseg_predictor.get_output_handle(self._humanseg_output_name)
        result = output_handle.copy_to_cpu()

        mask = result[0, 1]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
        return mask
    
    def remove_background_grabcut(self, image, rect=None, iter_count=10):
        """
        使用GrabCut算法移除背景

        Args:
            image: numpy数组格式的图像 (BGR格式)
            rect: 初始矩形区域 (x, y, w, h)，如果为None则使用整个图像
            iter_count: GrabCut迭代次数

        Returns:
            tuple: (result, mask)
                - result: 处理后的图像（RGBA格式，背景透明）
                - mask: 分割掩码
        """
        if image is None:
            return None, None

        h, w = image.shape[:2]

        # 初始化掩码
        mask = np.zeros((h, w), np.uint8)

        # 背景模型和前景模型
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # 如果没有提供rect，使用整个图像的中心区域
        if rect is None:
            # 缩小rect范围，避免边缘干扰
            margin_x = int(w * 0.08)
            margin_y = int(h * 0.08)
            rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)

        # 验证rect
        x, y, rw, rh = rect
        x = max(0, x)
        y = max(0, y)
        rw = max(10, min(rw, w - x))
        rh = max(10, min(rh, h - y))
        rect = (x, y, rw, rh)

        # 应用GrabCut
        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model,
                        iter_count, cv2.GC_INIT_WITH_RECT)
        except Exception:
            return None, None

        # 创建掩码：0和2为背景，1和3为前景
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # 形态学平滑掩码
        kernel = np.ones((5, 5), np.uint8)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

        # 创建RGBA图像
        result = image * mask2[:, :, np.newaxis]

        # 添加Alpha通道
        alpha = mask2 * 255
        result = np.dstack((result, alpha))

        return result, mask2
    
    def remove_background_color(self, image, lower_color=None, upper_color=None,
                                 color_space='hsv', erode_iter=2, dilate_iter=2):
        """
        基于颜色范围移除背景

        策略：先采样图像四角+边缘，找到主背景色，标记背景=0，前景=255

        Args:
            image: numpy数组格式的图像
            lower_color: 颜色下限
            upper_color: 颜色上限
            color_space: 颜色空间 ('hsv' 或 'rgb')
            erode_iter: 腐蚀迭代次数
            dilate_iter: 膨胀迭代次数

        Returns:
            tuple: (result, mask)  mask: 255=前景(人), 0=背景
        """
        if image is None:
            return None, None

        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 采样四角 + 四边中点，找出主背景色
        edge_pixels = []
        border = max(2, int(min(h, w) * 0.03))
        for y in range(0, border):
            for x in range(0, w, 5):
                edge_pixels.append(hsv[y, x])
        for y in range(h - border, h):
            for x in range(0, w, 5):
                edge_pixels.append(hsv[y, x])
        for x in range(0, border):
            for y in range(0, h, 5):
                edge_pixels.append(hsv[y, x])
        for x in range(w - border, w):
            for y in range(0, h, 5):
                edge_pixels.append(hsv[y, x])

        if len(edge_pixels) < 50:
            edge_pixels = hsv.reshape(-1, 3)[::20]

        edge = np.array(edge_pixels)
        median_h = np.median(edge[:, 0])
        median_s = np.median(edge[:, 1])
        median_v = np.median(edge[:, 2])

        # 宽度放得比较宽容，避免漏掉背景
        dH = 20
        dS = 40
        dV = 50
        low = np.array([max(0, median_h - dH), max(0, median_s - dS), max(0, median_v - dV)])
        high = np.array([min(180, median_h + dH), min(255, median_s + dS), min(255, median_v + dV)])

        bg_mask = cv2.inRange(hsv, low, high)

        # 前景 = 非背景
        fg_mask = cv2.bitwise_not(bg_mask)

        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        result = image.copy()
        result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = fg_mask

        return result, fg_mask
    
    def remove_background_edge(self, image, canny_low=50, canny_high=150):
        """
        基于边缘检测的人像分割（适用于背景复杂的情况）
        
        Args:
            image: numpy数组格式的图像
            canny_low: Canny低阈值
            canny_high: Canny高阈值
            
        Returns:
            tuple: (result, mask)
        """
        if image is None:
            return None, None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, canny_low, canny_high)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建掩码
        mask = np.zeros_like(gray)
        
        if contours:
            # 找到最大的轮廓（假设是人像）
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            
            # 填充轮廓内部
            mask = cv2.fillPoly(mask, [largest_contour], 255)
        
        # 形态学操作优化
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 创建结果图像
        result = image.copy()
        result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = mask
        
        return result, mask
    
    def replace_background(self, image, mask, bg_color='blue', 
                           feather_radius=5):
        """
        替换背景颜色
        
        Args:
            image: numpy数组格式的图像 (BGR或BGRA)
            mask: 人像掩码（255为前景，0为背景）
            bg_color: 背景颜色名称或BGR元组
            feather_radius: 边缘羽化半径
            
        Returns:
            numpy数组: 替换背景后的图像 (BGR格式)
        """
        if image is None or mask is None:
            return None
        
        # 获取背景颜色
        if isinstance(bg_color, str):
            bg_color = self.BACKGROUND_COLORS.get(bg_color, (255, 0, 0))
        
        h, w = image.shape[:2]
        
        # 创建背景图像
        background = np.full((h, w, 3), bg_color, dtype=np.uint8)
        
        # 确保mask是单通道
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # HumanSeg's low-resolution boundary confidence includes source
        # background colour. Pull the alpha edge inward before compositing so
        # a pale source background cannot become a visible white halo.
        mask_norm = mask.astype(np.float32) / 255.0
        mask_norm = cv2.erode(mask_norm, np.ones((3, 3), dtype=np.uint8), iterations=1)

        # Keep just a narrow transition. A large blur reintroduces the
        # uncertain pixels that the erosion intentionally removed.
        if feather_radius > 0:
            mask_norm = cv2.GaussianBlur(mask_norm, (0, 0), min(0.45, feather_radius * 0.09))
        
        # 扩展mask到3通道
        mask_3channel = np.stack([mask_norm] * 3, axis=-1)
        
        # 提取前景（人像）
        if image.shape[2] == 4:
            # 如果有Alpha通道，使用它
            foreground = image[:, :, :3]
        else:
            foreground = image
        
        result = (foreground * mask_3channel + background * (1 - mask_3channel)).astype(np.uint8)
        
        return result
    
    def auto_remove_and_replace(self, image, bg_color='blue', method='humanseg', feather_radius=5, rect=None):
        if image is None:
            return None

        # ① PP-HumanSeg 深度学习分割（主路径）
        mask = self._humanseg_predict(image)

        # ② 回退：GrabCut
        if mask is None:
            enhanced = cv2.convertScaleAbs(image, alpha=1.4, beta=8)
            _, mask = self.remove_background_grabcut(enhanced, rect=rect)

        # ③ 回退：颜色分割
        if mask is None or np.count_nonzero(mask) < 0.01 * mask.size:
            _, mask = self.remove_background_color(image)

        if mask is None or np.count_nonzero(mask) < 0.01 * mask.size:
            return image

        mask = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)

        # ④ 后处理
        mask = self._polish_mask(mask)

        # ⑤ 肖像模型（398x224）可能漏掉肩膀，用人物框做几何填充
        if self._humanseg_input_size == (398, 224) and rect is not None:
            mask = self._fill_body_region(mask, rect)

        result = self.replace_background(image, mask, bg_color, feather_radius=feather_radius)
        return result

    def _fill_body_region(self, mask, rect):
        """
        用人物框创建身体区域掩码，与人像分割掩码合并。
        PP-HumanSeg 是面部人像模型，半身照中肩膀可能漏掉。
        策略：从掩码底部向下膨胀，约束在椭圆身体区域内。
        """
        h, w = mask.shape
        rx, ry, rw, rh = rect
        rx = max(0, rx)
        ry = max(0, ry)
        rw = min(rw, w - rx)
        rh = min(rh, h - ry)

        if rw < 30 or rh < 30:
            return mask

        # 椭圆身体区域（限宽限高，不会填到背景）
        body_ellipse = np.zeros((h, w), dtype=np.uint8)
        center = (rx + rw // 2, ry + rh // 2)
        axes = (rw // 2, rh // 2)
        cv2.ellipse(body_ellipse, center, axes, 0, 0, 360, 255, -1)

        # 从掩码底部向下做纵向膨胀（只在椭圆区域内）
        ys, xs = np.where(mask > 127)
        if len(ys) < 100:
            return mask

        bottom_y = int(np.max(ys))
        # 在掩码最低行找到前景的左右边界
        bottom_row = mask[bottom_y, :]
        fg_cols = np.where(bottom_row > 127)[0]
        if len(fg_cols) < 2:
            return mask
        x_min = int(fg_cols[0])
        x_max = int(fg_cols[-1])

        # 纵向膨胀：从掩码底边向下一直到椭圆底边
        for y in range(bottom_y, min(center[1] + axes[1], h)):
            fill = np.zeros((h, w), dtype=np.uint8)
            # 每行比上一行宽 5%
            extra = int((y - bottom_y) * 0.05)
            x1 = max(x_min - extra, 0)
            x2 = min(x_max + extra, w)
            fill[y, x1:x2] = 255
            # 只在椭圆内生效
            mask = cv2.bitwise_or(mask, cv2.bitwise_and(fill, body_ellipse))

        return mask

    def _polish_mask(self, mask):
        """
        Preserve soft HumanSeg alpha while removing isolated mask noise.
        """
        if mask is None or mask.size == 0:
            return mask
        smoothed = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.8)
        return np.clip(smoothed, 0, 255).astype(np.uint8)

    def _fill_edge_background(self, mask):
        """
        从四边向外膨胀背景 → 只保留与边缘连通的背景区域。
        解决 GrabCut/颜色分割遗漏的小块背景斑点"没染上色"的问题。
        """
        h, w = mask.shape
        bg = cv2.bitwise_not(mask)
        k = np.ones((5, 5), np.uint8)
        bg = cv2.dilate(bg, k, iterations=2)
        # 把背景 pad 成 255，从 (0,0) flood-fill 连通背景
        pad = np.full((h + 2, w + 2), 255, dtype=np.uint8)
        pad[1:h+1, 1:w+1] = bg
        cv2.floodFill(pad, None, (0, 0), 127)
        bg = np.where(pad[1:h+1, 1:w+1] == 127, 255, 0).astype(np.uint8)
        return cv2.bitwise_not(bg)

    def _grabcut_smart(self, image, rect=None):
        """
        稳定版 GrabCut：
          1. 至少 5% 边框 = 肯定背景（确保 GC_BGD 有像素）
          2. 中央区 + 肤色 + rect 区 = 肯定/可能前景（确保 GC_FGD 有像素）
          3. 颜色背景检测仅用于扩大 GC_BGD 范围，不影响保证性约束
        """
        h, w = image.shape[:2]
        if h < 10 or w < 10:
            return None

        gc_mask = np.ones((h, w), np.uint8) * cv2.GC_PR_BGD

        # ① 保证性 GC_BGD：至少 5% 宽度的边框 = 肯定背景
        b = max(20, int(min(h, w) * 0.05))
        gc_mask[0:b, :] = cv2.GC_BGD
        gc_mask[h-b:h, :] = cv2.GC_BGD
        gc_mask[:, 0:b] = cv2.GC_BGD
        gc_mask[:, w-b:w] = cv2.GC_BGD

        # ② 扩大 GC_BGD：颜色检测到的真实背景像素
        bg_color = self._detect_background_region(image)
        if bg_color is not None and np.count_nonzero(bg_color) > 0.01 * bg_color.size:
            gc_mask[bg_color > 128] = cv2.GC_BGD

        # ③ 保证性 GC_FGD：中央 25% + 肤色
        cx, cy = w // 2, h // 2
        cw, ch = max(30, int(w * 0.25)), max(30, int(h * 0.25))
        x1 = max(b, cx - cw // 2)
        y1 = max(b, cy - ch // 2)
        x2 = min(w - b, x1 + cw)
        y2 = min(h - b, y1 + ch)
        if x2 > x1 and y2 > y1:
            gc_mask[y1:y2, x1:x2] = cv2.GC_FGD

        skin = self._get_skin_mask(image)
        if skin is not None and np.count_nonzero(skin) > 0.005 * skin.size:
            k = np.ones((5, 5), np.uint8)
            skin = cv2.dilate(skin, k, iterations=2)
            gc_mask[skin > 128] = cv2.GC_FGD

        # ④ rect 扩展可能前景
        if rect is not None and len(rect) == 4:
            rx, ry, rw, rh = rect
            rx = max(0, rx)
            ry = max(0, ry)
            rw = min(rw, w - rx)
            rh = min(rh, h - ry)
            if rw > 10 and rh > 10:
                gc_mask[ry:ry+rh, rx:rx+rw] = cv2.GC_PR_FGD

        # ⑤ 最终校验：至少有一个 BGD 和 一个 FGD
        if np.count_nonzero(gc_mask == cv2.GC_BGD) == 0 or np.count_nonzero(gc_mask == cv2.GC_FGD) == 0:
            return None

        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(image, gc_mask, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
        except Exception as e:
            print(f"[BG] GrabCut 异常: {e}")
            return None

        result_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
        result_mask = result_mask * 255
        return result_mask

    def _detect_background_region(self, image):
        """
        用颜色检测背景区域：采样四角+四边 → 找主背景色 → 标记匹配像素
        返回 0-255 掩码（255=背景）。如果背景太接近肤色则返回 None。
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        border = max(2, int(min(h, w) * 0.06))
        step = max(1, min(w, h) // 30)

        # 向量化采样四条边
        top_strip = hsv[0:border, ::step].reshape(-1, 3)
        bot_strip = hsv[h-border:h, ::step].reshape(-1, 3)
        left_strip = hsv[border:h-border, 0:border].reshape(-1, 3)
        right_strip = hsv[border:h-border, w-border:w].reshape(-1, 3)

        pixels = np.vstack([top_strip, bot_strip, left_strip, right_strip])
        if len(pixels) < 30:
            return None

        mh, ms, mv = np.median(pixels[:, 0]), np.median(pixels[:, 1]), np.median(pixels[:, 2])

        if 0 <= mh <= 25 and 20 <= ms <= 80 and mv > 80:
            return None

        low = np.array([max(0, mh - 12), max(0, ms - 30), max(0, mv - 40)])
        high = np.array([min(180, mh + 12), min(255, ms + 30), min(255, mv + 40)])
        bg = cv2.inRange(hsv, low, high)

        if np.count_nonzero(bg) < 0.03 * bg.size:
            return None
        return bg

    def _get_skin_mask(self, image):
        """HSV 肤色检测，返回 0-255 掩码（255=肤色）"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 东亚人肤色范围
        lower = np.array([0, 20, 50], dtype=np.uint8)
        upper = np.array([25, 170, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        return mask

    def _color_segment_smart(self, image):
        """
        智能颜色分割：从四角 flood-fill 标记背景 → 剩余=前景
        比颜色范围法更稳定，能处理背景渐变色和白衣白底场景。
        """
        h, w = image.shape[:2]

        # ① 肤色掩码：肤色绝不可能是背景
        skin = self._get_skin_mask(image)

        # ② 灰度图用于 flood fill 的容差控制
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ③ 标记遮罩
        marker = np.zeros((h, w), np.uint8)

        # 四边 8% 区域涂黑 = 肯定背景种子
        b = max(3, int(min(h, w) * 0.08))
        marker[0:b, :] = 255
        marker[h-b:h, :] = 255
        marker[:, 0:b] = 255
        marker[:, w-b:w] = 255

        # 肤色区域保护 = 不能涂
        if skin is not None:
            marker[skin > 128] = 0

        # ④ 从所有角落 flood-fill 背景
        corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
        for cy, cx in corners:
            if marker[cy, cx] == 255:
                try:
                    cv2.floodFill(gray, mask=None, loDiff=40, upDiff=40, seedPoint=(cx, cy), flags=8)
                except Exception:
                    pass

        # 用 flood fill 后的灰度图来判断：已填充区域=背景
        bg = np.zeros((h, w), np.uint8)
        # 取四角颜色，把灰度值接近四角中位数的像素标记为背景
        corner_vals = []
        for cy, cx in corners:
            corner_vals.append(gray[cy, cx])
        bg_gray = int(np.median(corner_vals))

        bg = cv2.inRange(gray, max(0, bg_gray - 45), min(255, bg_gray + 45))

        # 肤色保护
        if skin is not None:
            bg[skin > 64] = 0

        fg = cv2.bitwise_not(bg)

        # ⑤ 形态学：闭合内部空隙，开放外部噪点
        k = np.ones((7, 7), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)

        # ⑥ 保留中央大连通域，去掉边缘背景碎片
        center_y, center_x = h // 2, w // 2
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if num_labels > 1:
            best_label = 1
            best_dist = float('inf')
            for i in range(1, num_labels):
                c_cx, c_cy = centroids[i]
                area = stats[i, cv2.CC_STAT_AREA]
                dist = (c_cx - center_x)**2 + (c_cy - center_y)**2
                # 靠近中心且面积不小的优先
                if area > 0.05 * fg.size:
                    if dist < best_dist:
                        best_dist = dist
                        best_label = i
            fg = np.where(labels == best_label, 255, 0).astype('uint8')

        return fg

    def _deep_clean_mask(self, mask):
        """
        深度清理掩码：消除噪点、只保留最大前景区域、平滑边缘
        """
        if mask is None:
            return None

        mask = mask.copy()
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # 1) 中值滤波去掉椒盐噪声（人脸蓝色斑点类）
        k = 7 if max(mask.shape) > 500 else 3
        mask = cv2.medianBlur(mask, k)

        # 2) 二值化回 0/255
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 3) 保留最大连通区域（人像）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            clean = np.zeros_like(mask)
            cv2.drawContours(clean, [largest], -1, 255, cv2.FILLED)
            mask = clean

        # 4) 形态学闭合 + 开放填补边缘空隙
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 5) 轻微高斯模糊让边缘自然
        mask = cv2.GaussianBlur(mask, (7, 7), 2)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return mask
    
    def refine_mask(self, mask, min_area=1000, smooth=True):
        """
        优化分割掩码
        
        Args:
            mask: 原始掩码
            min_area: 最小区域面积
            smooth: 是否进行平滑处理
            
        Returns:
            numpy数组: 优化后的掩码
        """
        if mask is None:
            return None
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小区域
        filtered_contours = [cnt for cnt in contours 
                            if cv2.contourArea(cnt) > min_area]
        
        # 创建新的掩码
        refined_mask = np.zeros_like(mask)
        cv2.drawContours(refined_mask, filtered_contours, -1, 255, -1)
        
        # 平滑处理
        if smooth:
            kernel = np.ones((5, 5), np.uint8)
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
            refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)
        
        return refined_mask


# 测试代码
if __name__ == "__main__":
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    remover = BackgroundRemover()
    
    # 测试背景替换
    result = remover.auto_remove_and_replace(test_image, 'blue', 'color')
    print(f"Result shape: {result.shape if result is not None else 'None'}")
