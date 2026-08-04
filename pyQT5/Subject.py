import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class RegionAnalyzer:
    def __init__(self, image_path):
        self.original_image = self._load_image(image_path)
        self.processed_image = None
        self.labels = None
        self.regions = None
        self.calibration_regions = None
        self.cup_regions = None
        
    def _load_image(self, path):
        # 如果输入已经是numpy数组（直接传入图像数据）
        if isinstance(path, np.ndarray):
            img = path
        # 如果输入是文件路径字符串
        elif isinstance(path, str):
            img = plt.imread(path)
        else:
            raise ValueError("输入必须是图像路径或numpy数组")

        # 统一处理数组维度
        if img.ndim == 3:
            return np.mean(img, axis=2).astype(np.uint8)
        return img.astype(np.uint8)
    
    def apply_morphology(self, operations):
        """
        应用形态学操作序列
        :param operations: 操作列表，每个元素为元组 (operation, params)
        """
        self.processed_image = self.original_image.copy()
        for op in operations:
            if op[0] == 'close':
                self.processed_image = self._closed(self.processed_image, *op[1])
            elif op[0] == 'open':
                self.processed_image = self._opend(self.processed_image, *op[1])
            elif op[0] == 'hole_fill':
                self.processed_image = self.hole_filling(self.processed_image)
                
    def analyze_regions(self):
        """执行完整的区域分析流程"""
        if self.processed_image is None:
            raise ValueError("请先进行图像预处理")
            
        self.labels = self.connected_components_labeling(self.processed_image)
        self.regions = self.compute_region_properties(self.labels)
        self.calibration_regions, self.cup_regions = self.filter_regions(self.regions)
    
    @staticmethod
    def create_mask(regions, labels):
        mask = np.zeros_like(labels, dtype=np.uint8)
        for label in regions:
            mask[labels == label] = 255
        return mask
    
    @staticmethod
    def filter_regions(regions):
        calibration_regions = {}
        cup_regions = {}
        
        for label, reg in regions.items():
            if 0.8 < reg['rectangularity'] <= 1.0 and 1000 <= reg['area']:
                # calibration_regions[label] = reg
                cup_regions[label] = reg
            elif reg['rectangularity'] > 0.7 and 1000 <= reg['area']:
                # cup_regions[label] = reg
                calibration_regions[label] = reg
                
        return calibration_regions, cup_regions
    
    @staticmethod
    def compute_region_properties(labels):
        regions = {}
        height, width = labels.shape
        
        # 收集基础信息
        for y in range(height):
            for x in range(width):
                label = labels[y, x]
                if label > 0:
                    if label not in regions:
                        regions[label] = {
                            'min_x': x,
                            'max_x': x,
                            'min_y': y,
                            'max_y': y,
                            'area': 0,
                            'bbox_area': 0.0,
                            'rectangularity': 0.0,
                            'aspect_ratio': 0.0
                        }
                    reg = regions[label]
                    reg['min_x'] = min(reg['min_x'], x)
                    reg['max_x'] = max(reg['max_x'], x)
                    reg['min_y'] = min(reg['min_y'], y)
                    reg['max_y'] = max(reg['max_y'], y)
                    reg['area'] += 1
        
        # 计算衍生属性
        for label, reg in regions.items():
            w = reg['max_x'] - reg['min_x'] + 1
            h = reg['max_y'] - reg['min_y'] + 1
            reg['bbox_area'] = float(w * h)
            reg['rectangularity'] = reg['area'] / reg['bbox_area']
            reg['aspect_ratio'] = w / h if h != 0 else 0.0
            
        return regions
    
    @staticmethod
    def connected_components_labeling(binary):
        height, width = binary.shape
        labels = np.zeros((height, width), dtype=int)
        current_label = 1
        uf = {}  # 并查集

        def find(x):
            while uf.get(x, x) != x:
                uf[x] = uf.get(uf[x], uf[x])  # 路径压缩
                x = uf[x]
            return x

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                if root_x < root_y:
                    uf[root_y] = root_x
                else:
                    uf[root_x] = root_y

        # 第一遍扫描
        for y in range(height):
            for x in range(width):
                if binary[y, x] == 255:
                    neighbors = []
                    if y > 0 and labels[y-1, x] > 0:
                        neighbors.append(labels[y-1, x])
                    if x > 0 and labels[y, x-1] > 0:
                        neighbors.append(labels[y, x-1])

                    if not neighbors:
                        labels[y, x] = current_label
                        uf[current_label] = current_label
                        current_label += 1
                    else:
                        min_label = min(neighbors)
                        labels[y, x] = min_label
                        for n in neighbors:
                            union(min_label, n)

        # 建立标签映射
        label_map = {}
        new_label = 1
        for old_label in np.unique(labels[labels > 0]):
            root = find(old_label)
            if root not in label_map:
                label_map[root] = new_label
                new_label += 1
            label_map[old_label] = label_map[root]

        # 第二遍扫描
        return np.vectorize(lambda x: label_map.get(x, 0))(labels)
    
    @staticmethod
    def hole_filling(binary_image):
        filled = binary_image.copy()
        visited = np.zeros_like(binary_image, dtype=bool)
        height, width = binary_image.shape
        queue = deque()
        
        # 边界扫描
        for x in range(width):
            if filled[0, x] == 0 and not visited[0, x]:
                queue.append((0, x))
                visited[0, x] = True
            if filled[height-1, x] == 0 and not visited[height-1, x]:
                queue.append((height-1, x))
                visited[height-1, x] = True
        for y in range(height):
            if filled[y, 0] == 0 and not visited[y, 0]:
                queue.append((y, 0))
                visited[y, 0] = True
            if filled[y, width-1] == 0 and not visited[y, width-1]:
                queue.append((y, width-1))
                visited[y, width-1] = True
        
        # 洪水填充
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            y, x = queue.popleft()
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if filled[ny, nx] == 0 and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        
        filled[~visited & (filled == 0)] = 255
        return filled
    
    @staticmethod
    def _create_circular_kernel(radius):
        diameter = 2 * radius + 1
        kernel = np.zeros((diameter, diameter), dtype=bool)
        center = radius
        for y in range(diameter):
            for x in range(diameter):
                if (y - center)**2 + (x - center)**2 <= radius**2:
                    kernel[y, x] = True
        return kernel
    
    def _morph_erode(self, binary, kernel):
        h, w = binary.shape
        k_h, k_w = kernel.shape
        pad_y, pad_x = k_h//2, k_w//2
        eroded = np.zeros_like(binary)
        
        for y in range(pad_y, h - pad_y):
            for x in range(pad_x, w - pad_x):
                roi = binary[y-pad_y:y+pad_y+1, x-pad_x:x+pad_x+1]
                eroded[y, x] = 255 if np.all(roi[kernel]) else 0
        return eroded
    
    def _morph_dilate(self, binary, kernel):
        h, w = binary.shape
        k_h, k_w = kernel.shape
        pad_y, pad_x = k_h//2, k_w//2
        dilated = np.zeros_like(binary)
        
        for y in range(pad_y, h - pad_y):
            for x in range(pad_x, w - pad_x):
                roi = binary[y-pad_y:y+pad_y+1, x-pad_x:x+pad_x+1]
                dilated[y, x] = 255 if np.any(roi[kernel]) else 0
        return dilated
    
    def _closed(self, img, r_size=14, r_times=1):
        kernel = self._create_circular_kernel(r_size)
        processed = img.copy()
        for _ in range(r_times):
            processed = self._morph_dilate(processed, kernel)
            processed = self._morph_erode(processed, kernel)
        return processed
    
    def _opend(self, img, r_size=14, r_times=1):
        kernel = self._create_circular_kernel(r_size)
        processed = img.copy()
        for _ in range(r_times):
            processed = self._morph_erode(processed, kernel)
            processed = self._morph_dilate(processed, kernel)
        return processed
    

    def solve(self, close_size = 0):
        # 定义处理流程：闭运算 → 孔洞填充 → 开运算
        operations = [
            ('close', (close_size, 1)),
            ('hole_fill', ()),
            ('open', (10, 1))
        ]
        
        self.apply_morphology(operations)
        self.analyze_regions()

        #return self.processed_image
        #return self.create_mask(self.calibration_regions, self.labels)
        return np.maximum(self.create_mask(self.cup_regions, self.labels), self.create_mask(self.calibration_regions, self.labels))