# data_collection.py
import cv2
import numpy as np
from mss import mss
import os
import json
import time
from datetime import datetime
import pyautogui

class OSUDataCollector:
    def __init__(self, data_dir="osu_dataset"):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.labels_dir = os.path.join(data_dir, "labels")
        self.metadata_file = os.path.join(data_dir, "metadata.json")
        
        # 创建目录
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        # 截图区域
        self.monitor = {'top': 45, 'left': 1291, 'width': 1258, 'height': 708}
        self.sct = mss()
        
        # 数据统计
        self.collection_stats = {
            'total_frames': 0,
            'hit_circles': 0,
            'sliders': 0,
            'spinners': 0,
            'start_time': time.time()
        }
        
    def collect_data(self, duration_minutes=60, frames_per_second=2):
        """收集游戏数据"""
        print(f"开始收集数据，持续时间: {duration_minutes}分钟")
        
        end_time = time.time() + duration_minutes * 60
        frame_interval = 1.0 / frames_per_second
        
        while time.time() < end_time:
            frame_start = time.time()
            
            # 截图
            screenshot = self.sct.grab(self.monitor)
            img = np.array(screenshot)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_filename = f"frame_{timestamp}.jpg"
            label_filename = f"frame_{timestamp}.json"
            
            # 保存图像
            image_path = os.path.join(self.images_dir, image_filename)
            cv2.imwrite(image_path, img_bgr)
            
            # 使用现有检测代码生成初步标签（后续需要手动修正）
            preliminary_labels = self.generate_preliminary_labels(img_bgr)
            
            # 保存标签
            label_path = os.path.join(self.labels_dir, label_filename)
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(preliminary_labels, f, ensure_ascii=False, indent=2)
            
            # 更新统计
            self.collection_stats['total_frames'] += 1
            self.collection_stats['hit_circles'] += len(preliminary_labels.get('hit_circles', []))
            self.collection_stats['sliders'] += len(preliminary_labels.get('sliders', []))
            
            # 显示进度
            if self.collection_stats['total_frames'] % 10 == 0:
                self.print_progress()
            
            # 控制采集频率
            processing_time = time.time() - frame_start
            sleep_time = max(0, frame_interval - processing_time)
            time.sleep(sleep_time)
        
        self.save_metadata()
        print("数据收集完成！")
    
    def generate_preliminary_labels(self, img_bgr):
        """使用现有检测代码生成初步标签"""
        # 这里集成你现有的检测代码

        
        # 返回格式：

        labels = {
            'hit_circles': [],  # [{x, y, radius, timestamp}]
            'sliders': [],      # [{x, y, radius, points, timestamp}]
            'spinners': [],     # [{x, y, radius, timestamp}]
            'frame_timestamp': time.time()
        }
        
        # 示例：调用你现有的检测函数
        # all_objects = your_existing_detection_function(img_bgr)
        # for obj in all_objects:
        #     center, radius, area, obj_type = obj
        #     if obj_type == 'hit_circle':
        #         labels['hit_circles'].append({
        #             'x': center[0], 'y': center[1], 'radius': radius
        #         })
        
        return labels
    
    def print_progress(self):
        """打印收集进度"""
        elapsed = time.time() - self.collection_stats['start_time']
        fps = self.collection_stats['total_frames'] / elapsed
        
        print(f"已收集: {self.collection_stats['total_frames']} 帧 | "
              f"点击圈: {self.collection_stats['hit_circles']} | "
              f"滑条: {self.collection_stats['sliders']} | "
              f"FPS: {fps:.2f}")
    
    def save_metadata(self):
        """保存元数据"""
        metadata = {
            'collection_date': datetime.now().isoformat(),
            'monitor_region': self.monitor,
            'total_frames': self.collection_stats['total_frames'],
            'total_hit_circles': self.collection_stats['hit_circles'],
            'total_sliders': self.collection_stats['sliders'],
            'duration_seconds': time.time() - self.collection_stats['start_time']
        }
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
# 安全设置
pyautogui.FAILSAFE = True

# 颜色范围（保持你的现有设置）
LOWER_blue = np.array([100, 0, 50])   
UPPER_blue = np.array([113, 199, 245])   

LOWER_green = np.array([89, 0, 50])  
UPPER_green = np.array([94, 196, 255]) 

LOWER_red = np.array([149, 80, 50])       
UPPER_red = np.array([155, 92, 255])     

LOWER_yello = np.array([20, 110, 87])     
UPPER_yello = np.array([24, 148, 255])    

LOWER_purple = np.array([130, 70, 54])   
UPPER_purple = np.array([137, 255, 255])  

LOWER_white = np.array([0, 0, 194])      
UPPER_white = np.array([0, 0, 255])

LOWER_all = np.array([0, 0, 185])      
UPPER_all = np.array([179, 255, 190])       

monitor = {'top': -30, 'left': 1300, 'width': 1260, 'height': 800}

def improved_find_circles(img_bgr, lower_color, upper_color):
    """改进的圆环检测方法，专门解决重叠和误判问题"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # 更精细的形态学操作
    kernel_small = np.ones((2, 2), np.uint8)
    kernel_medium = np.ones((3, 3), np.uint8)
    
    # 先开运算去除噪声，再闭运算连接断点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)
    
    # 多尺度轮廓检测 - 关键改进！
    circles = []
    
    # 方法1: 直接轮廓检测（处理明显分离的圆环）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        result = analyze_contour(contour, img_bgr, mask)
        if result:
            circles.append(result)
    
    # 方法2: 距离变换 + 多阈值（处理轻微重叠）
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # 尝试多个阈值来分离不同重叠程度的圆环
    thresholds = [0.3, 0.5, 0.7]
    for thresh_ratio in thresholds:
        _, sure_fg = cv2.threshold(dist_transform, thresh_ratio * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        sub_contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sub_contours:
            result = analyze_contour(contour, img_bgr, mask)
            if result and result not in circles:  # 避免重复
                circles.append(result)
    
    return circles, mask

def analyze_contour(contour, img_bgr, mask):
    """分析单个轮廓，判断是否为有效的游戏元素"""
    area = cv2.contourArea(contour)
    if area < 137:  # 适当降低面积阈值，捕捉更多圆环
        return None
        
    # 圆形度检查
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None
        
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if circularity < 0.4:  # 降低圆形度要求，适应变形圆环
        return None
    
    # 获取最小外接圆
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    
    if radius < 30 or radius > 100:  # 合理的半径范围
        return None
    
    # 改进的实心圆检测
    min_circle_area = 3.14159 * radius * radius
    area_ratio = area / min_circle_area if min_circle_area > 0 else 0
    
    # 检查中心区域
    roi_radius = max(radius // 3, 8)  # 增大检查区域
    roi_mask = np.zeros_like(mask)
    cv2.circle(roi_mask, center, roi_radius, 255, -1)
    
    roi_area = cv2.bitwise_and(mask, roi_mask)
    white_pixels = cv2.countNonZero(roi_area)
    total_pixels = 3.14159 * roi_radius * roi_radius
    center_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
    
    # 更严格的实心圆判断条件
    is_solid = (area_ratio > 0.6 and center_ratio > 0.8)
    if is_solid:
        return None
    
    # 改进的对象类型判断 - 多重条件！
    object_type = classify_object_type(contour, center, radius, area, mask)
    
    return (center, radius, area, object_type)

def classify_object_type(contour, center, radius, area, mask):
    """使用多重条件准确分类点击圈和滑条"""
    
    # 条件1: 半径范围（主要条件）
    if radius > 55:  # 提高滑条半径阈值
        slider_score = 1
    elif radius > 45:
        slider_score = 0.5
    else:
        slider_score = 0
    
    # 条件2: 面积与理想圆面积的比例（滑条通常更"瘦"）
    ideal_area = 3.14159 * radius * radius
    area_ratio = area / ideal_area if ideal_area > 0 else 0
    if area_ratio < 0.4:
        slider_score += 0.5  # 面积比例小的可能是滑条环
    
    # 条件3: 轮廓复杂度（滑条可能有更复杂的结构）
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        solidity = area / hull_area
        if solidity < 0.8:  # 实心度低的可能是滑条
            slider_score += 0.4
    
    # 条件4: 检查周围区域是否有滑条特征
    slider_feature_score = check_slider_features(center, radius, mask)
    slider_score += slider_feature_score
    
    # 最终判断
    return 'slider' if slider_score >= 1.2 else 'hit_circle'

def check_slider_features(center, radius, mask):
    """检查滑条特有的视觉特征"""
    score = 0
    
    # 检查是否有同心圆结构（滑条特征）
    larger_radius = int(radius * 1.3)
    larger_mask = np.zeros_like(mask)
    cv2.circle(larger_mask, center, larger_radius, 255, 4)  # 画一个稍大的圆环
    
    # 检查大圆环上是否有像素
    ring_pixels = cv2.bitwise_and(mask, larger_mask)
    ring_pixel_count = cv2.countNonZero(ring_pixels)
    
    if ring_pixel_count > 50:  # 如果外环有足够像素，可能是滑条
        score += 0.4
    
    return score

def remove_duplicate_detections(circles, distance_threshold=20):
    """移除重复的检测结果"""
    unique_circles = []
    
    for circle in circles:
        center, radius, area, obj_type = circle
        is_duplicate = False
        
        for unique_circle in unique_circles:
            u_center, u_radius, u_area, u_type = unique_circle
            # 计算圆心距离
            distance = np.sqrt((center[0]-u_center[0])**2 + (center[1]-u_center[1])**2)
            
            if distance < distance_threshold:
                # 保留半径更大的检测结果（通常更准确）
                if radius > u_radius:
                    unique_circles.remove(unique_circle)
                    unique_circles.append(circle)
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_circles.append(circle)
    
    return unique_circles

class ImprovedOSUAIPlayer:
    def __init__(self):
        self.click_cooldown = 0.3  # 略微降低冷却时间
        self.last_click_time = 0
        self.active_sliders = []
        self.slider_start_time = {}
        
    def should_click(self, current_time):
        return current_time - self.last_click_time >= self.click_cooldown
    
    def process_slider(self, slider_info, current_time):
        center, radius, area, _ = slider_info
        absolute_x = monitor['left'] + center[0]
        absolute_y = monitor['top'] + center[1]
        
        slider_id = f"{center[0]}_{center[1]}"
        
        if slider_id not in self.active_sliders:
            # pyautogui.mouseDown(absolute_x, absolute_y)
            self.active_sliders.append(slider_id)
            self.slider_start_time[slider_id] = current_time
            print(f"开始滑条: ({absolute_x}, {absolute_y}), 半径: {radius}")
        
        # 简单的滑条持续时间逻辑（需要根据音乐节奏改进）
        slider_duration = 0.2  # 默认0.5秒
        if current_time - self.slider_start_time[slider_id] > slider_duration:
            # pyautogui.mouseUp(absolute_x, absolute_y)
            self.active_sliders.remove(slider_id)
            print(f"结束滑条: ({absolute_x}, {absolute_y})")
    
    def process_hit_circle(self, circle_info, current_time):
        if not self.should_click(current_time):
            return
            
        center, radius, area, _ = circle_info
        absolute_x = monitor['left'] + center[0]
        absolute_y = monitor['top'] + center[1]
        
        # 改进的时机判断：基于半径和游戏节奏
        if 12 < radius < 50:  # 扩大可点击的半径范围
            # pyautogui.click(absolute_x, absolute_y)
            self.last_click_time = current_time
            # print(f"点击: ({absolute_x}, {absolute_y}), 半径: {radius}")

def main():
    ai_player = ImprovedOSUAIPlayer()
    
    with mss() as sct:
        while True:
            current_time = time.time()


            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # 检测所有颜色的对象
            all_objects = []
            combined_mask = np.zeros((monitor['height'], monitor['width']), dtype=np.uint8)
            
            # 定义颜色检测序列
            color_ranges = [                (LOWER_blue, UPPER_blue),
                (LOWER_green, UPPER_green),
                (LOWER_red, UPPER_red),
                (LOWER_yello, UPPER_yello),
                (LOWER_purple, UPPER_purple),
                (LOWER_white, UPPER_white),
                (LOWER_all, UPPER_all)
            ]
            
            # 并行检测所有颜色
            for lower, upper in color_ranges:
                circles, mask = improved_find_circles(img_bgr, lower, upper)
                all_objects.extend(circles)
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # 移除重复检测
            all_objects = remove_duplicate_detections(all_objects)
            
            # 分离点击圈和滑条
            hit_circles = [obj for obj in all_objects if obj[3] == 'hit_circle']
            sliders = [obj for obj in all_objects if obj[3] == 'slider']
            
            # 可视化
            for obj in all_objects:
                center, radius, area, obj_type = obj
                
                if obj_type == 'slider':
                    cv2.circle(img_bgr, center, radius, (255, 0, 0), 5)  # 蓝色：滑条
                    cv2.putText(img_bgr, f"Slider R:{radius}", 
                               (center[0]-40, center[1]-radius-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
                else:
                    cv2.circle(img_bgr, center, radius, (0, 255, 0), 2)  # 绿色：点击圈
                    cv2.putText(img_bgr, f"Hit R:{radius}", 
                               (center[0]-30, center[1]-radius-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                cv2.circle(img_bgr, center, 2, (0, 0, 255), 4)  # 红色中心点
            
            # 处理逻辑：先处理滑条，再处理点击圈
            for slider in sliders:
                ai_player.process_slider(slider, current_time)
            
            # 点击圈按半径排序（小的优先，更紧急）
            hit_circles.sort(key=lambda x: x[1])
            for circle in hit_circles:
                ai_player.process_hit_circle(circle, current_time)
            
            # 显示调试信息
            cv2.putText(img_bgr, f"Hit: {len(hit_circles)}, Slider: {len(sliders)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
            
            cv2.imshow('Color Mask', combined_mask)
            cv2.imshow('OSU AI Detection', img_bgr)
            
            # 性能信息
            processing_time = time.time() - current_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            # print(f"FPS: {fps:.1f}, 点击圈: {len(hit_circles)}, 滑条: {len(sliders)}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # 清理：释放所有鼠标按键
                pyautogui.mouseUp()
                break
                
    cv2.destroyAllWindows()
# 使用方法
if __name__ == "__main__":
    collector = OSUDataCollector()
    collector.collect_data(duration_minutes=30, frames_per_second=2)  # 30分钟数据