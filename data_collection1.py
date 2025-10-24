# data_collection.py
import cv2
import numpy as np
from mss import mss
import os
import json
import time
from datetime import datetime

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
        self.monitor = {'top': -30, 'left': 1300, 'width': 1260, 'height': 800}
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

# 使用方法
if __name__ == "__main__":
    collector = OSUDataCollector()
    collector.collect_data(duration_minutes=30, frames_per_second=2)  # 30分钟数据