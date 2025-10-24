# model_inference.py
import cv2
import numpy as np
from ultralytics import YOLO
from mss import mss
import pyautogui
import time

class TrainedOSUDetector:
    def __init__(self, model_path='best_osu_detector.pt'):
        # 加载训练好的模型
        self.model = YOLO(model_path)
        
        # 游戏区域
        self.monitor = {'top': -30, 'left': 1300, 'width': 1260, 'height': 800}
        self.sct = mss()
        
        # 点击逻辑
        self.last_click_time = 0
        self.click_cooldown = 0.1
        
        # 置信度阈值
        self.confidence_threshold = 0.5
        
    def detect_objects(self, img_bgr):
        """使用训练好的模型检测对象"""
        results = self.model(img_bgr, conf=self.confidence_threshold)
        
        detections = {
            'hit_circles': [],
            'sliders': [],
            'spinners': []
        }
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取检测信息
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # 计算中心点和半径
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    radius = max((x2 - x1), (y2 - y1)) / 2
                    
                    detection = {
                        'center': (int(center_x), int(center_y)),
                        'radius': int(radius),
                        'confidence': float(confidence),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    }
                    
                    # 分类存储
                    if class_id == 0:  # hit_circle
                        detections['hit_circles'].append(detection)
                    elif class_id == 1:  # slider
                        detections['sliders'].append(detection)
                    elif class_id == 2:  # spinner
                        detections['spinners'].append(detection)
        
        return detections
    
    def should_click(self, current_time):
        """检查是否可以点击"""
        return current_time - self.last_click_time >= self.click_cooldown
    
    def process_detections(self, detections, current_time):
        """处理检测结果并执行点击"""
        if not self.should_click(current_time):
            return
        
        # 优先处理点击圈
        hit_circles = detections.get('hit_circles', [])
        if hit_circles:
            # 选择最紧急的点击圈（基于半径和置信度）
            hit_circles.sort(key=lambda x: x['radius'] * x['confidence'])
            target = hit_circles[0]
            
            # 转换为绝对坐标并点击
            absolute_x = self.monitor['left'] + target['center'][0]
            absolute_y = self.monitor['top'] + target['center'][1]
            
            pyautogui.click(absolute_x, absolute_y)
            self.last_click_time = current_time
            print(f"点击: ({absolute_x}, {absolute_y}), 置信度: {target['confidence']:.2f}")
    
    def visualize_detections(self, img_bgr, detections):
        """可视化检测结果"""
        colors = {
            'hit_circles': (0, 255, 0),  # 绿色
            'sliders': (255, 0, 0),      # 蓝色  
            'spinners': (0, 0, 255)      # 红色
        }
        
        for obj_type, objects in detections.items():
            color = colors.get(obj_type, (255, 255, 255))
            
            for obj in objects:
                center = obj['center']
                radius = obj['radius']
                confidence = obj['confidence']
                
                # 绘制圆圈
                cv2.circle(img_bgr, center, radius, color, 2)
                cv2.circle(img_bgr, center, 2, color, 3)
                
                # 绘制置信度
                label = f"{obj_type}: {confidence:.2f}"
                cv2.putText(img_bgr, label, (center[0]-40, center[1]-radius-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img_bgr
    
    def run(self):
        """运行检测循环"""
        print("启动训练模型检测器...")
        
        while True:
            current_time = time.time()
            
            # 截图
            screenshot = self.sct.grab(self.monitor)
            img = np.array(screenshot)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # 检测对象
            detections = self.detect_objects(img_bgr)
            
            # 处理点击逻辑
            self.process_detections(detections, current_time)
            
            # 可视化结果
            visualized_img = self.visualize_detections(img_bgr, detections)
            
            # 显示状态
            status_text = f"点击圈: {len(detections['hit_circles'])} | 滑条: {len(detections['sliders'])}"
            cv2.putText(visualized_img, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Trained OSU Detector', visualized_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = TrainedOSUDetector()
    detector.run()