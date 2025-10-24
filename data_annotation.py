# data_annotation.py
import cv2
import numpy as np
import os
import json
from pathlib import Path
import pickle

class OSUAnnotationTool:
    def __init__(self, data_dir="osu_dataset"):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.labels_dir = os.path.join(data_dir, "labels")
        self.annotated_dir = os.path.join(data_dir, "annotated")
        self.state_file = os.path.join(data_dir, "annotation_state.pkl")
        
        os.makedirs(self.annotated_dir, exist_ok=True)
        
        # 获取所有图像文件
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.jpg')])
        
        # 从上次标注的位置开始
        self.current_index = self.get_last_annotation_index()
        
        # 标注状态
        self.current_labels = []
        self.history_stack = []  # 撤销历史
        
        # 对象类型定义
        self.object_types = {
            'hit_circle': 0,
            'slider_head': 1, 
            'slider_tail': 2,
            'slider_tick': 3,
            'spinner': 4
        }
        
        self.current_object_type = 'hit_circle'
        self.drawing = False
        self.start_point = None
        self.temp_bbox = None
        
        # 颜色定义
        self.colors = {
            'hit_circle': (0, 255, 0),      # 绿色
            'slider_head': (255, 0, 0),     # 蓝色
            'slider_tail': (0, 0, 255),     # 红色
            'slider_tick': (255, 255, 0),   # 青色
            'spinner': (255, 0, 255)        # 紫色
        }
        
        # 加载当前图像的标注
        self.load_current_annotations()
    
    def get_last_annotation_index(self):
        """获取上次标注的图片索引"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                    return state.get('current_index', 0)
            except:
                pass
        return 0
    
    def save_annotation_state(self):
        """保存当前标注状态"""
        state = {'current_index': self.current_index}
        with open(self.state_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load_current_annotations(self):
        """加载当前图像的标注"""
        if self.current_index < len(self.image_files):
            label_filename = self.image_files[self.current_index].replace('.jpg', '.txt')
            label_path = os.path.join(self.annotated_dir, label_filename)
            
            self.current_labels = []
            if os.path.exists(label_path):
                image_path = os.path.join(self.images_dir, self.image_files[self.current_index])
                image = cv2.imread(image_path)
                h, w = image.shape[:2]
                
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, x_center, y_center, width, height = parts
                            
                            # 转换回绝对坐标
                            x_center_abs = float(x_center) * w
                            y_center_abs = float(y_center) * h
                            width_abs = float(width) * w
                            height_abs = float(height) * h
                            
                            x = int(x_center_abs - width_abs / 2)
                            y = int(y_center_abs - height_abs / 2)
                            width_int = int(width_abs)
                            height_int = int(height_abs)
                            
                            # 获取对象类型
                            obj_type = self.get_object_type(int(class_id))
                            self.current_labels.append((x, y, width_int, height_int, obj_type))
    
    def get_object_type(self, class_id):
        """根据类别ID获取对象类型"""
        type_map = {0: 'hit_circle', 1: 'slider_head', 2: 'slider_tail', 3: 'slider_tick', 4: 'spinner'}
        return type_map.get(class_id, 'hit_circle')
    
    def push_to_history(self):
        """将当前状态保存到历史记录"""
        self.history_stack.append(self.current_labels.copy())
        # 限制历史记录数量
        if len(self.history_stack) > 10:
            self.history_stack.pop(0)
    
    def undo_last_annotation(self):
        """撤销上一次标注"""
        if self.history_stack:
            self.current_labels = self.history_stack.pop()
    
    def run(self):
        """运行标注工具"""
        if not self.image_files:
            print("没有找到图像文件！")
            return
        
        cv2.namedWindow('OSU Annotation Tool')
        cv2.setMouseCallback('OSU Annotation Tool', self.mouse_callback)
        
        print("标注工具使用说明:")
        print("左键拖动 - 绘制边界框")
        print("1-5 - 切换对象类型")
        print("n - 下一张图像")
        print("p - 上一张图像") 
        print("z - 撤销")
        print("d - 删除当前图片和标签")  # 新增这一行
        print("q - 退出")
        while True:
            if self.current_index < len(self.image_files):
                self.load_current_image()
            else:
                print("所有图片都已标注完成！")
                break
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                self.save_annotations()
                self.save_annotation_state()
                self.current_index += 1
                self.current_labels = []
                self.load_current_annotations()
            elif key == ord('p'):
                self.save_annotations()
                self.save_annotation_state()
                self.current_index = max(0, self.current_index - 1)
                self.current_labels = []
                self.load_current_annotations()
            elif key == ord('z'):  # Ctrl+Z
                self.undo_last_annotation()
            elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                # 数字键切换对象类型
                type_map = {
                    ord('1'): 'hit_circle',
                    ord('2'): 'slider_head', 
                    ord('3'): 'slider_tail',
                    ord('4'): 'slider_tick',
                    ord('5'): 'spinner'
                }
                self.current_object_type = type_map.get(key, 'hit_circle')
    
    def load_current_image(self):
        """加载当前图像并显示标注"""
        image_path = os.path.join(self.images_dir, self.image_files[self.current_index])
        image = cv2.imread(image_path)
        
        # 显示现有标注
        for label in self.current_labels:
            x, y, w, h, obj_type = label
            color = self.colors.get(obj_type, (255, 255, 255))
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, obj_type, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 显示临时边界框（正在绘制的）
        if self.temp_bbox:
            start, end = self.temp_bbox
            color = self.colors.get(self.current_object_type, (255, 255, 255))
            cv2.rectangle(image, start, end, color, 2)
        
        # 显示简洁信息
        info_text = f"{self.current_index+1}/{len(self.image_files)} | {self.current_object_type} | {len(self.current_labels)}"
        cv2.putText(image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('OSU Annotation Tool', image)
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.temp_bbox = None
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                end_point = (x, y)
                self.temp_bbox = (self.start_point, end_point)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_point = (x, y)
            # 保存当前状态到历史
            self.push_to_history()
            
            # 计算边界框
            x1, y1 = self.start_point
            x2, y2 = end_point
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            width = x_max - x_min
            height = y_max - y_min
            
            # 添加到标注
            if width > 10 and height > 10:  # 最小尺寸过滤
                self.current_labels.append((x_min, y_min, width, height, self.current_object_type))
            
            self.temp_bbox = None
    
    def save_annotations(self):
        """保存标注到文件"""
        # 转换为YOLO格式
        yolo_labels = []
        image_path = os.path.join(self.images_dir, self.image_files[self.current_index])
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        for x, y, width, height, obj_type in self.current_labels:
            # 转换为相对坐标
            x_center = (x + width / 2) / w
            y_center = (y + height / 2) / h
            width_rel = width / w
            height_rel = height / h
            
            # 类别ID
            class_id = self.object_types[obj_type]
            
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width_rel:.6f} {height_rel:.6f}")
        
        # 保存YOLO格式标签
        label_filename = self.image_files[self.current_index].replace('.jpg', '.txt')
        label_path = os.path.join(self.annotated_dir, label_filename)
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_labels))
    def delete_current_image_and_label(self):
        """删除当前图片和对应的标签文件"""
        if self.current_index < len(self.image_files):
            current_image = self.image_files[self.current_index]
            
            # 删除图片文件
            image_path = os.path.join(self.images_dir, current_image)
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"已删除图片: {current_image}")
            
            # 删除标签文件
            label_filename = current_image.replace('.jpg', '.txt')
            label_path = os.path.join(self.annotated_dir, label_filename)
            if os.path.exists(label_path):
                os.remove(label_path)
                print(f"已删除标签: {label_filename}")
            
            # 从文件列表中移除
            self.image_files.pop(self.current_index)
            
            # 调整索引
            if self.current_index >= len(self.image_files):
                self.current_index = max(0, len(self.image_files) - 1)
            
            # 重新加载当前标注
            if self.image_files:
                self.current_labels = []
                self.load_current_annotations()
            else:
                print("所有图片已删除")

# 运行标注工具
if __name__ == "__main__":
    tool = OSUAnnotationTool()
    tool.run()