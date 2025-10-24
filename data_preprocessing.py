# data_preprocessing.py
import os
import shutil
from sklearn.model_selection import train_test_split
import random

def prepare_yolo_dataset(data_dir="osu_dataset", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """准备YOLO格式的数据集"""
    
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "annotated")
    
    # 创建目录结构
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(data_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'labels', split), exist_ok=True)
    
    # 获取所有标注文件
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    image_files = [f.replace('.txt', '.jpg') for f in label_files]
    
    # 确保图像文件存在
    valid_pairs = []
    for img_file, label_file in zip(image_files, label_files):
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, label_file)
        
        if os.path.exists(img_path) and os.path.exists(label_path):
            valid_pairs.append((img_file, label_file))
    
    # 数据集划分
    train_pairs, temp_pairs = train_test_split(valid_pairs, train_size=train_ratio, random_state=42)
    val_pairs, test_pairs = train_test_split(temp_pairs, train_size=val_ratio/(val_ratio+test_ratio), random_state=42)
    
    # 复制文件到相应目录
    splits = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }
    
    for split_name, pairs in splits.items():
        print(f"处理 {split_name} 集: {len(pairs)} 个样本")
        
        for img_file, label_file in pairs:
            # 复制图像
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(data_dir, 'images', split_name, img_file)
            shutil.copy2(src_img, dst_img)
            
            # 复制标签
            src_label = os.path.join(labels_dir, label_file)
            dst_label = os.path.join(data_dir, 'labels', split_name, label_file)
            shutil.copy2(src_label, dst_label)
    
    print("数据集准备完成！")
    print(f"训练集: {len(train_pairs)}")
    print(f"验证集: {len(val_pairs)}")  
    print(f"测试集: {len(test_pairs)}")

if __name__ == "__main__":
    prepare_yolo_dataset()