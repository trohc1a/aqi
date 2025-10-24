# train_yolo.py
from ultralytics import YOLO
import os
import yaml

def prepare_dataset_config(data_dir="osu_dataset"):
    """准备数据集配置文件"""
    config = {
        'path': os.path.abspath(data_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        
        'nc': 3,  # 类别数量
        'names': ['hit_circle', 'slider', 'spinner']  # 类别名称
    }
    
    config_path = os.path.join(data_dir, 'dataset.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

def train_model():
    """训练YOLO模型"""
    # 准备数据配置
    data_config = prepare_dataset_config()
    
    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 使用nano版本，适合集成显卡
    
    # 训练参数
    training_params = {
        'data': data_config,
        'epochs': 100,
        'imgsz': 640,
        'batch': 8,  # 根据你的GPU内存调整
        'workers': 2,
        'patience': 10,  # 早停
        'lr0': 0.01,     # 初始学习率
        'lrf': 0.01,     # 最终学习率
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'box': 7.5,      # 框损失权重
        'cls': 0.5,      # 分类损失权重
        'dfl': 1.5,      # DFL损失权重
    }
    
    # 开始训练
    results = model.train(**training_params)
    
    # 保存最终模型
    model.save('best_osu_detector.pt')
    
    return model, results

def evaluate_model(model, data_config):
    """评估模型性能"""
    # 在验证集上评估
    metrics = model.val(data=data_config)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return metrics

if __name__ == "__main__":
    # 训练模型
    print("开始训练OSU目标检测模型...")
    model, results = train_model()
    
    # 评估模型
    data_config = prepare_dataset_config()
    metrics = evaluate_model(model, data_config)
    
    print("训练完成！")