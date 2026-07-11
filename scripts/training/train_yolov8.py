"""
YOLOv8 训练脚本 — 检测脓毒症阳性样本中的异常细胞
"""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ultralytics import YOLO
import torch

PROJECT_ROOT = Path("/home/hyl/project/sepsis_yolo")
DATA_YAML = str(PROJECT_ROOT / "config" / "dataset.yaml")
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

def train_yolov8():
    """训练 YOLOv8 检测异常血细胞"""
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载预训练模型
    model = YOLO("yolov8n.pt")
    
    # 训练
    results = model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        augment=True,
        patience=10,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(OUTPUT_DIR),
        name="yolov8_sepsis",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        cos_lr=True,
        amp=True,
        val=True,
        save=True,
        save_period=10,
    )
    
    # 保存最佳模型到 models 目录
    best_model = OUTPUT_DIR / "yolov8_sepsis" / "weights" / "best.pt"
    if best_model.exists():
        import shutil
        shutil.copy(best_model, MODEL_DIR / "yolov8_sepsis_best.pt")
        print(f"✅ 最佳模型已保存至: {MODEL_DIR / 'yolov8_sepsis_best.pt'}")

if __name__ == "__main__":
    train_yolov8()
