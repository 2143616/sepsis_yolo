"""
数据增强 — 原始73例正样本→2920例，保持9类标签
"""
import cv2, numpy as np, os
from pathlib import Path
from tqdm import tqdm
import albumentations as A
import random
random.seed(42)

RAW_POS_DIR = Path("/home/hyl/project/sepsis_yolo/data/raw/positive")
AUG_DIR = Path("/home/hyl/project/sepsis_yolo/data/augmented")
LABELS_DIR = Path("/home/hyl/project/sepsis_yolo/data/labels")
AUG_DIR.mkdir(parents=True, exist_ok=True)

aug_pipeline = A.Compose([
    A.Rotate(limit=30, p=0.9),
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.8),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.6),
    A.GaussNoise(p=0.5), A.Blur(blur_limit=3, p=0.3),
    A.RandomGamma(gamma_limit=(80,120), p=0.4),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['cls'], min_visibility=0.3))

def augment_dataset():
    pos_files = sorted(RAW_POS_DIR.glob("*.png"))
    print(f"原始正样本: {len(pos_files)} 例")
    target_per = 2920 // len(pos_files) + 1
    total = 0
    for img_path in tqdm(pos_files):
        img = cv2.imread(str(img_path))
        label_path = LABELS_DIR / (img_path.stem + ".txt")
        bboxes, clses = [], []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        bboxes.append([float(p) for p in parts[1:]])
                        clses.append(int(parts[0]))
        cv2.imwrite(str(AUG_DIR / img_path.name), img); total += 1
        for i in range(target_per - 1):
            try:
                aug = aug_pipeline(image=img, bboxes=bboxes, cls=clses)
                aug_name = f"{img_path.stem}_aug{i:03d}.png"
                cv2.imwrite(str(AUG_DIR / aug_name), aug['image'])
                aug_label = AUG_DIR.parent / "labels" / f"{img_path.stem}_aug{i:03d}.txt"
                with open(aug_label, 'w') as f:
                    for c, bb in zip(aug['cls'], aug['bboxes']):
                        f.write(f"{c} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")
                total += 1
                if total >= 2920: break
            except: continue
        if total >= 2920: break
    print(f"增强后正样本: {total} 例 (含9分类标签)")

if __name__ == "__main__":
    augment_dataset()
    print("✅ 数据增强完成")
