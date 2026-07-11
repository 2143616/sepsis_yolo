"""
脓毒症形态学算法开发 — 合成血细胞图片生成
8 种异常形态 + 正常 = 9 分类
异常1-异常8：每种有独特的形态学特征
"""
import numpy as np
import cv2
from pathlib import Path
import random
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

RAW_DIR = Path("/home/hyl/project/sepsis_yolo/data/raw")
IMG_SIZE = (640, 640)

# 8 种异常细胞的绘制参数: (color, shape_desc, nucleus_ratio)
ABNORMAL_TYPES = [
    ( (60, 30, 180), "不规则大细胞", 0.6 ),   # 异常1
    ( (40, 80, 200), "多核巨细胞",   0.7 ),   # 异常2
    ( (90, 20, 140), "彗星形细胞",   0.4 ),   # 异常3
    ( (30, 90, 160), "碎裂细胞",     0.3 ),   # 异常4
    ( (110, 50, 80), "哑铃形细胞",   0.5 ),   # 异常5
    ( (50, 60, 220), "空泡细胞",     0.3 ),   # 异常6
    ( (70, 100, 50), "毛刺细胞",     0.4 ),   # 异常7
    ( (130, 30, 60), "桑椹样细胞",   0.7 ),   # 异常8
]

def draw_normal_cells(img):
    """绘制正常红细胞+白细胞"""
    for _ in range(random.randint(80, 200)):
        cx, cy = random.randint(10, 630), random.randint(10, 630)
        r = random.randint(4, 8)
        s = random.randint(180, 220)
        cv2.circle(img, (cx, cy), r, (s, s-20, s-10), -1)
    for _ in range(random.randint(3, 15)):
        cx, cy = random.randint(15, 625), random.randint(15, 625)
        r = random.randint(8, 14)
        cv2.circle(img, (cx, cy), r, (80, 40, 120), -1)
        cv2.circle(img, (cx-2, cy-2), r//2, (50, 20, 80), -1)

def draw_abnormal_cell(img, atyped_idx):
    """绘制指定类型的异常细胞，返回边界框"""
    color, _desc, nuc_ratio = ABNORMAL_TYPES[atyped_idx]
    cx, cy = random.randint(20, 620), random.randint(20, 620)
    r = random.randint(10, 18)
    
    if atyped_idx == 0:   # 异常1: 不规则大细胞
        pts = cv2.ellipse2Poly((cx, cy), (r, r+random.randint(2,6)),
                               random.randint(0,180), 0, 360, 8)
        cv2.fillPoly(img, [pts], color)
    elif atyped_idx == 1: # 异常2: 多核巨细胞
        cv2.circle(img, (cx, cy), r, color, -1)
        for dx, dy in [(-4,-3),(3,-4),(0,3),(5,2)]:
            cv2.circle(img, (cx+dx, cy+dy), r//3, (20, 40, 120), -1)
    elif atyped_idx == 2: # 异常3: 彗星形细胞
        pts = np.array([(cx,cy-r),(cx-r//2,cy),(cx,cy+r),(cx+r,cy)], np.int32)
        cv2.fillPoly(img, [pts], color)
        cv2.circle(img, (cx-r//3, cy), r//4, (20, 10, 80), -1)
    elif atyped_idx == 3: # 异常4: 碎裂细胞
        for _ in range(3):
            ox, oy = random.randint(-r//2, r//2), random.randint(-r//2, r//2)
            cv2.circle(img, (cx+ox, cy+oy), r//2, color, -1)
            cv2.circle(img, (cx+ox-2, cy+oy-2), r//4, (20, 40, 100), -1)
    elif atyped_idx == 4: # 异常5: 哑铃形细胞
        cv2.circle(img, (cx-r//2, cy), r//2, color, -1)
        cv2.circle(img, (cx+r//2, cy), r//2, color, -1)
        cv2.rectangle(img, (cx-2, cy-r//3), (cx+2, cy+r//3), color, -1)
    elif atyped_idx == 5: # 异常6: 空泡细胞
        cv2.circle(img, (cx, cy), r, color, -1)
        cv2.circle(img, (cx, cy), r//2, (220, 210, 200), -1)
    elif atyped_idx == 6: # 异常7: 毛刺细胞
        cv2.circle(img, (cx, cy), r, color, -1)
        for a in range(0, 360, 45):
            rad = np.deg2rad(a)
            ex, ey = int(cx + r*1.4*np.cos(rad)), int(cy + r*1.4*np.sin(rad))
            cv2.line(img, (cx, cy), (ex, ey), color, 2)
    elif atyped_idx == 7: # 异常8: 桑椹样细胞
        for dx, dy in [(0,0),(-5,-4),(5,-4),(-4,4),(4,4),(-7,0),(7,0)]:
            cv2.circle(img, (cx+dx, cy+dy), r//3, color, -1)
            cv2.circle(img, (cx+dx-1, cy+dy-1), r//5, (20, 10, 70), -1)
    
    x1, y1, x2, y2 = cx-r-5, cy-r-5, cx+r+5, cy+r+5
    bbox = [max(0,x1), max(0,y1), min(639,x2), min(639,y2)]
    return bbox, atyped_idx  # class label = atyped_idx (1-8 for abnormal)

def create_blood_cell_image(has_sepsis_markers=False):
    """生成模拟血细胞涂片图像"""
    img = np.ones((*IMG_SIZE, 3), dtype=np.uint8) * 220
    draw_normal_cells(img)
    bboxes = []
    if has_sepsis_markers:
        n_abnormal = random.randint(3, 8)
        for _ in range(n_abnormal):
            atyped = random.randint(0, 7)
            bbox, cls = draw_abnormal_cell(img, atyped)
            bboxes.append((*bbox, cls))
    return img, bboxes

def save_dataset():
    pos_dir = RAW_DIR / "positive"
    neg_dir = RAW_DIR / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)
    
    print("生成正样本（脓毒症阳性，含8类异常细胞）...")
    for i in tqdm(range(73)):
        img, bboxes = create_blood_cell_image(has_sepsis_markers=True)
        cv2.imwrite(str(pos_dir / f"sepsis_pos_{i:04d}.png"), img)
        label_path = RAW_DIR.parent / "labels" / f"sepsis_pos_{i:04d}.txt"
        h, w = img.shape[:2]
        with open(label_path, 'w') as f:
            for x1, y1, x2, y2, cls in bboxes:
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    
    print("生成负样本（正常）...")
    n_neg = min(500, 23783)
    for i in tqdm(range(n_neg)):
        img, _ = create_blood_cell_image(has_sepsis_markers=False)
        cv2.imwrite(str(neg_dir / f"normal_{i:04d}.png"), img)
    
    print(f"正样本: 73 例 (含异常1-8), 负样本: {n_neg} 例")
    print("原始数据集规格: 正样本73例, 负样本23783例")

if __name__ == "__main__":
    save_dataset()
    print("✅ 合成数据集生成完成")
