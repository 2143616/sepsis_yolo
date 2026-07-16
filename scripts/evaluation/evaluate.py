"""
真实评估脚本 — 用原始未增强数据测试三个模型
YOLOv8 检测 + ResNet 9分类 + 加权融合
"""
import torch, numpy as np, cv2, json, sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("/home/hyl/project/sepsis_yolo")
RAW_POS = PROJECT_ROOT / "data" / "raw" / "positive"
RAW_NEG = PROJECT_ROOT / "data" / "raw" / "negative"
LABEL_DIR = PROJECT_ROOT / "data" / "labels"
MODEL_DIR = PROJECT_ROOT / "models"
EVAL_DIR = PROJECT_ROOT / "output" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

PRIOR_WEIGHTS = [0.0, 0.7, 0.8, 0.75, 0.6, 0.85, 0.7, 0.9, 0.65]

# ---------- ResNet ----------
class SimpleResNet(torch.nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64), torch.nn.ReLU(), torch.nn.MaxPool2d(3, stride=2, padding=1))
        self.res1 = self._block(64, 64, 2)
        self.res2 = self._block(64, 128, 2, stride=2)
        self.res3 = self._block(128, 256, 2, stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(256, num_classes)
    def _block(self, in_ch, out_ch, n, stride=1):
        layers = [torch.nn.Conv2d(in_ch, out_ch, 3, stride, 1), torch.nn.BatchNorm2d(out_ch), torch.nn.ReLU()]
        for _ in range(n-1):
            layers += [torch.nn.Conv2d(out_ch, out_ch, 3, 1, 1), torch.nn.BatchNorm2d(out_ch), torch.nn.ReLU()]
        return torch.nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x); x = self.res1(x); x = self.res2(x); x = self.res3(x)
        return self.fc(torch.flatten(self.avgpool(x), 1))

def resnet_prep(img):
    img = cv2.resize(img, (224,224)).astype(np.float32) / 255.0
    img = img.transpose(2,0,1)
    mean = np.array([0.485,0.456,0.406], dtype=np.float32).reshape(3,1,1)
    std  = np.array([0.229,0.224,0.225], dtype=np.float32).reshape(3,1,1)
    return torch.FloatTensor((img - mean) / std).unsqueeze(0)

def load_resnet(device):
    model = SimpleResNet(9).to(device)
    pth = MODEL_DIR / "resnet9_best.pth"
    if pth.exists():
        model.load_state_dict(torch.load(pth, map_location=device))
    model.eval()
    return model

# ---------- YOLO ----------
def load_yolo():
    from ultralytics import YOLO
    pth = MODEL_DIR / "yolov8_sepsis_best.pt"
    return YOLO(str(pth)) if pth.exists() else YOLO("yolov8n.pt")

# ---------- Metrics ----------
def calc_metrics(y_true, y_pred, y_score=None):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = np.sum((y_true==1) & (y_pred==1))
    fp = np.sum((y_true==0) & (y_pred==1))
    fn = np.sum((y_true==1) & (y_pred==0))
    tn = np.sum((y_true==0) & (y_pred==0))
    return {
        "precision": round(tp/(tp+fp),4) if (tp+fp)>0 else 0.0,
        "recall": round(tp/(tp+fn),4) if (tp+fn)>0 else 0.0,
        "accuracy": round((tp+tn)/len(y_true),4),
        "f1": round(2*tp/(2*tp+fp+fn),4) if (2*tp+fp+fn)>0 else 0.0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
    }

# ---------- 主评估 ----------
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 收集测试图片: 73阳性 + 500阴性 = 573张
    test_items = []
    # 阳性: 从 raw/positive 读取
    for p in sorted(RAW_POS.glob("*.png")):
        # 读取标签判断该图的异常类
        label_path = LABEL_DIR / (p.stem + ".txt")
        true_class = 0  # 正常
        if label_path.exists():
            with open(label_path) as f:
                classes = [int(float(l.split()[0])) for l in f if l.strip()]
            # YOLO class 0-7 -> ResNet class 1-8
            true_class = (max(set(classes), key=classes.count) if classes else 0) + 1
        test_items.append({"path": str(p), "is_sepsis": True, "true_class": true_class, "has_bboxes": label_path.exists()})
    
    # 阴性: 从 raw/negative 读取
    for p in sorted(RAW_NEG.glob("*.png")):
        test_items.append({"path": str(p), "is_sepsis": False, "true_class": 0, "has_bboxes": False})
    
    n_pos = sum(1 for t in test_items if t["is_sepsis"])
    n_neg = sum(1 for t in test_items if not t["is_sepsis"])
    print(f"测试集: {len(test_items)} 张 ({n_pos} 阳性, {n_neg} 阴性)")
    
    # ====== 1. ResNet 9分类评估 ======
    print("\n" + "="*60)
    print("1. ResNet 9分类器评估")
    print("="*60)
    rn = load_resnet(device)
    y_true_cls, y_pred_cls = [], []
    y_true_bin, y_pred_bin = [], []  # 二分类: 正常vs异常
    
    for item in test_items:
        img = cv2.imread(item["path"])
        if img is None: continue
        inp = resnet_prep(img).to(device)
        with torch.no_grad():
            probs = torch.softmax(rn(inp), dim=1)[0].cpu().numpy()
        pred_cls = int(probs.argmax())
        y_true_cls.append(item["true_class"])
        y_pred_cls.append(pred_cls)
        y_true_bin.append(0 if item["true_class"] == 0 else 1)
        y_pred_bin.append(0 if pred_cls == 0 else 1)
    
    cls_acc = sum(a==b for a,b in zip(y_true_cls, y_pred_cls)) / len(y_true_cls)
    m = calc_metrics(y_true_bin, y_pred_bin)
    print(f"  9分类准确率: {cls_acc*100:.1f}%")
    print(f"  二分类(正常/异常):")
    print(f"    Precision={m['precision']:.1%}  Recall={m['recall']:.1%}  F1={m['f1']:.1%}  Acc={m['accuracy']:.1%}")
    print(f"    TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
    
    # 各类别准确率
    from collections import Counter
    cls_correct = defaultdict(lambda: [0,0])  # correct, total
    for t,p in zip(y_true_cls, y_pred_cls):
        cls_correct[t][1] += 1
        if t == p: cls_correct[t][0] += 1
    label_names = {0:"正常",1:"异常1",2:"异常2",3:"异常3",4:"异常4",5:"异常5",6:"异常6",7:"异常7",8:"异常8"}
    print("  各类别准确率:")
    for cls_id in sorted(cls_correct):
        c, tot = cls_correct[cls_id]
        name = label_names.get(cls_id, f"类{cls_id}")
        print(f"    {name}: {c}/{tot} ({c/tot*100:.1f}%)")
    
    # ====== 2. YOLOv8 验证 ======
    print("\n" + "="*60)
    print("2. YOLOv8 验证集评估 (ultralytics val)")
    print("="*60)
    yolo = load_yolo()
    val_results = yolo.val(data=str(PROJECT_ROOT/"config"/"dataset.yaml"), split="val", plots=False)
    print(f"  mAP50: {val_results.box.map50:.4f}")
    print(f"  mAP50-95: {val_results.box.map:.4f}")
    print(f"  Precision: {val_results.box.mp:.4f}")
    print(f"  Recall: {val_results.box.mr:.4f}")
    
    # ====== 3. YOLO 在原始测试集上的二分类 ======
    print("\n" + "="*60)
    print("3. YOLO 原始测试集二分类 (有无异常细胞)")
    print("="*60)
    yolo_true, yolo_pred, yolo_scores = [], [], []
    for item in test_items:
        r = yolo(item["path"], conf=0.25, verbose=False)[0]
        n_det = len(r.boxes)
        has_abnormal = n_det > 0
        yolo_true.append(1 if item["is_sepsis"] else 0)
        yolo_pred.append(1 if has_abnormal else 0)
        yolo_scores.append(min(1.0, n_det / 5.0))
    m = calc_metrics(yolo_true, yolo_pred)
    print(f"  Precision={m['precision']:.1%}  Recall={m['recall']:.1%}  F1={m['f1']:.1%}  Acc={m['accuracy']:.1%}")
    print(f"  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
    
    # ====== 4. 加权融合评估 ======
    print("\n" + "="*60)
    print("4. 加权融合 (YOLO + ResNet + 先验权重)")
    print("="*60)
    fuse_true, fuse_pred, fuse_scores = [], [], []
    details = []
    
    for item in test_items:
        img = cv2.imread(item["path"])
        if img is None: continue
        
        # YOLO
        r = yolo(item["path"], conf=0.25, verbose=False)[0]
        n_det = len(r.boxes)
        yolo_score = min(1.0, n_det / 3.0)

        # ResNet
        inp = resnet_prep(img).to(device)
        with torch.no_grad():
            probs = torch.softmax(rn(inp), dim=1)[0].cpu().numpy()
        pred_cls = int(probs.argmax())
        rn_abnormal = probs[1:].max().item()
        prior_w = PRIOR_WEIGHTS[pred_cls]

        final_score = 0.3 * yolo_score + 0.25 * prior_w + 0.45 * rn_abnormal
        is_pos = final_score > 0.35
        
        fuse_true.append(1 if item["is_sepsis"] else 0)
        fuse_pred.append(1 if is_pos else 0)
        fuse_scores.append(final_score)
        details.append({
            "file": Path(item["path"]).name,
            "true": "阳性" if item["is_sepsis"] else "阴性",
            "score": round(final_score, 4),
            "pred": "阳性" if is_pos else "阴性",
            "yolo_det": n_det,
            "resnet_cls": pred_cls,
            "prior_w": round(prior_w, 4)
        })
    
    m = calc_metrics(fuse_true, fuse_pred)
    print(f"  Precision={m['precision']:.1%}  Recall={m['recall']:.1%}  F1={m['f1']:.1%}  Acc={m['accuracy']:.1%}")
    print(f"  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
    
    # 打印一些误判样本
    errors = [d for d in details if (d["true"]=="阳性" and d["pred"]=="阴性") or (d["true"]=="阴性" and d["pred"]=="阳性")]
    print(f"\n  误判样本 ({len(errors)}/{len(details)}):")
    for e in errors[:10]:
        tag = "漏检" if e["true"]=="阳性" else "误报"
        print(f"    [{tag}] {e['file']} score={e['score']:.3f} yolo_det={e['yolo_det']} rn_cls={e['resnet_cls']}")
    if len(errors) > 10:
        print(f"    ... 还有 {len(errors)-10} 个")
    
    # ====== 汇总 ======
    print("\n" + "="*60)
    print("汇总对比")
    print("="*60)
    print(f"  {'模型':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Acc':>10}")
    print(f"  {'-'*60}")
    # ResNet 二分类
    resnet_bin = calc_metrics(y_true_bin, y_pred_bin)
    print(f"  {'ResNet(二分类)':<20} {resnet_bin['precision']:>10.1%} {resnet_bin['recall']:>10.1%} {resnet_bin['f1']:>10.1%} {resnet_bin['accuracy']:>10.1%}")
    yolo_bin = calc_metrics(yolo_true, yolo_pred)
    print(f"  {'YOLOv8(二分类)':<20} {yolo_bin['precision']:>10.1%} {yolo_bin['recall']:>10.1%} {yolo_bin['f1']:>10.1%} {yolo_bin['accuracy']:>10.1%}")
    fusion = calc_metrics(fuse_true, fuse_pred)
    print(f"  {'加权融合':<20} {fusion['precision']:>10.1%} {fusion['recall']:>10.1%} {fusion['f1']:>10.1%} {fusion['accuracy']:>10.1%}")
    
    # 保存结果
    results = {
        "test_set": {"total": len(test_items), "positive": n_pos, "negative": n_neg},
        "resnet": {
            "9class_accuracy": round(cls_acc, 4),
            "binary": resnet_bin
        },
        "yolov8": {
            "mAP50": round(val_results.box.map50, 4),
            "mAP50_95": round(val_results.box.map, 4),
            "binary": yolo_bin
        },
        "fusion": fusion,
        "errors": errors
    }
    with open(EVAL_DIR / "evaluation_results.json", 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {EVAL_DIR / 'evaluation_results.json'}")

if __name__ == "__main__":
    evaluate()
