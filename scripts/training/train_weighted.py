"""
YOLOv8 + 9分类 ResNet 加权推理
异常1~8 → 更高脓毒症权重
"""
import torch, numpy as np, cv2, json
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path("/home/hyl/project/sepsis_yolo")
MODEL_DIR = PROJECT_ROOT / "models"
WEIGHT_FILE = PROJECT_ROOT / "output" / "resnet" / "prior_weights.json"
EVAL_RESULT = PROJECT_ROOT / "output" / "evaluation" / "evaluation_results.json"

# 从评估结果中加载最优阈值和权重，没跑过评估就用默认值
def _load_eval_params():
    try:
        with open(EVAL_RESULT) as f:
            data = json.load(f)
        fusion = data.get("fusion", {})
        # F1最优: 保证Precision不降的前提下最大化召回
        thresh = fusion.get("threshold_f1") or fusion.get("threshold_youden")
        if thresh is None:
            thresh = 0.35
        # 最优融合权重
        weights = fusion.get("weights_optimal", [0.30, 0.25, 0.45])
        return float(thresh), (float(weights[0]), float(weights[1]), float(weights[2]))
    except Exception:
        return 0.35, (0.30, 0.25, 0.45)

OPTIMAL_THRESHOLD, FUSION_WEIGHTS = _load_eval_params()

# 先验权重: 9类 [正常, 异常1~8]
PRIOR_WEIGHTS = [0.0, 0.7, 0.8, 0.75, 0.6, 0.85, 0.7, 0.9, 0.65]

class SepsisDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        yolo_p = MODEL_DIR / "yolov8_sepsis_best.pt"
        self.yolo = YOLO(str(yolo_p)) if yolo_p.exists() else YOLO("yolov8n.pt")

        from train_resnet_classifier import SimpleResNet
        self.resnet = SimpleResNet(9).to(self.device)
        rn_p = MODEL_DIR / "resnet9_best.pth"
        if rn_p.exists():
            self.resnet.load_state_dict(torch.load(rn_p, map_location=self.device))
        self.resnet.eval()

    def _prep(self, img):
        img = cv2.resize(img, (224,224)).transpose(2,0,1).astype(np.float32)/255.0
        img = (img - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
        return torch.FloatTensor(img).unsqueeze(0).to(self.device)

    def predict(self, image_path, conf_threshold=0.25):
        img = cv2.imread(str(image_path))
        if img is None: return {"error": "无法读取图片"}

        # YOLOv8 检测
        yolo_r = self.yolo(image_path, conf=conf_threshold)[0]
        n_det = len(yolo_r.boxes)

        # ResNet 9分类
        inp = self._prep(img)
        with torch.no_grad():
            logits = self.resnet(inp)
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = probs.argmax().item()

        # 先验加权: 异常类→高权重
        prior_w = PRIOR_WEIGHTS[pred_class]
        yolo_score = min(1.0, n_det / 3.0)
        # ResNet 异常概率贡献
        rn_abnormal = probs[1:].max().item()

        # 融合: YOLO 检测 + 先验知识 + ResNet 分类（权重来自评估优化）
        w_y, w_p, w_r = FUSION_WEIGHTS
        final_score = w_y * yolo_score + w_p * prior_w + w_r * rn_abnormal

        result = {
            "sepsis_score": round(final_score, 4),
            "abnormal_cells_detected": n_det,
            "resnet_pred_class": int(pred_class),
            "resnet_pred_label": f"异常{pred_class}" if pred_class > 0 else "正常",
            "prior_weight": round(prior_w, 4),
            "is_sepsis_positive": final_score > OPTIMAL_THRESHOLD,
            "detections": []
        }
        for box in yolo_r.boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            result["detections"].append({
                "bbox": [int(x1),int(y1),int(x2),int(y2)],
                "confidence": round(box.conf[0].item(), 4)
            })
        return result

def demo():
    d = SepsisDetector()
    print("="*60)
    print("脓毒症检测 — 9分类 ResNet + 先验加权")
    print("先验权重:", dict(zip(['正常']+[f'异常{i}' for i in range(1,9)], PRIOR_WEIGHTS)))
    print("="*60)
    test_dir = PROJECT_ROOT / "data" / "augmented"
    for f in list(test_dir.glob("*.png"))[:5]:
        r = d.predict(f)
        st = "⚠️ 阳性" if r["is_sepsis_positive"] else "✅ 阴性"
        print(f"\n{f.name}: {st}")
        print(f"  评分={r['sepsis_score']}  异常细胞={r['abnormal_cells_detected']}")
        print(f"  ResNet分类={r['resnet_pred_label']}  先验权重={r['prior_weight']}")

if __name__ == "__main__":
    demo()
