"""
9 分类 ResNet: 正常 + 异常1~异常8
先验知识权重: 异常类→更高脓毒症预测权重
"""
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np, cv2
from pathlib import Path
from tqdm import tqdm
import random, json

random.seed(42); np.random.seed(42); torch.manual_seed(42)
PROJECT_ROOT = Path("/home/hyl/project/sepsis_yolo")
AUG_DIR = PROJECT_ROOT / "data" / "augmented"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output" / "resnet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 先验权重: 正常=0, 异常1~8 → 高权重 (异常越典型权重越高)
PRIOR_WEIGHTS = [0.0, 0.7, 0.8, 0.75, 0.6, 0.85, 0.7, 0.9, 0.65]

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1))
        self.res1 = self._block(64, 64, 2)
        self.res2 = self._block(64, 128, 2, stride=2)
        self.res3 = self._block(128, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _block(self, in_ch, out_ch, n, stride=1):
        layers = [nn.Conv2d(in_ch, out_ch, 3, stride, 1), nn.BatchNorm2d(out_ch), nn.ReLU()]
        for _ in range(n-1):
            layers += [nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.ReLU()]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x); x = self.res2(x); x = self.res3(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))

class BloodCellDataset(Dataset):
    def __init__(self, img_dir):
        self.paths = list(Path(img_dir).glob("*.png"))
        self.labels = []
        for p in self.paths:
            if "sepsis_pos" in p.name:
                # 从文件名确定主导异常类型（用第一个异常标签或随机分配）
                label_path = Path(str(p).replace("augmented", "labels").replace(".png", ".txt"))
                if label_path.exists():
                    with open(label_path) as f:
                        cls_list = [int(l.split()[0]) for l in f if l.strip()]
                    # 该图的主要异常类型 = 出现最多的异常类
                    self.labels.append(max(set(cls_list), key=cls_list.count) if cls_list else 1)
                else:
                    self.labels.append(random.randint(1, 8))
            else:
                self.labels.append(0)  # 正常

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]))
        img = cv2.resize(img, (224, 224)).transpose(2,0,1).astype(np.float32) / 255.0
        img = (img - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
        return torch.FloatTensor(img), self.labels[idx]

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"分类数: 9 (正常 + 异常1~8)")
    print(f"先验权重: {dict(zip(['正常']+[f'异常{i}' for i in range(1,9)], PRIOR_WEIGHTS))}")

    ds = BloodCellDataset(AUG_DIR)
    n_train = int(0.8 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, len(ds)-n_train])
    train_loader = DataLoader(train_ds, 32, True, num_workers=4)
    val_loader = DataLoader(val_ds, 32, False, num_workers=4)

    model = SimpleResNet(9).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0]+[3.0]*8).to(device))
    optimizer = optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)

    best_acc = 0.0
    for epoch in range(20):
        model.train()
        loss_sum = 0.0
        for inp, lbl in tqdm(train_loader, desc=f"Epoch {epoch+1}/20", leave=False):
            inp, lbl = inp.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inp), lbl)
            loss.backward(); optimizer.step()
            loss_sum += loss.item()
        model.eval()
        correct = sum((model(i.to(device)).max(1)[1] == l.to(device)).sum().item()
                      for i, l in val_loader)
        acc = 100 * correct / len(val_ds)
        scheduler.step()
        print(f"  Epoch {epoch+1}: loss={loss_sum/len(train_loader):.4f}  val_acc={acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_DIR / "resnet9_best.pth")
    print(f"✅ 最佳准确率: {best_acc:.2f}%")

    # 保存先验权重映射
    with open(OUTPUT_DIR / "prior_weights.json", 'w') as f:
        json.dump({f"异常{i}": w for i, w in enumerate(PRIOR_WEIGHTS)}, f, ensure_ascii=False)
    print("✅ 先验权重已保存")

if __name__ == "__main__":
    train()
