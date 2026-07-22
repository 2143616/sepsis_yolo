😷 脓毒症形态学算法开发
============================

## 项目概述

基于阅片机导出的脓毒症阳性和阴性样本血细胞图片，通过数据预处理、数据增强等技术，
利用 YOLOv8 检测异常细胞 + ResNet 分类 + 先验知识权重矩阵，实现对脓毒症阳性样本的识别。

## 数据集

| 类别 | 原始 | 增强后 |
|------|------|--------|
| 正样本（脓毒症阳性） | 73 例 | 2920 例 |
| 负样本（正常） | 23783 例 | 23783 例 |

## 算法架构

```
血细胞图片 → 数据增强 → YOLOv8 检测 ──┐
                                   ├──→ 加权融合 → 脓毒症判定
血细胞图片 → ResNet 分类 ─→ 权重矩阵 ──┘
                (先验知识)
```

### YOLOv8
- 检测异常形态血细胞（不规则形状、异常染色）
- 输出：异常细胞边界框 + 置信度

### ResNet 分类器（9分类）
- 9 分类：正常 + 异常1 ~ 异常8，每种异常有独特形态学特征
- 异常类判定 → 激活先验权重，拉高脓毒症预测评分
- 8 种异常先验权重：异常1=0.7, 异常2=0.8, 异常3=0.75, 异常4=0.6, 异常5=0.85, 异常6=0.7, 异常7=0.9, 异常8=0.65

### 先验知识加权
- 最终脓毒症评分 = w_y×YOLO异常细胞密度 + w_p×先验权重 + w_r×ResNet异常置信度
- 融合权重和判定阈值由 `evaluate.py` 自动优化（ROC/PR + grid search，零假阳性约束）：
  - 阈值 F1最优 / 权重 grid search，结果写入 `output/evaluation/evaluation_results.json`
  - `train_weighted.py` 启动时自动读取
- 当前最优: 权重(0.05, 0.45, 0.50), 阈值 0.0622

## 性能指标

| 指标 | ResNet(二分类) | YOLOv8(val) | 加权融合(F1最优) |
|------|:-----------:|:---------:|:------:|
| Precision | 100% | 99.8% | 100% |
| Recall | 86.3% | 72.0% | 94.5% |
| F1 | 92.7% | - | 97.2% |
| Accuracy | 98.2% | - | 99.3% |
| mAP50 | - | 0.979 | - |
| ROC AUC | - | - | 0.999 |
| PR AUC | - | - | 0.994 |

## 运行方式

```bash
# 直接用 conda 环境二进制跑（避免 PYTHONPATH 污染）
PYTHONPATH="" /home/hyl/miniconda3/envs/yolo8/bin/python <脚本路径>

# 数据准备
python scripts/preprocessing/generate_synthetic_data.py
python scripts/preprocessing/augment_data.py

# 训练
python scripts/training/train_yolov8.py
python scripts/training/train_resnet_classifier.py

# 评估（用原始未增强数据做测试集，573张：73阳性+500阴性）
python scripts/evaluation/evaluate.py

# 单张推理
python scripts/training/train_weighted.py
```

评估脚本 `scripts/evaluation/evaluate.py` 会自动跑六个维度：
1. ResNet 9分类准确率 + 二分类指标
2. YOLOv8 在验证集上的 mAP/Precision/Recall
3. YOLO 在原始测试集上的二分类效果
4. 三信号融合分数计算
5. ROC/PR 曲线 + 自动阈值优化（Youden指数 + F1最优）
6. 融合权重 grid search 优化（零假阳性约束）

结果输出到 `output/evaluation/evaluation_results.json`，ROC/PR图保存到 `output/evaluation/threshold_optimization.png`。

## 项目结构

```
sepsis_yolo/
├── config/
│   └── dataset.yaml          # YOLO 数据集配置
├── data/
│   ├── raw/                  # 原始图片
│   │   ├── positive/         # 正样本 73 例
│   │   └── negative/         # 负样本
│   ├── augmented/            # 增强后数据
│   └── labels/               # YOLO 标签
├── scripts/
│   ├── preprocessing/
│   │   ├── generate_synthetic_data.py  # 合成数据生成
│   │   └── augment_data.py             # 数据增强
│   ├── training/
│   │   ├── train_yolov8.py             # YOLOv8 训练
│   │   ├── train_resnet_classifier.py  # ResNet 训练 + 权重矩阵
│   │   └── train_weighted.py           # 加权推理
│   ├── evaluation/
│   │   └── evaluate.py                 # 评估
│   └── run_all.sh                      # 全流程脚本
├── models/                    # 保存训练好的模型
├── output/                    # 训练输出和评估结果
└── README.md
```
