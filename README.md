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
- 最终脓毒症评分 = 0.5×YOLO异常细胞密度 + 0.4×先验权重(异常类) + 0.1×ResNet置信度
- 当检测到异常细胞时，先验权重进一步放大

## 性能指标

| 指标 | 数值 |
|------|------|
| Precision | 87% |
| Recall | 75% |

## 运行方式

```bash
# 激活环境
conda activate yolo8

# 全流程运行
bash scripts/run_all.sh

# 或分步运行
python scripts/preprocessing/generate_synthetic_data.py
python scripts/preprocessing/augment_data.py
python scripts/training/train_yolov8.py
python scripts/training/train_resnet_classifier.py
python scripts/training/train_weighted.py      # 加权推理演示
python scripts/evaluation/evaluate.py
```

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
