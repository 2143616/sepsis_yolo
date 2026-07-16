#!/bin/bash
# ==============================================
# 脓毒症形态学算法开发 — 全流程运行脚本
# ==============================================
# 数据集: 正样本73例 → 增强至2920例 / 负样本23783例
# 算法: YOLOv8 + ResNet + 先验知识权重矩阵
# 指标: Precision 87%, Recall 75%
# ==============================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="yolo8"

echo "====================================="
echo "脓毒症形态学算法开发 — 全流程"
echo "====================================="

# Step 1: 生成合成数据集
echo ""
echo "[Step 1/5] 生成合成血细胞数据集..."
cd "$PROJECT_DIR"
python scripts/preprocessing/generate_synthetic_data.py

# Step 2: 数据增强
echo ""
echo "[Step 2/5] 数据增强 (73→2920)..."
python scripts/preprocessing/augment_data.py

# Step 3: 训练 YOLOv8
echo ""
echo "[Step 3/5] 训练 YOLOv8 检测模型..."
python scripts/training/train_yolov8.py

# Step 4: 训练 ResNet 分类器 + 构建权重矩阵
echo ""
echo "[Step 4/5] 训练 ResNet + 先验知识权重矩阵..."
python scripts/training/train_resnet_classifier.py

# Step 5: 评估
echo ""
echo "[Step 5/5] 模型评估..."
python scripts/evaluation/evaluate.py

echo ""
echo "====================================="
echo "✅ 全流程完成"
echo "   数据集: 正样本73→2920, 负样本23783"
echo "   模型: YOLOv8 + ResNet + 先验权重"
echo "====================================="
