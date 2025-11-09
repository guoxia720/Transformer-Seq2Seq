# Transformer-Seq2Seq Implementation

完整的 Encoder-Decoder Transformer 从零实现，用于序列到序列（Seq2Seq）任务，如机器翻译。

---

## 项目特性

-  Encoder-Decoder 架构
  - Multi-Head Self-Attention
  - Masked Multi-Head Attention（Decoder）
  - Encoder-Decoder Cross-Attention
  - Position-wise Feed-Forward Network
  - Residual Connection + Layer Normalization
  - Sinusoidal Positional Encoding
-  Mask 机制
  - Source Padding Mask（屏蔽 PAD）
  - Target Causal Mask（屏蔽未来信息）
  - Memory Mask（Encoder-Decoder Attention）
- 训练优化
  - AdamW Optimizer
  - Learning Rate Warmup Scheduler
  - Gradient Clipping
  - Dropout Regularization
- 可复现性
  - 固定随机种子
  - 完整训练脚本
  - 模型保存与加载

---

## 项目结构

```
Transformer-Seq2Seq/
├── src/
│   ├── model.py          # Transformer 模型实现
│   ├── data.py           # 数据加载与预处理
│   ├── train.py          # 训练脚本
│   └── utils.py          # 工具函数（mask、checkpoint等）
├── scripts/
│   └── run.sh            # 一键运行训练脚本
├── checkpoints/          # 模型检查点
├── results/              # 训练结果
│   ├── loss_curve.png    # 损失曲线
│   └── metrics.csv       # 训练指标
├── requirements.txt      # Python 依赖
├── README.md             # 项目说明
└── .gitignore            # Git 忽略文件
```

---

## 快速开始

### 1. 环境配置

Python 版本: 3.8+
PyTorch 版本: 2.0+

安装依赖：

```bash
pip install -r requirements.txt
```

### 2. 数据集

默认使用 IWSLT2017  German-English 翻译数据集。


### 3. 训练模型

#### 方法 1: 使用脚本（推荐）

```bash
bash scripts/run.sh
```

#### 方法 2: 直接运行

```bash
python src/train_test.py \
    --dataset iwslt2017 \
    --dataset_config de-en \
    --batch_size  \128
    --epochs 20 \
    --d_model 512 \
    --num_heads 8 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --seed 42
```

### 4. 恢复训练

```bash
python src/train_test.py --resume --checkpoint_path checkpoints/best_model.pt
```

---

## 超参数配置

| 参数                     | 默认值       | 说明         |
| ------------------------ |-----------| ------------ |
| --dataset                | iwslt2017 | 数据集名称   |
| --dataset_config         | de-en     | 数据集配置（语言对） |
| --batch_size             | 128       | 批次大小     |
| --epochs                 | 20        | 训练轮数     |
| --d_model                | 512       | 模型维度     |
| --num_heads              | 8         | 注意力头数   |
| --num_encoder_layers     | 6         | Encoder 层数 |
| --num_decoder_layers     | 6         | Decoder 层数 |
| --d_ff                   | 2048      | FFN 隐藏层维度 |
| --max_len                | 64        | 最大序列长度 |
| --dropout                | 0.1       | Dropout 比例 |
| --lr                     | 1.0       | 学习率       |
| --warmup_steps           | 3000      | Warmup 步数  |
| --clip_grad              | 1.0       | 梯度裁剪阈值 |
| --seed                   | 42        | 随机种子     |

---

## 训练结果

训练完成后，结果保存在 src/results/ 目录：

1. loss_curve.png: 训练和验证损失曲线
2. metrics.csv: 每个 epoch 的详细指标

模型检查点保存在 checkpoints/ 目录：
- best_model.pt: 验证集最佳模型
- final_model.pt: 最终模型
- checkpoint_epoch_N.pt: 定期保存的检查点

---

## 模型架构详解

### Encoder

```
Input → Embedding → Positional Encoding
  ↓
[Encoder Layer] × N
  ├─ Multi-Head Self-Attention
  ├─ Add & Norm
  ├─ Feed-Forward Network
  └─ Add & Norm
  ↓
Encoder Output
```

### Decoder

```
Target → Embedding → Positional Encoding
  ↓
[Decoder Layer] × N
  ├─ Masked Multi-Head Self-Attention
  ├─ Add & Norm
  ├─ Encoder-Decoder Cross-Attention
  ├─ Add & Norm
  ├─ Feed-Forward Network
  └─ Add & Norm
  ↓
Linear → Softmax → Output
```

### Mask 机制

1. Source Padding Mask: 屏蔽源序列中的 PAD token
2. Target Causal Mask: 防止 Decoder 看到未来信息（下三角矩阵）
3. Memory Mask: 屏蔽 Encoder 输出中的 PAD 位置

---

## 消融实验：移除位置编码

python train_without_pos_code.py

---
## 不同模型变体的比较
1. 比较不同模型大小 [[256, 384, 512]]
python compare_d_model.py
2. 比较不同数目注意力头 [2, 4, 8]
python compare_head.py

---

## 实现

- 完整 Encoder-Decoder (含 Masked Decoder)
- Sinusoidal 位置编码
- 完整 Mask 机制 (Padding + Causal + Memory)
- AdamW + Warmup 学习率调度器
- 梯度裁剪
- 模型保存与加载
- 训练曲线可视化
- 固定随机种子


---

## License

MIT License

---

## 参考文献

1. Vaswani et al., "Attention Is All You Need", NeurIPS 2017
