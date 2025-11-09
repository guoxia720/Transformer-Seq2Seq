#!/bin/bash

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Training configuration
DATASET="wmt16"
DATASET_CONFIG="de-en"
BATCH_SIZE=32
EPOCHS=10
D_MODEL=512
NUM_HEADS=8
NUM_ENCODER_LAYERS=6
NUM_DECODER_LAYERS=6
D_FF=2048
MAX_LEN=128
DROPOUT=0.1
LR=1e-4
WARMUP_STEPS=4000
SEED=42

echo "=========================================="
echo "Starting Transformer Seq2Seq Training"
echo "=========================================="
echo "Dataset: ${DATASET} (${DATASET_CONFIG})"
echo "Batch Size: ${BATCH_SIZE}"
echo "Epochs: ${EPOCHS}"
echo "Model Dimension: ${D_MODEL}"
echo "Number of Heads: ${NUM_HEADS}"
echo "Encoder Layers: ${NUM_ENCODER_LAYERS}"
echo "Decoder Layers: ${NUM_DECODER_LAYERS}"
echo "Seed: ${SEED}"
echo "=========================================="

# Run training
python src/train.py \
    --dataset ${DATASET} \
    --dataset_config ${DATASET_CONFIG} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --d_model ${D_MODEL} \
    --num_heads ${NUM_HEADS} \
    --num_encoder_layers ${NUM_ENCODER_LAYERS} \
    --num_decoder_layers ${NUM_DECODER_LAYERS} \
    --d_ff ${D_FF} \
    --max_len ${MAX_LEN} \
    --dropout ${DROPOUT} \
    --lr ${LR} \
    --warmup_steps ${WARMUP_STEPS} \
    --seed ${SEED}

echo "=========================================="
echo "Training Completed!"
echo "=========================================="
