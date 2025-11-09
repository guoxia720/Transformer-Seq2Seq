import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import os
import time
import matplotlib.pyplot as plt
import numpy as np

# ===================== 字体配置（简化为英文，避免中文兼容问题） =====================
def setup_font():
    # 使用默认英文字体，彻底消除中文缺失警告
    plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ Font configured successfully (using default English font)")

setup_font()

# 导入核心模块
from model import Transformer
from data import get_dataloaders, make_collate
from utils import (
    set_seed, create_tgt_mask, create_src_mask, create_memory_mask,
    count_parameters, compute_bleu
)

# ===================== 简化超参配置（固定d_model=384，仅测试3个num_heads） =====================
TEST_NUM_HEADS = [2, 4, 8]  # 仅测试这3个注意力头数
FIXED_PARAMS = {
    "d_model": 512,  # 固定模型维度为384
    "batch_size": 128,
    "max_len": 64,
    "num_workers": 0,
    "num_encoder_layers": 4,
    "num_decoder_layers": 4,
    "d_ff": lambda d: d * 4,
    "epochs": 10,
    "lr": 1.0,
    "weight_decay": 1e-5,
    "warmup_steps": 3000,
    "clip_grad": 1.0,
    "seed": 42,
    "dataset": "iwslt2017",
    "dataset_config": "iwslt2017-de-en",
    "cache_dir": "./dataset",
    "dropout": 0.1
}

# 结果保存路径（修改目录名+文件名，与之前区分）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "num_heads_comparison")  # 目录名改为num_heads相关
os.makedirs(RESULT_DIR, exist_ok=True)

# ===================== 核心训练/验证逻辑（无修改，复用） =====================
def get_lr_scheduler(optimizer, d_model, warmup_steps):
    def lr_lambda(step):
        step = step if step > 0 else 1
        scale = d_model ** (-0.5)
        return scale * min(step ** (-0.5), step * (warmup_steps ** (-1.5)))
    return LambdaLR(optimizer, lr_lambda)

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, pad_idx, clip_grad=1.0):
    model.train()
    total_loss = 0
    for batch in dataloader:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = create_src_mask(src, pad_idx).to(device)
        tgt_mask = create_tgt_mask(tgt_input, pad_idx).to(device)
        memory_mask = create_memory_mask(src, tgt_input, pad_idx).to(device)

        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask, memory_mask)
        loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, pad_idx, tgt_tokenizer):
    model.eval()
    total_loss = 0
    all_pred_texts, all_ref_texts = [], []
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = create_src_mask(src, pad_idx).to(device)
            tgt_mask = create_tgt_mask(tgt_input, pad_idx).to(device)
            memory_mask = create_memory_mask(src, tgt_input, pad_idx).to(device)

            output = model(src, tgt_input, src_mask, tgt_mask, memory_mask)
            loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
            total_loss += loss.item()

            pred_tokens = output.argmax(dim=-1).cpu()
            for pred in pred_tokens:
                valid_pred = pred[(pred != pad_idx) & (pred != tgt_tokenizer.eos_token_id)]
                all_pred_texts.append(tgt_tokenizer.decode(valid_pred, skip_special_tokens=True))
            for ref in tgt_output.cpu():
                valid_ref = ref[(ref != pad_idx) & (ref != tgt_tokenizer.eos_token_id)]
                all_ref_texts.append(tgt_tokenizer.decode(valid_ref, skip_special_tokens=True))
    return total_loss / len(dataloader), compute_bleu(all_pred_texts, all_ref_texts)

# ===================== 单组训练（接收num_heads参数，返回每轮指标） =====================
def train_single_num_head(num_head):
    params = FIXED_PARAMS.copy()
    params["num_heads"] = num_head
    params["d_ff"] = params["d_ff"](params["d_model"])  # 使用固定的d_model=384
    set_seed(params["seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'=' * 60}")
    print(f"当前训练：d_model={params['d_model']}（固定）, num_heads={num_head}, dropout={params['dropout']}")
    print(f"{'=' * 60}")

    # 加载数据和模型
    train_loader, val_loader, _, tgt_tokenizer, src_vocab_size, tgt_vocab_size, pad_idx = get_dataloaders(
        dataset_name=params["dataset"],
        dataset_config=params["dataset_config"],
        batch_size=params["batch_size"],
        max_len=params["max_len"],
        num_workers=params["num_workers"],
        cache_dir=params["cache_dir"]
    )
    train_loader.collate_fn = make_collate(pad_idx)
    val_loader.collate_fn = make_collate(pad_idx)

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=params["d_model"],
        num_heads=params["num_heads"],
        num_encoder_layers=params["num_encoder_layers"],
        num_decoder_layers=params["num_decoder_layers"],
        d_ff=params["d_ff"],
        max_seq_len=params["max_len"],
        dropout=params["dropout"],
        pad_idx=pad_idx
    ).to(device)
    print(f"模型参数量：{count_parameters(model):,}")

    # 优化器和调度器
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=params["lr"], betas=(0.9, 0.98), eps=1e-8, weight_decay=params["weight_decay"])
    scheduler = get_lr_scheduler(optimizer, params["d_model"], params["warmup_steps"])

    # 记录每轮指标
    train_losses, val_losses, val_bleus = [], [], []
    for epoch in range(params["epochs"]):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, pad_idx, params["clip_grad"])
        val_loss, val_bleu = validate(model, val_loader, criterion, device, pad_idx, tgt_tokenizer)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_bleus.append(val_bleu)

        # 终端打印训练过程（保持原格式）
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val BLEU: {val_bleu:.2f} | Time: {epoch_time:.1f}s")

    return {
        "num_heads": num_head,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_bleus": val_bleus
    }

# ===================== 主函数（训练3个num_heads + 生成2个对比曲线） =====================
def main():
    print(f"开始测试num_heads组合：{TEST_NUM_HEADS}（d_model固定为384）")
    print(f"固定dropout值：{FIXED_PARAMS['dropout']}")

    # 训练所有num_heads，收集指标
    all_results = []
    for num_head in TEST_NUM_HEADS:
        result = train_single_num_head(num_head)
        all_results.append(result)

    # 生成并保存【val_loss对比曲线】（文件名含num_heads，与之前区分）
    plt.figure(figsize=(10, 6))
    for result in all_results:
        num_head = result["num_heads"]
        plt.plot(
            range(1, FIXED_PARAMS["epochs"] + 1),
            result["val_losses"],
            marker='o',
            label=f'num_heads={num_head}'
        )
    plt.xlabel('Epoch')  # 英文替换
    plt.ylabel('Validation Loss')  # 英文替换
    plt.title('Validation Loss Comparison of Different Number of Attention Heads (d_model=384, dropout=0.1)')  # 英文替换
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    val_loss_path = os.path.join(RESULT_DIR, "val_loss_comparison_num_heads.pdf")  # 保留原文件名标识
    plt.savefig(val_loss_path, bbox_inches='tight', dpi=None)  # 删除fonttype=42参数
    plt.close()
    print(f"\n✅ Validation loss comparison curve saved to: {val_loss_path}")

    # 生成并保存【BLEU对比曲线】（文件名含num_heads，与之前区分）
    plt.figure(figsize=(10, 6))
    for result in all_results:
        num_head = result["num_heads"]
        plt.plot(
            range(1, FIXED_PARAMS["epochs"] + 1),
            result["val_bleus"],
            marker='s',
            label=f'num_heads={num_head}'
        )
    plt.xlabel('Epoch')  # 英文替换
    plt.ylabel('BLEU Score')  # 英文替换
    plt.title('BLEU Score Comparison of Different Number of Attention Heads (d_model=384, dropout=0.1)')  # 英文替换
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    bleu_path = os.path.join(RESULT_DIR, "bleu_comparison_num_heads.pdf")  # 保留原文件名标识
    plt.savefig(bleu_path, bbox_inches='tight', dpi=None)  # 删除fonttype=42参数
    plt.close()
    print(f" BLEU score comparison curve saved to: {bleu_path}")

    print(f"\n{'=' * 60}")
    print("All num_heads training completed! Generated 2 comparison curve PDF files.")
    print(f"File path: {RESULT_DIR}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()