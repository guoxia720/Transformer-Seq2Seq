import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import argparse
import os
import time
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

from model import Transformer
from data import get_dataloaders, make_collate
from utils import (
    set_seed, create_tgt_mask, create_src_mask, create_memory_mask,
    save_checkpoint, load_checkpoint, count_parameters, compute_bleu
)


def get_lr_scheduler(optimizer, d_model, warmup_steps):
    def lr_lambda(step):
        if step == 0:
            step = 1
        scale = d_model ** (-0.5)
        return scale * min(step ** (-0.5), step * (warmup_steps ** (-1.5)))  # 去掉最后一个乘法
    return LambdaLR(optimizer, lr_lambda)



def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, pad_idx, clip_grad=1.0):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = create_src_mask(src, pad_idx).to(device)
        tgt_mask = create_tgt_mask(tgt_input, pad_idx).to(device)
        memory_mask = create_memory_mask(src, tgt_input, pad_idx).to(device)

        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask, memory_mask)

        output = output.contiguous().view(-1, output.size(-1))
        tgt_output = tgt_output.contiguous().view(-1)
        loss = criterion(output, tgt_output)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.6f}"
        })

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, pad_idx, tgt_tokenizer):
    """Validate the model + 计算BLEU分数（修复Token处理逻辑）"""
    model.eval()
    total_loss = 0
    all_pred_texts = []  # 存储解码后的预测文本（不是Token ID）
    all_ref_texts = []   # 存储解码后的参考文本（不是Token ID）

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch in progress_bar:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = create_src_mask(src, pad_idx).to(device)
            tgt_mask = create_tgt_mask(tgt_input, pad_idx).to(device)
            memory_mask = create_memory_mask(src, tgt_input, pad_idx).to(device)

            output = model(src, tgt_input, src_mask, tgt_mask, memory_mask)
            output_flat = output.contiguous().view(-1, output.size(-1))
            tgt_output_flat = tgt_output.contiguous().view(-1)
            loss = criterion(output_flat, tgt_output_flat)

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

            # 核心修复：解码Token ID → 文本，同时过滤无效Token
            pred_tokens = output.argmax(dim=-1).cpu()  # (batch_size, tgt_seq_len)
            tgt_output_tokens = tgt_output.cpu()       # (batch_size, tgt_seq_len)

            # 遍历每个样本，解码并过滤Pad/EOS
            for pred in pred_tokens:
                # 过滤Pad Token（pad_idx=58100）和EOS Token（tgt_tokenizer.eos_token_id）
                valid_pred = pred[(pred != pad_idx) & (pred != tgt_tokenizer.eos_token_id)]
                # 解码成文本
                pred_text = tgt_tokenizer.decode(valid_pred, skip_special_tokens=True)
                all_pred_texts.append(pred_text)

            for ref in tgt_output_tokens:
                valid_ref = ref[(ref != pad_idx) & (ref != tgt_tokenizer.eos_token_id)]
                ref_text = tgt_tokenizer.decode(valid_ref, skip_special_tokens=True)
                all_ref_texts.append(ref_text)

    avg_loss = total_loss / len(dataloader)
    # 直接传入文本计算BLEU（无需再在compute_bleu里分词，避免重复处理）
    val_bleu = compute_bleu(all_pred_texts, all_ref_texts)  # 需同步修改compute_bleu参数
    return avg_loss, val_bleu


def plot_losses(train_losses, val_losses, save_path='results/loss_curve.png'):
    """Plot training and validation losses"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss curve saved to {save_path}")


def save_metrics(train_losses, val_losses, val_bleus, save_path='results/metrics_example.csv'):
    """Save training metrics (Loss + BLEU) to CSV"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Validation BLEU'])  # 新增BLEU列
        for epoch, (train_loss, val_loss, val_bleu) in enumerate(zip(train_losses, val_losses, val_bleus), 1):
            writer.writerow([epoch, train_loss, val_loss, val_bleu])
    print(f"Metrics saved to {save_path}")


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, src_tokenizer, tgt_tokenizer, src_vocab_size, tgt_vocab_size, pad_idx = get_dataloaders(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=args.num_workers,
    )

    # 设置 collate_fn（Windows 多进程安全）
    train_loader.collate_fn = make_collate(pad_idx)
    val_loader.collate_fn = make_collate(pad_idx)

    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {tgt_vocab_size}")
    print(f"Padding index: {pad_idx}")

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_len,
        dropout=args.dropout,
        pad_idx=pad_idx
    ).to(device)

    print(f"Model created with {count_parameters(model):,} trainable parameters")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=args.weight_decay
    )
    scheduler = get_lr_scheduler(optimizer, args.d_model, args.warmup_steps)
    #scheduler = None

    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.checkpoint_path, device)
        start_epoch += 1

    train_losses = []
    val_losses = []
    val_bleus = []  # 存储每个epoch的BLEU分数
    best_val_loss = float('inf')
    best_val_bleu = 0.0  # 跟踪最佳BLEU分数

    print("\n" + "=" * 50)
    print("Starting Training...")
    print("=" * 50 + "\n")

#测试
    batch = next(iter(train_loader))
    src = batch['src']
    tgt = batch['tgt']

    # 测试 mask 生成
    #from utils import create_tgt_mask
    #tgt_mask = create_tgt_mask(tgt)
    #print(tgt_mask[0].int())  # 打印第一个样本的 mask 矩阵（下三角）

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 criterion, device, pad_idx, args.clip_grad)
        train_losses.append(train_loss)

        val_loss, val_bleu = validate(model, val_loader, criterion, device, pad_idx, tgt_tokenizer)
        val_losses.append(val_loss)
        val_bleus.append(val_bleu)  # 保存BLEU分数

        epoch_time = time.time() - epoch_start_time

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val BLEU:   {val_bleu:.2f}分")  # 直观显示翻译质量
        print(f"  Time:       {epoch_time:.2f}s")

        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_bleu = val_bleu  # 更新最佳BLEU
            save_checkpoint(model, optimizer, epoch, val_loss, "checkpoints/best_model.pt")
            print(f"  ✓ New best model saved (Val Loss: {val_loss:.4f}, Val BLEU: {val_bleu:.2f}分)")

    print("\n" + "=" * 50)
    print("Training Completed!")
    print("=" * 50 + "\n")

    save_checkpoint(model, optimizer, args.epochs - 1, val_losses[-1], "checkpoints/final_model.pt")
    plot_losses(train_losses, val_losses)
    save_metrics(train_losses, val_losses, val_bleus)

    print(f"\nBest Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation BLEU: {best_val_bleu:.2f}分")  # 输出最佳BLEU


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Seq2Seq Model")

    parser.add_argument('--dataset', type=str, default='iwslt2017', help='Dataset name')
    parser.add_argument('--dataset_config', type=str, default='iwslt2017-de-en', help='Dataset configuration')

    parser.add_argument('--max_len', type=int, default=64, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')

    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=4, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=3000, help='Warmup steps for learning rate')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/best_model.pt', help='Checkpoint path')

    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    main(args)
