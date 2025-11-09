import torch
import numpy as np
import random
import os


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_padding_mask(seq, pad_idx=0):
    """
    Create padding mask for sequences
    Args:
        seq: [batch_size, seq_len]
        pad_idx: padding token index
    Returns:
        mask: [batch_size, 1, 1, seq_len]
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def create_causal_mask(size, device):
    """
    Create causal (look-ahead) mask for decoder
    Args:
        size: sequence length
        device: torch device
    Returns:
        mask: [1, size, size] - lower triangular matrix
    """
    mask = torch.tril(torch.ones(size, size, device=device)).bool()
    return mask.unsqueeze(0)


def create_tgt_mask(tgt, pad_idx=0):
    """
    Create combined padding + causal mask for target sequence
    Args:
        tgt: [batch_size, tgt_seq_len]
        pad_idx: padding token index
    Returns:
        mask: [batch_size, tgt_seq_len, tgt_seq_len]
    """
    batch_size, tgt_len = tgt.size()
    device = tgt.device

    # Padding mask: [batch_size, 1, tgt_len]
    tgt_padding_mask = (tgt != pad_idx).unsqueeze(1)

    # Causal mask: [1, tgt_len, tgt_len]
    causal_mask = create_causal_mask(tgt_len, device)

    # Combine masks: [batch_size, tgt_len, tgt_len]
    tgt_mask = tgt_padding_mask & causal_mask

    return tgt_mask


def create_src_mask(src, pad_idx=0):
    """
    Create padding mask for source sequence
    Args:
        src: [batch_size, src_seq_len]
        pad_idx: padding token index
    Returns:
        mask: [batch_size, 1, src_seq_len]
    """
    src_mask = (src != pad_idx).unsqueeze(1)
    return src_mask


def create_memory_mask(src, tgt, pad_idx=0):
    """
    Create mask for encoder-decoder attention (memory mask)
    Args:
        src: [batch_size, src_seq_len]
        tgt: [batch_size, tgt_seq_len]
        pad_idx: padding token index
    Returns:
        mask: [batch_size, tgt_seq_len, src_seq_len]
    """
    # Source padding mask: [batch_size, 1, src_seq_len]
    src_padding_mask = (src != pad_idx).unsqueeze(1)

    # Expand to match target sequence length
    tgt_len = tgt.size(1)
    memory_mask = src_padding_mask.expand(-1, tgt_len, -1)

    return memory_mask


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {path} (Epoch: {epoch}, Loss: {loss:.4f})")
    return epoch, loss


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ========== 新增：BLEU分数计算函数（用于量化翻译质量） ==========
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# 自动下载必要的nltk资源（首次运行会下载，后续无需重复）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize


def compute_bleu(pred_texts, ref_texts):
    """
    修复：同步过滤空白文本，确保 pred 和 ref 数量一致
    添加：平滑函数，解决 0 分警告
    """
    # 1. 同步过滤：只保留 pred 和 ref 都非空的样本（避免数量不一致）
    filtered_pairs = []
    for pred, ref in zip(pred_texts, ref_texts):
        # 过滤空白字符串（strip() 去掉前后空格，避免纯空格文本）
        if pred.strip() and ref.strip():
            filtered_pairs.append((pred, ref))

    # 2. 若过滤后无有效样本，直接返回 0.0（避免后续报错）
    if not filtered_pairs:
        return 0.0

    # 3. 拆分过滤后的 pred 和 ref（此时两者长度完全一致）
    filtered_preds, filtered_refs = zip(*filtered_pairs)

    # 4. 转换为 BLEU 要求的格式：
    # - refs：列表的列表（每个样本的参考文本是一个列表，支持多个参考）
    # - preds：列表（每个样本的预测文本是一个列表）
    ref_tokens = [[word_tokenize(ref.lower())] for ref in filtered_refs]  # 注意外层嵌套
    pred_tokens = [word_tokenize(pred.lower()) for pred in filtered_preds]

    # 5. 添加平滑函数（解决“无 n-gram 重叠”的警告，避免 BLEU 直接为 0）
    smoothing = SmoothingFunction().method1  # 简单平滑，适合低质量预测

    # 6. 计算 BLEU（权重可调整，4-gram 权重平均）
    bleu_score = corpus_bleu(
        ref_tokens,
        pred_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing  # 启用平滑
    )

    # 转为百分比（0-100，更直观）
    return bleu_score * 100