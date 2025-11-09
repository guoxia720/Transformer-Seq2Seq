import torch
import torch.nn.functional as F
from train_test import Transformer  # 导入你的Transformer模型
from data import get_dataloaders  # 复用data.py的tokenizer配置
import os

# ==============================================================================
# ===== 已根据你的路径配置完成，直接运行即可！ =====
# ==============================================================================
# 1. 模型路径（相对路径，适配你的目录结构）
MODEL_PATH = "./checkpoints/final_model.pt"  # 你的模型路径：src/checkpoints/final_model.pt

# 2. 测试用的德语文本（可直接修改成你想测试的句子）
TEST_SRC_TEXT = "Heute ist ein wunderschöner Tag, und ich freue mich darauf, ins Museum zu gehen."

# 3. 模型参数（用train.py默认值，没修改过就不用动）
D_MODEL = 512
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
D_FF = 2048
MAX_SEQ_LEN = 5000
DROPOUT = 0.1
MAX_GEN_LEN = 128  # 最大生成长度


# ==============================================================================
# ===== 核心逻辑（无需修改） =====
# ==============================================================================
def create_causal_mask(seq_len, device):
    """创建因果掩码（防止解码时看到未来词）"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask == 0


def create_padding_mask(seq, pad_idx, device):
    """创建padding掩码（屏蔽pad token）"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def translate_text(
        model,
        src_text,
        src_tokenizer,
        tgt_tokenizer,
        pad_idx,
        max_gen_len=128,
        device="cuda" if torch.cuda.is_available() else "cpu"
):
    """单句翻译：德语→英语"""
    # 1. 源文本预处理（和训练时编码逻辑一致）
    src_tokens = src_tokenizer.encode(
        src_text,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    src_mask = create_padding_mask(src_tokens, pad_idx, device)

    # 2. 编码源文本（禁用梯度，加速推理）
    model.eval()
    with torch.no_grad():
        encoder_output = model.encode(src_tokens, src_mask)

    # 3. 初始化目标序列（以bos token开头）
    tgt_bos_token = tgt_tokenizer.bos_token_id
    tgt_eos_token = tgt_tokenizer.eos_token_id
    generated_tgt = torch.tensor([[tgt_bos_token]], device=device)

    # 4. 自回归生成（逐词预测）
    for _ in range(max_gen_len - 1):
        tgt_seq_len = generated_tgt.size(1)
        # 因果掩码：防止看到未来词
        tgt_mask = create_causal_mask(tgt_seq_len, device).unsqueeze(0)
        # padding掩码：屏蔽pad token
        tgt_pad_mask = create_padding_mask(generated_tgt, pad_idx, device)

        with torch.no_grad():
            # 解码预测下一个词
            output = model.decode(
                tgt=generated_tgt,
                encoder_output=encoder_output,
                tgt_mask=tgt_mask,
                memory_mask=src_mask
            )
            # 投影到词汇表，选概率最高的词（贪心搜索）
            next_token_logits = model.output_projection(output[:, -1, :])
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # 追加到目标序列
        generated_tgt = torch.cat([generated_tgt, next_token_id], dim=-1)
        # 生成eos token则终止
        if next_token_id.item() == tgt_eos_token:
            break

    # 5. 解码为自然文本（跳过特殊token）
    translated_text = tgt_tokenizer.decode(
        generated_tgt.squeeze().cpu().numpy(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    return translated_text


def main():
    # 检查模型文件是否存在（提前报错，避免后续麻烦）
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"模型文件未找到！请确认路径：{os.path.abspath(MODEL_PATH)}\n"
            f"当前目录：{os.getcwd()}\n"
            f"预期目录结构：Transformer-Seq2Seq/src/ → 包含 infer.py 和 checkpoints/ 文件夹"
        )

    # 1. 加载tokenizer（复用data.py配置，确保编码一致）
    print("✅ 加载tokenizer...")
    _, _, src_tokenizer, tgt_tokenizer, src_vocab_size, tgt_vocab_size, pad_idx = get_dataloaders(
        batch_size=32,
        max_len=MAX_SEQ_LEN,
        num_workers=0
    )

    # 2. 初始化模型（参数与train.py完全对齐）
    print("✅ 初始化模型...")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
        pad_idx=pad_idx
    )

    # 3. 加载模型权重（关键修改：只取model_state_dict部分）
    print(f"✅ 加载模型权重：{MODEL_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 运行设备：{device}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # 判断是否是完整训练状态（含epoch等），如果是则取model_state_dict，否则直接加载
    if "model_state_dict" in checkpoint.keys():
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()  # 切换到评估模式，禁用dropout

    # 4. 执行翻译并输出结果
    print(f"\n=== 🚀 翻译结果 ===")
    print(f"输入德语：{TEST_SRC_TEXT}")
    translated_text = translate_text(
        model=model,
        src_text=TEST_SRC_TEXT,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        pad_idx=pad_idx,
        max_gen_len=MAX_GEN_LEN,
        device=device
    )
    print(f"生成英语：{translated_text}")
    print(f"=== 🎉 推理完成 ===")


if __name__ == "__main__":
    main()