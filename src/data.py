import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DownloadMode
from transformers import MarianTokenizer
from torch.nn.utils.rnn import pad_sequence
import os

class TranslationDataset(Dataset):
    """Pre-encoded dataset for machine translation tasks"""

    def __init__(self, data, src_tokenizer, tgt_tokenizer, max_len=128):
        """
        data: datasets.Dataset 对象
        src_tokenizer, tgt_tokenizer: HuggingFace tokenizer
        max_len: 最大序列长度
        """
        self.max_len = max_len
        self.pad_idx = tgt_tokenizer.pad_token_id

        # 预先 encode 所有文本
        print("Pre-encoding dataset...")
        self.src_encoded = []
        self.tgt_encoded = []
        for item in data:
            src_text = item['translation']['de']
            tgt_text = item['translation']['en']

            src_tokens = src_tokenizer.encode(src_text, max_length=max_len, truncation=True)
            tgt_tokens = tgt_tokenizer.encode(tgt_text, max_length=max_len, truncation=True)

            self.src_encoded.append(torch.tensor(src_tokens, dtype=torch.long))
            self.tgt_encoded.append(torch.tensor(tgt_tokens, dtype=torch.long))
        print(f"Finished encoding {len(self.src_encoded)} examples.")

    def __len__(self):
        return len(self.src_encoded)

    def __getitem__(self, idx):
        return {
            'src': self.src_encoded[idx],
            'tgt': self.tgt_encoded[idx]
        }


def collate_fn(batch, pad_idx):
    """Custom collate function for batching"""
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]

    # Pad sequences
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)

    return {
        'src': src_padded,
        'tgt': tgt_padded
    }
def make_collate(pad_idx):
    """
    返回一个全局 collate 函数，避免 lambda 导致 Windows 多进程 DataLoader 出错
    """
    def _collate(batch):
        return collate_fn(batch, pad_idx)
    return _collate


def get_dataloaders(dataset_name='iwslt2017', dataset_config='iwslt2017-de-en',
                    batch_size=32, max_len=128, num_workers=0, cache_dir='./dataset'):
    """Return train/validation DataLoaders"""

    print(f"Loading dataset: {dataset_name} ({dataset_config})...")
    data_dir = os.path.join(os.path.dirname(__file__), "dataset", "iwslt2017", "iwslt2017-de-en", "1.0.0")
    # 验证本地数据集目录是否存在（提前报错，避免远程尝试）
    required_files = [
        os.path.join(data_dir, "iwslt2017-train.arrow"),
        os.path.join(data_dir, "iwslt2017-validation.arrow"),
        os.path.join(data_dir, "dataset_info.json")
    ]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"本地文件不存在：{file}\n请确认数据集已正确下载到该目录")
    dataset = load_dataset(
        "arrow",  # 格式指定为 Arrow
        data_files={
            "train": os.path.join(data_dir, "iwslt2017-train.arrow"),
            "validation": os.path.join(data_dir, "iwslt2017-validation.arrow"),
            "test": os.path.join(data_dir, "iwslt2017-test.arrow")  # 测试集可选
        },
        split={
            "train": "train",  # 仅取前1000个训练样本（原完整206k，改后快速跑通）
            "validation": "validation"  # 仅取前200个验证样本（原1000，进一步提速）
        },

        cache_dir=cache_dir,
        trust_remote_code=False,
        #download_mode=DownloadMode.FORCE_LOCAL,  # 强制本地加载，不联网

    )

    print("Loading tokenizers...")
    src_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en", cache_dir=cache_dir, local_files_only=True)
    tgt_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en", cache_dir=cache_dir, local_files_only=True)

    pad_idx = tgt_tokenizer.pad_token_id

    train_dataset = TranslationDataset(dataset['train'], src_tokenizer, tgt_tokenizer, max_len)
    val_dataset = TranslationDataset(dataset['validation'], src_tokenizer, tgt_tokenizer, max_len)

    # 注意这里不再使用 lambda，直接传参
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, src_tokenizer, tgt_tokenizer, src_tokenizer.vocab_size, tgt_tokenizer.vocab_size, pad_idx
