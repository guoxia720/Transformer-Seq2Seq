import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:, :x.size(1), :]#序列特征+对应长度的位置编码
        return self.dropout(x) #加dropout防止过拟合


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""

    #初始化
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: [batch_size, num_heads, seq_len, d_k]
            mask: [batch_size, 1, seq_len, seq_len] or [batch_size, 1, 1, seq_len]
        Returns:
            output: [batch_size, num_heads, seq_len, d_k]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        # 计算注意力分数：Q·K^T / sqrt(d_k)（缩放避免分数过大导致softmax饱和）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用mask：将需要屏蔽的位置（如Pad、未来词）设为-1e9，softmax后权重趋近于0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重：softmax归一化（每行和为1）
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和：权重×V，得到单个头的输出
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: [batch_size, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)

        # Q/K/V 先通过线性层投影（维度不变，仍是d_model）
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Adjust mask dimensions for multi-head
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]

        # Apply attention
        x, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_o(x)

        return output, attention_weights


class PositionWiseFeedForward(nn.Module):
    """position-wise FFN"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        # 线性变换→ReLU激活→dropout→线性变换（还原维度）
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single Transformer Encoder Layer"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            src_mask: [batch_size, seq_len, seq_len]
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class DecoderLayer(nn.Module):
    """Single Transformer Decoder Layer"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        Args:
            x: [batch_size, tgt_seq_len, d_model]
            encoder_output: [batch_size, src_seq_len, d_model]
            tgt_mask: [batch_size, tgt_seq_len, tgt_seq_len] (causal mask)
            src_mask: [batch_size, tgt_seq_len, src_seq_len] (padding mask)
        """
        # Masked self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Encoder-decoder attention with residual connection and layer norm
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x


class TransformerEncoder(nn.Module):
    """Stack of Encoder Layers"""

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, src_mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            src_mask: [batch_size, seq_len, seq_len]
        """
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class TransformerDecoder(nn.Module):
    """Stack of Decoder Layers"""

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        Args:
            x: [batch_size, tgt_seq_len, d_model]
            encoder_output: [batch_size, src_seq_len, d_model]
            tgt_mask: [batch_size, tgt_seq_len, tgt_seq_len]
            src_mask: [batch_size, tgt_seq_len, src_seq_len]
        """
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return x


class Transformer(nn.Module):
    """完整Seq2Seq Transformer模型"""

    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            d_model=512,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_ff=2048,
            max_seq_len=5000,
            dropout=0.1,
            pad_idx=0
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder and Decoder
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Args:
            src: [batch_size, src_seq_len]
            tgt: [batch_size, tgt_seq_len]
            src_mask: [batch_size, src_seq_len, src_seq_len]
            tgt_mask: [batch_size, tgt_seq_len, tgt_seq_len]
            memory_mask: [batch_size, tgt_seq_len, src_seq_len]
        Returns:
            output: [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # Encode source sequence
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoding(src_embedded)
        encoder_output = self.encoder(src_embedded, src_mask)

        # Decode target sequence
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded)
        decoder_output = self.decoder(tgt_embedded, encoder_output, tgt_mask, memory_mask)

        # Project to vocabulary
        output = self.output_projection(decoder_output)

        return output

    def encode(self, src, src_mask=None):
        """Encode source sequence only"""
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoding(src_embedded)
        return self.encoder(src_embedded, src_mask)

    def decode(self, tgt, encoder_output, tgt_mask=None, memory_mask=None):
        """Decode target sequence given encoder output"""
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded)
        decoder_output = self.decoder(tgt_embedded, encoder_output, tgt_mask, memory_mask)
        return self.output_projection(decoder_output)