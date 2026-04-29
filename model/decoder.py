import torch
import torch.nn as nn


from .positional_encoding import PositionalEncoding
from .decoder_block import DecoderBlock


class Decoder(nn.Module):
    """
    Transformer Decoder

    该模块包含词嵌入层、位置编码层、多个Decoder Block和输出层，用于解码目标序列。

    Args:
        vocab_size: 词汇表大小
        d_model: 输入特征维度
        num_heads: 注意力头数
        num_layers: 解码器层数
        max_seq_len: 最大序列长度
        dropout: Dropout概率
        d_ff: 前馈网络隐藏层维度
        device: 设备（'cpu'或'cuda'）
    """
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 max_seq_len=5000, dropout=0.1, d_ff=2048, device='cpu'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, device)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, dropout, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.device = device
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, self_mask,
                cross_mask,self_caches=None, cross_caches=None,start_pos=0):
        """
        Decoder前向传播

        Args:
            x: 解码器输入，形状为 [batch_size, seq_len]
            encoder_output: 编码器输出，形状为 [batch_size, src_seq_len, d_model]
            self_mask: 解码器自注意力掩码，形状为 [batch_size, 1, seq_len, seq_len]
            cross_mask: 交叉注意力掩码，形状为 [batch_size, 1, seq_len, src_seq_len]

        Returns:
            解码器输出，形状为 [batch_size, seq_len, vocab_size]
        """
        # 词嵌入
        x = self.embedding(x)
        # 添加位置编码
        x = self.positional_encoding(x,start_pos=start_pos)

        # 通过Decoder Block
        for i,layer in enumerate(self.layers):
            x = layer(x, encoder_output, self_mask, cross_mask,
            self_cache = self_caches[i] if self_caches else None,
            cross_cache = cross_caches[i] if cross_caches else None
            )

        # 最后一个LayerNorm
        x = self.norm(x)
        # 输出层
        x = self.linear(x)

        return x