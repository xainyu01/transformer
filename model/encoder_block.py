import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward


class EncoderBlock(nn.Module):
    """
    Transformer Encoder Block

    该模块包含多头自注意力机制和前馈神经网络，是Transformer编码器的基本单元。

    Args:
        d_model: 输入特征维度
        num_heads: 注意力头数
        dropout: Dropout概率，默认0.1
        d_ff: 前馈网络隐藏层维度，默认2048
    """
    def __init__(self, d_model, num_heads, dropout=0.1, d_ff=2048):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads, d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        """
        Encoder Block前向传播

        Args:
            x: 输入张量，形状为 [batch_size, seq_len, d_model]
            mask: 自注意力掩码，形状为 [batch_size, 1, seq_len, seq_len]
        
        Returns:
            输出张量，形状为 [batch_size, seq_len, d_model]
        """
        # 自注意力
        attn_output, _ = self.attn(x, x, x, mask)
        # 残差连接 + LayerNorm
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # 前馈神经网络
        ff_output = self.ff(x)
        # 残差连接 + LayerNorm
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x