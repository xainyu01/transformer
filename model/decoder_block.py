import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward


class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block

    该模块包含自注意力、交叉注意力和前馈神经网络，是Transformer解码器的基本单元。

    Args:
        d_model: 输入特征维度
        num_heads: 注意力头数
        dropout: Dropout概率，默认0.1
        d_ff: 前馈网络隐藏层维度，默认2048
    """
    def __init__(self, d_model, num_heads, dropout=0.1, d_ff=2048):
        super().__init__()
        self.self_attn = MultiHeadAttention(num_heads, d_model, dropout)
        self.cross_attn = MultiHeadAttention(num_heads, d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, self_mask,
                cross_mask,self_cache=None,cross_cache=None):
        """
        Decoder Block前向传播

        Args:
            x: 解码器输入，形状为 [batch_size, seq_len, d_model]
            encoder_output: 编码器输出，形状为 [batch_size, src_seq_len, d_model]
            self_mask: 解码器自注意力掩码，形状为 [batch_size, 1, seq_len, seq_len]
            cross_mask: 交叉注意力掩码，形状为 [batch_size, 1, seq_len, src_seq_len]
        
        Returns:
            输出张量，形状为 [batch_size, seq_len, d_model]
        """
        # 自注意力
        self_attn_output, _ = self.self_attn(x, x, x, self_mask,kv_cache=self_cache)
        x = x + self.dropout(self_attn_output)
        x = self.norm1(x)
        
        # 交叉注意力
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, cross_mask,kv_cache=cross_cache)
        x = x + self.dropout(cross_attn_output)
        x = self.norm2(x)
        
        # 前馈网络
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)
        
        return x