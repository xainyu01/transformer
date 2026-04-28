import torch
import torch.nn as nn


from .positional_encoding import PositionalEncoding
from .encoder_block import EncoderBlock


class Encoder(nn.Module):
    """
    Transformer Encoder

    该模块包含词嵌入层、位置编码层和多个Encoder Block，用于编码输入序列。

    Args:
        vocab_size: 词汇表大小
        d_model: 输入特征维度
        num_heads: 注意力头数
        num_layers: 编码器层数
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
            EncoderBlock(d_model, num_heads, dropout, d_ff) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.device = device
    
    def forward(self, x, mask):
        """
        Encoder前向传播

        Args:
            x: 输入序列，形状为 [batch_size, seq_len]
            mask: 自注意力掩码，形状为 [batch_size, 1, seq_len, seq_len]
        
        Returns:
            编码后的输出，形状为 [batch_size, seq_len, d_model]
        """
        # 词嵌入
        x = self.embedding(x)
        # 添加位置编码
        x = self.positional_encoding(x)
        
        # 通过Encoder Block
        for layer in self.layers:
            x = layer(x, mask)
        
        # 最后一个LayerNorm
        x = self.norm(x)
        
        return x