import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    """
    位置编码模块，支持批量输入和单步查询。
    """
    def __init__(self, d_model, max_seq_len=5000, device='cpu'):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = device

        # 创建位置编码矩阵
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 添加 batch 维度 → (1, max_seq_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x, start_pos=0):
        """
        训练时：x 形状 (batch, seq_len, d_model)，将对应位置编码相加
        start_pos: 当 seq_len > 1 时，起始位置索引（用于部分序列的场景）
                   通常训练时不传，默认为 0。
        """
        # 取出需要的切片：(1, seq_len, d_model)
        seq_len = x.size(1)
        pos_enc = self.pe[:, start_pos:start_pos+seq_len, :].to(x.device)
        return x + pos_enc

    def get_position(self, pos, device=None):
        """
        返回单个位置的编码 (1, 1, d_model)
        pos: 整数，位置索引
        """
        if device is None:
            device = self.pe.device
        return self.pe[:, pos:pos+1, :].to(device)