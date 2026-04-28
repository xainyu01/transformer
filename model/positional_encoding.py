import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    """
    位置编码模块，用于为输入添加位置信息

    该模块使用正弦和余弦函数生成位置编码，解决Transformer无法处理序列顺序的问题。

    Args:
        d_model: 特征维度
        max_seq_len: 最大序列长度，默认5000
        device: 设备（'cpu'或'cuda'）
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
        
        # 添加batch维度
        self.pe = pe.unsqueeze(0).to(device)
        
    def forward(self, x):
        """
        将位置编码添加到输入中

        Args:
            x: 输入张量，形状为 [batch_size, seq_len, d_model]
        
        Returns:
            添加了位置编码的输入，形状为 [batch_size, seq_len, d_model]
        """
        # 位置编码的形状为 [1, max_seq_len, d_model]
        # 输入x的形状为 [batch_size, seq_len, d_model]
        # 我们只需要前seq_len个位置编码
        x = x + self.pe[:, :x.size(1), :]
        return x