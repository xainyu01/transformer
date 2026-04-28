import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    前馈神经网络层，包含两个线性层和ReLU激活函数

    该模块是Transformer中的前馈网络，用于在注意力机制后进一步处理信息。

    Args:
        d_model: 输入特征维度
        d_ff: 隐藏层维度，默认2048
        dropout: Dropout概率，默认0.1
    """
    def __init__(self,d_model,d_ff=2048,dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
       
    def forward(self,x):
        """
        前向传播

        Args:
            x: 输入张量，形状为 [batch_size, seq_len, d_model]
        
        Returns:
            输出张量，形状为 [batch_size, seq_len, d_model]
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x





