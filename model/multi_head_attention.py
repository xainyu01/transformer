import torch
import torch.nn as nn 
import numpy as np
from sympy.codegen.ast import Return


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块框架

    该模块实现Transformer中的多头自注意力或交叉注意力。
    包含Q/K/V线性投影、多头分割、注意力计算、输出投影等核心步骤。

    Args:
        d_model: 输入特征维度
        num_heads: 注意力头数
        dropout: Dropout概率，默认0.1
        bias: 是否在Linear层使用偏置，默认True
    """
    def __init__( self, n_heads=8, d_model=512,dropout=0.1,bias=True):
       super().__init__()
       assert d_model % n_heads == 0, "d_model must be divisible by num_heads"
       self.n_heads = n_heads
       self.d_model = d_model
       self.d_k = d_model // n_heads
       self.w_q = nn.Linear(d_model, d_model,bias=bias)
       self.w_k = nn.Linear(d_model, d_model,bias=bias)
       self.w_v = nn.Linear(d_model, d_model,bias=bias)
       self.w_o = nn.Linear(d_model, d_model,bias=bias)#输出投影
       self.dropout = nn.Dropout(0.1)
       self.softmax = nn.Softmax(dim=-1)
    
    def split_heads(self, x):
        """多头分割
        
        将输入特征进行多头分割，并转置为[batch_size, n_heads, seq_len, d_k]
        
        Args:
            x: 输入特征，形状为[batch_size, seq_len, d_model]
        
        Returns:
            分割后的特征，形状为[batch_size, n_heads, seq_len, d_k]
            输入特征，形状为[batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        return x.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
    
    def forward(self, query, key, value, mask=None,kv_cache=None):
        """
        多头注意力计算
        
        Args:
            query: 查询特征，形状为[batch_size, seq_len, d_model]
            key: 关键字特征，形状为[batch_size, seq_len, d_model]
            value: 值特征，形状为[batch_size, seq_len, d_model]
            mask: 掩码，形状为[batch_size, 1, seq_len, seq_len]（用于自注意力）
                或 [batch_size, seq_len_q, seq_len_k]（用于交叉注意力）
        
        Returns:
            多头注意力结果，形状为[batch_size, seq_len, d_model]
            注意力权重，形状为[batch_size, n_heads, seq_len, seq_len]
        """
        # 线性投影
        query = self.split_heads(self.w_q(query))
        if kv_cache is not None and not kv_cache.is_frozen:
            key = self.split_heads(self.w_k(key))
            value = self.split_heads(self.w_v(value))
            kv_cache.update(key, value)
            key, value = kv_cache.get()
        elif kv_cache is not None and kv_cache.is_frozen:
            key, value = kv_cache.get()
        else:
            key = self.split_heads(self.w_k(key))
            value = self.split_heads(self.w_v(value))


        # 计算注意力分数
        # scores: [batch_size, n_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # 应用掩码（如果提供）
        if mask is not None:
            # 确保mask形状与scores匹配
            if mask.dim() == 3:
                # 将[batch_size, seq_len_q, seq_len_k] -> [batch_size, 1, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(1)
            # 掩码值为0的位置设为负无穷
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 计算注意力权重（softmax）
        attention_weights = self.softmax(scores)
        
        # 应用Dropout（关键！防止过拟合）
        attention_weights = self.dropout(attention_weights)
        
        # 计算注意力输出
        # attention_output: [batch_size, n_heads, seq_len_q, d_k]
        attention_output = torch.matmul(attention_weights, value)
        
        # 合并多头
        # 转置回[batch_size, seq_len_q, n_heads, d_k]
        attention_output = attention_output.transpose(1, 2)
        # 重塑为[batch_size, seq_len_q, d_model]
        attention_output = attention_output.contiguous().view(
            attention_output.size(0), 
            attention_output.size(1), 
            self.d_model
        )
        
        # 输出投影
        attention_output = self.w_o(attention_output)
        
        return attention_output, attention_weights

    def W_k(self,x)->torch.Tensor:
        return self.w_k(x)

    def W_v(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_v(x)

    def loadCrossCache(self,crossCacheS=None):
        return


    
        

