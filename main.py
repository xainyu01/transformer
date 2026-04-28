import torch
from model import Transformer
from utils import generate_mask, generate_causal_mask

# 初始化模型
src_vocab_size = 10000
tgt_vocab_size = 1000
d_model = 512
num_heads = 8
num_layers = 6
batch_size = 32
src_seq_len = 50
tgt_seq_len = 50

# 创建模型
transformer = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    device='cpu'
)

# 生成随机输入
src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

# 生成掩码
src_mask = generate_mask(src_seq_len, 'cpu')
tgt_mask = generate_causal_mask(tgt_seq_len, 'cpu')

# 前向传播
output = transformer(src, tgt, src_mask, tgt_mask)

# 打印输出形状
print("输入形状 (src):", src.shape)
print("输入形状 (tgt):", tgt.shape)
print("输出形状:", output.shape)