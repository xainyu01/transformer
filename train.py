import os
# 设置离线模式环境变量
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp  # 修复混合精度警告
from model.transformer import Transformer
from data_loader import get_dataloaders
import math
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=======使用设备: {device}=======")

if device.type == 'cuda':
    print(f"显卡型号: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 超参数
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 6
D_FF = 2048
DROPOUT = 0.1
MAX_LEN = 128  # 确保是 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4

# 1. 加载数据
print("=======加载数据======")
train_loader, val_loader, src_vocab_size, tgt_vocab_size, pad_id = get_dataloaders(
    batch_size=BATCH_SIZE, 
    max_len=MAX_LEN
)

# 2. 初始化模型
print("======构建模型======")
model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=D_MODEL,
    num_heads=N_HEADS,
    num_layers=N_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT,
    max_seq_len=MAX_LEN,
    device=device
).to(device)

# 3. 优化器与 Loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

# 混合精度训练 scaler (修复弃用警告)
scaler = torch.amp.GradScaler('cuda')

def generate_masks(src, tgt, pad_id):
    """
    生成训练所需的三种掩码：
    - src_attention_mask: [batch, 1, src_len, src_len] 屏蔽源序列的 padding 位置
    - tgt_attention_mask: [batch, 1, tgt_len, tgt_len] 因果下三角 + 屏蔽目标序列的 padding 位置
    - cross_attention_mask: [batch, 1, tgt_len, src_len] 复用 src_attention_mask 屏蔽源端 padding

    Args:
        src:  [batch, src_len]
        tgt:  [batch, tgt_len]
        pad_id: padding 标记的整数 id
    Returns:
        src_attention_mask, tgt_attention_mask, cross_attention_mask
    """
    device = src.device
    src_len = src.size(1)
    tgt_len = tgt.size(1)

    # ---- 源序列掩码：屏蔽源端的 <pad> ----
    # 标记哪些位置是 pad： [batch, src_len] -> [batch, 1, 1, src_len]
    src_pad_positions = (src == pad_id).unsqueeze(1).unsqueeze(2)
    # 初始化全 1 矩阵 [batch, 1, src_len, src_len]
    src_attention_mask = torch.ones(src.size(0), 1, src_len, src_len, device=device)
    # 将被关注维度中对应 pad 的列全部置 0
    src_attention_mask = src_attention_mask.masked_fill(
        src_pad_positions.expand_as(src_attention_mask) == 1, 0
    )

    # ---- 目标序列掩码：因果下三角 + 屏蔽目标端 <pad> ----
    tgt_attention_mask = torch.ones(tgt.size(0), 1, tgt_len, tgt_len, device=device)

    # 因果部分：生成上三角为 True 的 mask
    causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()
    # 广播到所有 batch 和 head：把上三角位置置 0
    tgt_attention_mask = tgt_attention_mask.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0)

    # Padding 部分：标记目标端 pad 位置 [batch, tgt_len] -> [batch, 1, 1, tgt_len]
    tgt_pad_positions = (tgt == pad_id).unsqueeze(1).unsqueeze(2)
    tgt_attention_mask = tgt_attention_mask.masked_fill(
        tgt_pad_positions.expand_as(tgt_attention_mask) == 1, 0
    )

    # ---- 交叉注意力掩码：与源序列掩码逻辑完全一致 ----
    cross_attention_mask = src_attention_mask  # 形状 [batch, 1, src_len, src_len] 在交叉注意力中会自动匹配

    return src_attention_mask, tgt_attention_mask, cross_attention_mask

# 训练函数
def train_model():
    print("=========开始训练========")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        num_batches = 0
        
        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            src = batch['src'].to(device)  # [batch, 128]
            tgt = batch['tgt'].to(device)  # [batch, 128]
            
            # 正确做法：decoder输入 = 完整序列 (包括sos和eos)
            tgt_input = tgt  # [batch, 128]
            labels = tgt[:, 1:]  # [batch, 127] (移除sos)
            
            # 确保 labels 是 Long 类型 (双重保险)
            labels = labels.long()
            
            # 生成 Masks (使用实际长度 128)
            src_mask, tgt_mask, cross_mask = generate_masks(src, tgt_input, pad_id)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # Forward: 输出 [batch, 128, vocab]
                outputs = model(src, tgt_input, src_mask, cross_mask)
                
                # 关键修复：只取前 labels.size(1) 个时间步 (127)
                outputs = outputs[:, :labels.size(1), :]
                
                # 使用 reshape
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
            
            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} 完成，平均 Loss: {avg_loss:.4f}")
        
        # 保存模型
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/transformer_zh_en_epoch{epoch+1}.pth")

# 关键修复：Windows 多进程安全
if __name__ == '__main__':
    train_model()
    print("=======训练完成========")