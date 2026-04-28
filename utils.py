import torch

def generate_mask(seq_len, device='cpu'):
    """
    生成自注意力掩码（上三角矩阵）

    Args:
        seq_len: 序列长度
        device: 设备（'cpu'或'cuda'）
    
    Returns:
        自注意力掩码，形状为 [1, 1, seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0).to(device)

def generate_causal_mask(seq_len, device='cpu'):
    """
    生成因果掩码（用于Decoder自注意力）

    Args:
        seq_len: 序列长度
        device: 设备（'cpu'或'cuda'）
    
    Returns:
        因果掩码，形状为 [1, 1, seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0).to(device)