# model/encoder.py
import torch
import torch.nn as nn
from model.decoder_block import DecoderBlock
from model.positional_encoding import PositionalEncoding
from model.encoder_block import EncoderBlock

# 确保导入了之前的 PositionalEncoding, EncoderBlock 等

class Encoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6, 
                 max_seq_len=5000, dropout=0.1, d_ff=2048, device='cpu'):
        super().__init__()
        # ❌ 删除了 self.embedding = nn.Embedding(...)
        
        # 位置编码保留，因为它是对向量进行操作
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
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, d_model] (已经是 Embedding 后的向量)
            mask: 自注意力掩码
        """
        # ✅ 直接添加位置编码，不再做 Embedding
        x = self.positional_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        return x
    


    # model/decoder.py
import torch
import torch.nn as nn
# 确保导入了之前的 PositionalEncoding, DecoderBlock 等

class Decoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6, 
                 max_seq_len=5000, dropout=0.1, d_ff=2048, device='cpu'):
        super().__init__()
        # ❌ 删除了 self.embedding
        # ❌ 删除了 self.linear (输出层移到 Transformer 类中)
        
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, device)
        
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, dropout, d_ff) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.device = device
    
    def forward(self, x, encoder_output, self_mask, cross_mask):
        """
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, d_model] (已经是 Embedding 后的向量)
        """
        # ✅ 直接添加位置编码
        x = self.positional_encoding(x)
        
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, cross_mask)
        
        x = self.norm(x)
        # ❌ 删除了 return self.linear(x)，只返回特征向量
        return x
    


    # model/transformer.py
from .encoder import Encoder
from .decoder import Decoder
import torch.nn as nn
import torch
import math

class Transformer(nn.Module):
    """
    完整的 Transformer 模型控制器
    负责：Embedding -> Encoder/Decoder -> Output Projection
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 max_seq_len=5000, dropout=0.1, d_ff=2048, device='cpu'):
        super().__init__()
        
        # 1. 独立的 Embedding 层
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        
        # 2. Encoder 和 Decoder (内部已包含 PositionalEncoding)
        self.encoder = Encoder(d_model=d_model, num_heads=num_heads, num_layers=num_layers, 
                               max_seq_len=max_seq_len, dropout=dropout, d_ff=d_ff, device=device)
        
        self.decoder = Decoder(d_model=d_model, num_heads=num_heads, num_layers=num_layers, 
                               max_seq_len=max_seq_len, dropout=dropout, d_ff=d_ff, device=device)
        
        # 3. 最终输出层
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
        
        self.d_model = d_model
        self.device = device
        
        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask, cross_mask):
        """
        Args:
            src: [batch_size, src_seq_len] (Token IDs)
            tgt: [batch_size, tgt_seq_len] (Token IDs, Decoder Input)
            src_mask: [batch_size, 1, src_len, src_len] (Encoder Self-Attn Mask)
            tgt_mask: [batch_size, 1, tgt_len, tgt_len] (Decoder Self-Attn Mask)
            cross_mask: [batch_size, 1, tgt_len, src_len] (Cross-Attn Mask)
        
        Returns:
            logits: [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # 1. Embedding + Scaling
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        
        # 2. Encoder
        # Encoder 内部会处理 Positional Encoding
        encoder_output = self.encoder(src_emb, src_mask)
        
        # 3. Decoder
        # Decoder 内部会处理 Positional Encoding
        # 注意：您的 DecoderBlock 需要能接收 cross_mask
        # 检查 decoder_block.py 的 forward 签名是否为 (x, enc_out, self_mask, cross_mask)
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask, cross_mask)
        
        # 4. Output Projection
        logits = self.final_layer(decoder_output)
        
        return logits