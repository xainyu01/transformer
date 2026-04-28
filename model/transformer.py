# model/transformer.py
from .encoder import Encoder
from .decoder import Decoder
import torch.nn as nn

# 用于导入所有模型组件


class Transformer(nn.Module):
    """
    Transformer 模型，包含Encoder和Decoder
    """
    def __init__(self, src_vocab_size,tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 max_seq_len=5000, dropout=0.1, d_ff=2048, device='cpu'):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, num_layers, 
                              max_seq_len, dropout, d_ff, device)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, num_layers, 
                              max_seq_len, dropout, d_ff, device)
        self.device = device
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        前向传播
        
        Args:
            src: 编码器输入，形状为 [batch_size, src_seq_len]
            tgt: 解码器输入，形状为 [batch_size, tgt_seq_len]
            src_mask: 编码器自注意力掩码，形状为 [batch_size, 1, src_seq_len, src_seq_len]
            tgt_mask: 解码器自注意力掩码，形状为 [batch_size, 1, tgt_seq_len, tgt_seq_len]
        
        Returns:
            输出，形状为 [batch_size, tgt_seq_len, vocab_size]
        """
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        return decoder_output