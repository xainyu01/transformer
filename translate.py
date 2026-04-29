import torch
import json
from model.transformer import Transformer
from data_loader import Vocab, build_vocab, get_dataloaders
from model.KVCache import KVCache
import math

# ------------------- 工具函数 -------------------
def generate_masks(src, tgt, pad_id):
    """训练用的三种掩码"""
    device = src.device
    src_len = src.size(1)
    tgt_len = tgt.size(1)

    # src_mask: 屏蔽源端 <pad>，形状 (batch, 1, src_len, src_len)
    src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2).float()
    src_mask = src_mask.expand(-1, -1, src_len, -1)

    # tgt_mask: 因果下三角 + 屏蔽目标端 <pad>，形状 (batch, 1, tgt_len, tgt_len)
    causal = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()
    tgt_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2).float()
    tgt_mask = tgt_mask.expand(-1, -1, tgt_len, -1)
    tgt_mask = tgt_mask.masked_fill(causal.unsqueeze(0).unsqueeze(0), 0)

    # cross_mask: 交叉注意力只需屏蔽源端 <pad>，形状 (batch, 1, 1, src_len) 广播到所有 query
    cross_mask = (src != pad_id).unsqueeze(1).unsqueeze(2).float()
    return src_mask, tgt_mask, cross_mask


def translate(
    model, src_sentence, src_vocab, tgt_vocab, max_len=128, device='cpu',
    use_cache=True   # 新增：True 使用 KV Cache，False 使用原始逐序列推理
):
    """
    统一的翻译接口，支持两种推理模式。

    注意：若 use_cache=True，模型需支持 self_caches / cross_caches 参数（见文末修改说明）。
    """
    model.eval()

    # ------- 1. 源句编码（两种模式通用）-------
    src_tokens = list(src_sentence)
    src_ids = [src_vocab.stoi.get(t, src_vocab.unk_id) for t in src_tokens]
    src_ids = src_ids[:max_len]
    src_ids += [src_vocab.pad_id] * (max_len - len(src_ids))
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)  # (1, src_len)

    # 源端 pad 掩码 (1, 1, src_len, src_len) 用于编码器自注意力
    src_mask = (src_tensor != src_vocab.pad_id).unsqueeze(1).unsqueeze(2).float()
    src_mask = src_mask.expand(-1, -1, max_len, -1)

    with torch.no_grad():
        encoder_output = model.encoder(src_tensor, src_mask)   # (1, src_len, d_model)

    # ------- 2. 根据模式选择解码方式 -------
    if use_cache:
        return _decode_with_cache(
            model, encoder_output, src_tensor, src_vocab, tgt_vocab,
            max_len, device
        )
    else:
        return _decode_without_cache(
            model, encoder_output, src_tensor, src_vocab, tgt_vocab,
            max_len, device
        )


def _decode_without_cache(model, encoder_output, src_tensor, src_vocab, tgt_vocab, max_len, device):
    """原始的逐序列推理（无缓存）"""
    tgt_ids = [tgt_vocab.sos_id]
    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=device)  # (1, cur_len)
        cur_len = tgt_tensor.size(1)

        # 因果掩码 (1,1,cur_len,cur_len)
        tgt_mask = torch.tril(torch.ones(cur_len, cur_len, device=device)).unsqueeze(0).unsqueeze(0)

        # 交叉注意力掩码：截取 (1,1,cur_len,src_len)
        cross_mask = (src_tensor != src_vocab.pad_id).unsqueeze(1).unsqueeze(2).float()
        cross_mask = cross_mask.expand(-1, -1, cur_len, -1)   # 形状 (1,1,cur_len,src_len)

        with torch.no_grad():
            # 你的原始 decoder 调用，可能需要调整参数名（与训练时一致）
            decoder_output = model.decoder(tgt_tensor, encoder_output, tgt_mask, cross_mask)

        next_token = decoder_output[0, -1, :].argmax().item()
        tgt_ids.append(next_token)
        if next_token == tgt_vocab.eos_id:
            break

    return ' '.join(tgt_vocab.itos[idx] for idx in tgt_ids[1:-1])


def _decode_with_cache(model, encoder_output, src_tensor, src_vocab, tgt_vocab, max_len, device):
    """KV Cache 加速推理"""
    # 预计算交叉注意力缓存（只做一次）
    decoder_layers = model.decoder.layers
    num_layers = len(decoder_layers)
    cross_caches = []
    for layer in decoder_layers:
        K = layer.cross_attn.W_k(encoder_output)  # (1, src_len, d_model)
        V = layer.cross_attn.W_v(encoder_output)
        K = K.view(1, -1, layer.cross_attn.n_heads, layer.cross_attn.d_k).transpose(1, 2)
        V = V.view(1, -1, layer.cross_attn.n_heads, layer.cross_attn.d_k).transpose(1, 2)
        cache = KVCache()
        cache.update(K, V)
        cache.set_frozen(True)
        cross_caches.append(cache)

    # 固定交叉注意力掩码 (1,1,1,src_len)
    cross_mask = (src_tensor != src_vocab.pad_id).unsqueeze(1).unsqueeze(2).float()

    # 初始化自注意力缓存（每层一个空缓存）
    self_caches = [KVCache() for _ in range(num_layers)]

    tgt_ids = [tgt_vocab.sos_id]
    for step in range(max_len):
        current_input = torch.tensor([[tgt_ids[-1]]], dtype=torch.long, device=device)  # (1,1)

        # 带缓存的解码器前向
        with torch.no_grad():
            x = model.decoder(
                current_input,
                encoder_output,
                self_mask=None,
                cross_mask=cross_mask,
                self_caches=self_caches,
                cross_caches=cross_caches,
                start_pos=step
            )

        next_token = x[0, -1, :].argmax().item()
        tgt_ids.append(next_token)
        if next_token == tgt_vocab.eos_id:
            break

    return ' '.join(tgt_vocab.itos[idx] for idx in tgt_ids[1:-1])


# ------------------- 测试主程序 -------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据与词汇表
    json_file = 'data/opus100_zh_en_train.jsonl'
    print("加载数据并构建词汇表...")
    with open(json_file, 'r', encoding='utf-8') as f:
        texts = [json.loads(line) for line in f if line.strip()]
    src_texts = [item['zh'] for item in texts]
    tgt_texts = [item['en'] for item in texts]
    src_vocab = build_vocab(src_texts, min_freq=2, max_size=5000)
    tgt_vocab = build_vocab(tgt_texts, min_freq=2, max_size=5000)
    print(f"源词汇表: {len(src_vocab)}  目标词汇表: {len(tgt_vocab)}")

    # 加载模型
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=128,
        device=device
    ).to(device)

    state_dict = torch.load("checkpoints/transformer_zh_en_epoch5.pth", map_location=device)

    # 移除可能缺失的固定编码键（如果有的话）
    for key in ["encoder.positional_encoding.pe", "decoder.positional_encoding.pe"]:
        if key in state_dict:
            del state_dict[key]

    model.load_state_dict(state_dict,strict=False)


    def _reset_positional_encoding(module):
        if hasattr(module, 'pe'):
            # 重新计算 pe 并注册为 buffer
            max_seq_len = module.max_seq_len
            d_model = module.d_model
            position = torch.arange(max_seq_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_seq_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            module.register_buffer('pe', pe.unsqueeze(0))


    _reset_positional_encoding(model.encoder.positional_encoding)
    _reset_positional_encoding(model.decoder.positional_encoding)

    model.eval()

    # 选择推理模式
    print("\n请选择推理模式：")
    print("  1. 原始模式（无缓存）")
    print("  2. KV Cache 模式（加速，需模型已适配）")
    choice = input("输入序号 (1/2): ").strip()
    use_cache = (choice == '2')

    # 交互循环
    print("\n输入中文（输入 q 退出）：")
    while True:
        sentence = input("> ").strip()
        if sentence.lower() == 'q':
            break
        translation = translate(
            model, sentence, src_vocab, tgt_vocab,
            max_len=128, device=device, use_cache=use_cache
        )
        print(translation)