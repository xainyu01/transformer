import torch
import json
from model.transformer import Transformer
from data_loader import Vocab, build_vocab, get_dataloaders

def generate_masks(src, tgt, pad_id):
    """生成训练用的三种掩码，返回 (src_mask, tgt_mask, cross_mask)"""
    device = src.device
    src_len = src.size(1)
    tgt_len = tgt.size(1)

    # src_mask: 屏蔽源端 <pad>
    src_pad = (src == pad_id).unsqueeze(1).unsqueeze(2)
    src_mask = torch.ones(src.size(0), 1, src_len, src_len, device=device)
    src_mask = src_mask.masked_fill(src_pad.expand_as(src_mask) == 1, 0)

    # tgt_mask: 因果下三角 + 屏蔽目标端 <pad>
    tgt_mask = torch.ones(tgt.size(0), 1, tgt_len, tgt_len, device=device)
    causal = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()
    tgt_mask = tgt_mask.masked_fill(causal.unsqueeze(0).unsqueeze(0), 0)
    tgt_pad = (tgt == pad_id).unsqueeze(1).unsqueeze(2)
    tgt_mask = tgt_mask.masked_fill(tgt_pad.expand_as(tgt_mask) == 1, 0)

    cross_mask = src_mask   # 复用
    return src_mask, tgt_mask, cross_mask



def translate(model, src_sentence, src_vocab, tgt_vocab, max_len=128, device='cpu'):
    model.eval()

    # ---------- 编码源句（只做一次） ----------
    src_tokens = list(src_sentence)
    src_ids = [src_vocab.stoi.get(t, src_vocab.unk_id) for t in src_tokens]
    src_ids = src_ids[:max_len]
    src_ids += [src_vocab.pad_id] * (max_len - len(src_ids))
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)  # (1, src_len)

    # 编码器自注意力掩码（屏蔽 <pad>）
    src_mask = (src_tensor != src_vocab.pad_id).unsqueeze(1).unsqueeze(2)
    src_mask = src_mask.expand(-1, -1, max_len, -1)              # (1, 1, src_len, src_len)

    # 编码器一次前向
    encoder_output = model.encode (src_tensor, src_mask)         # (1, src_len, d_model)

    # ---------- 自回归生成 ----------
    tgt_ids = [tgt_vocab.sos_id]
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=device)  # (1, cur_len)
            cur_len = tgt_tensor.size(1)

            # 目标端因果掩码（下三角）
            tgt_mask = torch.tril(torch.ones(cur_len, cur_len, device=device))
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)       # (1, 1, cur_len, cur_len)

            # 交叉注意力掩码：从 src_mask 动态截取，形状 (1, 1, cur_len, src_len)
            cross_mask = src_mask[:, :, :cur_len, :]

            # 直接调用解码器（绕过 Transformer.forward）
            decoder_output = model.decoder(tgt_tensor, encoder_output, tgt_mask, cross_mask)

            # 取最后一个位置的 logits 并贪婪采样
            next_token_logits = decoder_output[0, -1, :]        # (vocab_size,)
            next_token = next_token_logits.argmax(dim=-1).item()

            tgt_ids.append(next_token)
            if next_token == tgt_vocab.eos_id:
                break

    # 解码：跳过 <sos> 和 <eos>
    translated_tokens = [tgt_vocab.itos[idx] for idx in tgt_ids[1:-1]]
    return ' '.join(translated_tokens)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 重建词汇表 ----------
    json_file = 'data/opus100_zh_en_train.jsonl'          # 你的数据路径
    print("加载数据并构建词汇表...")
    with open(json_file, 'r', encoding='utf-8') as f:
        texts = [json.loads(line) for line in f if line.strip()]

    src_texts = [item['zh'] for item in texts]
    tgt_texts = [item['en'] for item in texts]
    src_vocab = build_vocab(src_texts, min_freq=2, max_size=5000)
    tgt_vocab = build_vocab(tgt_texts, min_freq=2, max_size=5000)
    print(f"源词汇表大小: {len(src_vocab)}, 目标词汇表大小: {len(tgt_vocab)}")

    # ---------- 加载模型 ----------
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

    model.load_state_dict(torch.load(
        r"C:\Users\Admin\Desktop\temp\LLM\transformer\427\checkpoints\transformer_zh_en_epoch5.pth",
        map_location=device
    ))
    model.eval()

    train_loader, _, _, _, _ = get_dataloaders(
        batch_size=16, max_len=128, json_file=json_file
    )

    # 在一小批训练数据上跑前向，查看预测的 top token
    model.eval()
    batch = next(iter(train_loader))  # 需要重新获取一下 train_loader
    src = batch['src'].to(device)
    tgt = batch['tgt'].to(device)

    # 取一个样本
    src_sample = src[0:1, :]
    tgt_sample = tgt[0:1, :]
    src_mask, tgt_mask, _ = generate_masks(src_sample, tgt_sample, src_vocab.pad_id)
    with torch.no_grad():
        output = model(src_sample, tgt_sample, src_mask, tgt_mask)
    pred_tokens = output[0, 0, :].topk(5).indices.tolist()
    pred_words = [tgt_vocab.itos[idx] for idx in pred_tokens]
    print("预测的 top5 词:", pred_words)

    print("模型准备完毕，输入中文（输入 q 退出）：")
    while True:
        sentence = input("> ").strip()
        if sentence.lower() == 'q':
            break
        translation = translate(model, sentence, src_vocab, tgt_vocab, max_len=128, device=device)
        print(translation)


