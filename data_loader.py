import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from collections import Counter

class Vocab:
    """自定义词汇表类，固定 PAD=0, UNK=1, SOS=2, EOS=3"""
    def __init__(self, tokens, pad_id=0, unk_id=1, sos_id=2, eos_id=3):
        self.stoi = {}
        self.itos = {}
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        # 固定映射，确保特殊 token 占据前 4 个 ID
        self.stoi['<pad>'] = pad_id; self.itos[pad_id] = '<pad>'
        self.stoi['<unk>'] = unk_id; self.itos[unk_id] = '<unk>'
        self.stoi['<sos>'] = sos_id; self.itos[sos_id] = '<sos>'
        self.stoi['<eos>'] = eos_id; self.itos[eos_id] = '<eos>'

        # 其他 token 从 4 开始编号
        for idx, token in enumerate(tokens, start=4):
            self.stoi[token] = idx
            self.itos[idx] = token

    def __len__(self):
        return len(self.stoi)

def build_vocab(texts, min_freq=1, max_size=None):
    """构建词汇表，确保 <pad> <unk> <sos> <eos> 固定在最前面"""
    all_tokens = []
    for text in texts:
        # 中文按字符分割
        all_tokens.extend(list(text))

    token_counts = Counter(all_tokens)

    # 过滤低频词
    tokens = [t for t, count in token_counts.items() if count >= min_freq]

    # 限制词汇表大小
    if max_size is not None:
        tokens = tokens[:max_size]

    # 移除可能重复的特殊 token
    special_tokens = {'<pad>', '<unk>', '<sos>', '<eos>'}
    tokens = [t for t in tokens if t not in special_tokens]

    # 最终传递给 Vocab 的 tokens 列表不包含特殊 token，
    # Vocab 内部会自动在最前面插入特殊 token
    return Vocab(tokens, pad_id=0, unk_id=1, sos_id=2, eos_id=3)

class TranslationDataset(Dataset):
    """翻译数据集，目标序列自动添加 <sos> 和 <eos>"""
    def __init__(self, json_file, src_vocab, tgt_vocab, max_len=128):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.pad_id = src_vocab.pad_id

        self.data = []
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        self.src_texts = [item['zh'] for item in self.data]
        self.tgt_texts = [item['en'] for item in self.data]

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        # 源序列（中文按字符）
        src_tokens = list(self.src_texts[idx])
        src_ids = [self.src_vocab.stoi.get(t, self.src_vocab.unk_id) for t in src_tokens]
        src_ids = [min(i, len(self.src_vocab)-1) for i in src_ids]

        # 目标序列（英文按空格分词，并添加 <sos> 和 <eos>）
        tgt_tokens = self.tgt_texts[idx].split()
        tgt_tokens = ['<sos>'] + tgt_tokens + ['<eos>']
        tgt_ids = [self.tgt_vocab.stoi.get(t, self.tgt_vocab.unk_id) for t in tgt_tokens]
        tgt_ids = [min(i, len(self.tgt_vocab)-1) for i in tgt_ids]

        # 截断和填充
        src_ids = src_ids[:self.max_len]
        src_ids += [self.pad_id] * (self.max_len - len(src_ids))

        tgt_ids = tgt_ids[:self.max_len]
        tgt_ids += [self.pad_id] * (self.max_len - len(tgt_ids))

        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(tgt_ids, dtype=torch.long)
        }

def get_dataloaders(batch_size=32, max_len=128, json_file='data/opus100_zh_en_train.jsonl'):
    """创建训练和验证数据加载器"""
    os.makedirs('data', exist_ok=True)

    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    print(f"🚀 加载数据集: {json_file}")

    with open(json_file, 'r', encoding='utf-8') as f:
        texts = [json.loads(line) for line in f if line.strip()]

    src_texts = [item['zh'] for item in texts]
    tgt_texts = [item['en'] for item in texts]

    print("构建词汇表...")
    src_vocab = build_vocab(src_texts, min_freq=2, max_size=5000)
    tgt_vocab = build_vocab(tgt_texts, min_freq=2, max_size=5000)

    print(f"词汇表大小: src={len(src_vocab)}, tgt={len(tgt_vocab)}")
    print(f"特殊 token: pad={src_vocab.pad_id}, unk={src_vocab.unk_id}, "
          f"sos={src_vocab.sos_id}, eos={src_vocab.eos_id}")

    full_dataset = TranslationDataset(json_file, src_vocab, tgt_vocab, max_len=max_len)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, len(src_vocab), len(tgt_vocab), src_vocab.pad_id