#  Transformer 中英翻译

从零手写的 **Transformer 机器翻译模型**（中文 → 英文），基于 PyTorch 2.x 实现。

涵盖了 **多头注意力、混合精度训练、自回归推理、掩码生成** 等完整流程，适合作为 AI 工程岗求职项目。



![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)



![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)



![License](https://img.shields.io/badge/License-MIT-green.svg)



***

## 项目结构



```
transformer/

├── model/

│   ├── transformer.py          # 模型主文件（Encoder + Decoder）

│   ├── encoder.py              # 编码器

│   ├── encoder\_block.py        # 编码器层

│   ├── decoder.py              # 解码器

│   ├── decoder\_block.py        # 解码器层

│   ├── feed\_forward.py         # 前馈网络（FFN）

│   ├── multi\_head\_attention.py # 多头注意力（手写核心计算）

│   └── positional\_encoding.py  # 正弦位置编码

├── data\_loader.py              # 数据加载、词汇表构建

├── train.py                    # 训练脚本（混合精度）

├── translate.py                # 推理脚本（自回归生成）

├── download\_opus100.py         # 数据集下载工具

├── requirements.txt            # 依赖包

├── .gitignore

└── data/

&#x20;   ├── opus100\_zh\_en\_train.jsonl

&#x20;   ├── opus100\_zh\_en\_validation.jsonl

&#x20;   └── opus100\_zh\_en\_test.jsonl
```



***

##  快速开始

### 1. 环境配置



```
\# 克隆仓库

git clone https://github.com/yourname/transformer.git

cd transformer

\# 安装依赖（只需 PyTorch + numpy + tqdm）

pip install -r requirements.txt
```

### 2. 下载数据集



```
python download\_opus100.py
```

（若已包含 `data/` 下的 `.jsonl` 文件则跳过此步）

### 3. 训练模型



```
python train.py
```

训练权重将保存在 `checkpoints/` 目录下。

### 4. 交互式翻译



```
python translate.py
```

输入中文句子，模型将逐词生成英文翻译。



***

## 模型架构



| 组件        | 说明                                                            |
| --------- | ------------------------------------------------------------- |
| **位置编码**  | 正弦位置编码（Sin/Cos），支持动态长度外推                                      |
| **多头注意力** | 手写 Q/K/V 拆分、多头重排、Scaled Dot-Product Attention、合并与输出投影         |
| **编码器**   | 6 层 EncoderBlock（Self-Attention + FFN + 残差 + LayerNorm）       |
| **解码器**   | 6 层 DecoderBlock（Self-Attention + Cross-Attention + FFN + 残差） |
| **前馈网络**  | 两层 Linear + GELU 激活，中间维度 2048                                 |
| **推理策略**  | 自回归贪婪解码，动态生成因果掩码与交叉注意力掩码                                      |

### 训练配置



| 参数              | 值                                        |
| --------------- | ---------------------------------------- |
| 模型维度 `d_model`  | 512                                      |
| 注意力头数 `n_heads` | 8                                        |
| 编码器 / 解码器层数     | 6                                        |
| 前馈隐藏层 `d_ff`    | 2048                                     |
| Dropout         | 0.1                                      |
| 最大序列长度          | 128                                      |
| Batch Size      | 16                                       |
| 学习率             | 1e-4                                     |
| 优化器             | Adam (β₁=0.9, β₂=0.98)                   |
| 损失函数            | CrossEntropyLoss (ignore\_index=pad\_id) |
| 混合精度            | `torch.amp.autocast` + `GradScaler`      |
| 硬件              | NVIDIA RTX 5060                          |



***

## 🔑 关键技术细节

### 1. 掩码生成（三种掩码）



* **源端掩码**（Encoder Self-Attention）：屏蔽 `形状 `\[B, 1, src\_len, src\_len]\`

* **目标端掩码**（Decoder Self-Attention）：下三角因果掩码 + `<pad>` 屏蔽，形状 `[B, 1, tgt_len, tgt_len]`

* **交叉注意力掩码**（Cross-Attention）：复用源端掩码，通过 `src_mask[:,:,:cur_len,:]` 动态截取，形状 `[B, 1, tgt_len, src_len]`

### 2. 自回归推理

直接调用 `model.encoder` 和 `model.decoder`，在循环中动态构造掩码，避免 `Transformer.forward` 无法适应变长序列的问题。

### 3. 混合精度训练

利用 `torch.amp.autocast('cuda')` 自动将线性层转为 `float16` 加速，`GradScaler` 防止梯度下溢。

### 4. 输出与损失

模型最后一层返回 **logits**（未归一化分数），配合 `CrossEntropyLoss(ignore_index=pad_id)` 自动进行 Softmax 并忽略填充位置。

### 5. 位置编码

使用 `register_buffer` 注册位置编码张量，保证其随模型自动迁移设备 / 类型，且不被训练。



***

##  已知问题

当前训练数据量较少（目标词汇表约 400 词），推理输出可能重复出现 \`。

**解决方案**：扩充平行语料，增大词汇表，或采用 BPE 子词分词。



***

## 期间问题


### 1. 手写 MultiHeadAttention 的前向传播

核心步骤：
1. **Q/K/V 投影**：用 `nn.Linear` 将输入投影到 Q、K、V，各 `[batch, seq_len, d_model]`
2. **多头拆分**：`view` 成 `[batch, seq_len, n_heads, d_k]`，再 `transpose` 为 `[batch, n_heads, seq_len, d_k]`
3. **Scaled Dot-Product Attention**：`scores = Q @ K^T / sqrt(d_k)`，对 scores 用 `softmax`，再乘以 V
4. **多头合并**：`transpose` 回 `[batch, seq_len, n_heads, d_k]`，`reshape` 为 `[batch, seq_len, d_model]`
5. **输出投影**：经 `w_o` 线性层得到最终输出

### 2. 三种掩码的形状、生成方式与作用

| 掩码 | 形状 | 作用 |
|------|------|------|
| **源端掩码** | `[B, 1, src_len, src_len]` | 屏蔽 Encoder 自注意力中的 `<pad>` 列 |
| **目标端掩码** | `[B, 1, tgt_len, tgt_len]` | 因果下三角 + `<pad>` 屏蔽，Decoder 自注意力用 |
| **交叉掩码** | `[B, 1, tgt_len, src_len]` | 屏蔽 Cross-Attention 中源端的 `<pad>` 列，从 `src_mask` 截取前 `tgt_len` 行复用 |

生成方式：源端掩码通过 `(src == pad_id).unsqueeze(1).unsqueeze(2)` 标记 pad 位置后 `masked_fill`；目标端掩码在此基础上再加上 `triu` 上三角因果约束。

### 3. `nn.ModuleList` vs `nn.Sequential` 的选择

- `nn.Sequential`：子模块按顺序自动调用，输入输出必须是单一张量，**不能传递额外参数**
- `nn.ModuleList`：只是把子模块放在一个能被 PyTorch 识别的列表里，需要**手动写 for 循环**调用，可灵活传参
- 本项目选择 `ModuleList` 是因为每一层 Encoder/Decoder 都需要传入 `mask` 参数，`Sequential` 无法满足

### 4. 残差连接 + LayerNorm 的位置

- 本项目采用 **Post-LN**：`x = LayerNorm(x + Sublayer(x))`
- 即先做残差加法，再通过 LayerNorm
- 另一种是 Pre-LN：`x = x + Sublayer(LayerNorm(x))`，现在大模型更倾向 Pre-LN（训练更稳定）

### 5. 输出投影 `w_o` 的作用

有两个核心作用：
1. **多头信息融合**：拼接只是物理并列，`w_o` 的矩阵乘法让每个输出维度都是所有头特征的加权和，实现跨头信息交互
2. **残差连接对齐**：多头拼完的特征空间与输入 `x` 的空间不同，`w_o` 将其重新映射回与输入兼容的语义空间，保证 `x + Sublayer(x)` 的有效性

去掉 `w_o`，残差连接变成两个不同空间向量的强行相加，训练稳定性会下降。

### 6. `register_buffer` 与 `nn.Parameter` 的区别

| | `nn.Parameter` | `register_buffer` |
|--|:--:|:--:|
| 是否可训练 | 是，参与梯度更新 | 否，固定不变 |
| 是否出现在 `parameters()` | 是 | 否 |
| 是否跟随 `model.to(device/dtype)` | 是 | 是 |
| 是否保存到 `state_dict()` | 是 | 默认是，`persistent=False` 可不保存 |

位置编码用 `register_buffer`，因为它需要随模型迁移设备/类型，但不需要被训练。


### 7. `CrossEntropyLoss` 为何接收 logits 而非 Softmax 后的概率

- `nn.CrossEntropyLoss` 内部集成了 `LogSoftmax` + `NLLLoss`
- 它**要求输入是原始 logits**（未归一化分数），而非概率
- 如果在模型最后一层先做 Softmax，再送入损失函数，等效于两次 softmax，会导致数值下溢和梯度消失
- 返回 logits 保持了数值稳定性，且在推理阶段可根据需要灵活选择 argmax、束搜索等解码策略

### 8. 混合精度训练中 `autocast` + `GradScaler` 的原理

- `autocast`：自动将 `nn.Linear`、`matmul` 等计算转为 `float16` 利用 Tensor Core 加速，同时保持 `softmax`、`loss` 等敏感操作在 `float32`
- `GradScaler`：解决 `float16` 梯度下溢问题——先将 loss 乘以一个大因子放大梯度 → 反向传播 → 更新前除以相同因子还原梯度
- 两者配合，在 RTX 5060 上可实现约 2 倍训练加速 + 显存减半
- 若用 `bfloat16`，则不需要 `GradScaler`，因为 `bfloat16` 的指数范围与 `float32` 一致

### 9. 自回归推理的动态掩码设计与 Encoder 前向复用

- **Encoder 只跑一次**：源端输入固定，`encoder_output` 在整个生成过程中复用
- **Decoder 逐步生成**：每步输入当前已生成的 token 序列 `[<sos>, ..., token_t]`，长度 `cur_len` 从 1 增长到 `max_len`
- **动态掩码**：因果掩码 `tgt_mask` 根据 `cur_len` 实时生成下三角矩阵；交叉掩码从 `src_mask` 截取 `[:cur_len, :]` 匹配当前长度
- 绕过 `Transformer.forward`，直接调用 `model.encoder` 和 `model.decoder`，比每次重新编码源端更高效

### 10. `contiguous()` 的作用与 `view()` / `reshape()` 的区别

| 操作 | 对内存连续性的要求 | 是否可能复制数据 |
|------|:---:|:---:|
| `view()` | **必须连续**，否则直接报错 | 不复制，共享内存 |
| `reshape()` | 不要求，自动处理 | 尽量不复制，必要时复制 |
| `contiguous()` | — | 不连续时复制一份连续版本，连续时返回自身 |

- `transpose` / `permute` 后张量不连续，若之后要用 `view` 改变形状，必须先 `contiguous()`
- 本项目多头合并时先 `transpose` 再 `view`，因此必须加 `.contiguous()`
- 用 `.reshape()` 可以省去手动 `contiguous()`，因为它在不连续时会自动复制


***

## License

MIT License. 欢迎 Star ⭐ 和 PR。

📅 Date**: 2026-04-28

