# 开发日志

## 2026-04-29 – KV Cache 推理加速集成

### 目标
为手写 Transformer 翻译模型添加 KV Cache 自回归推理，理解自注意力与交叉注意力缓存机制，并实现两种推理模式的切换。

### 完成内容

#### 1. KVCache 模块 (`model/KVCache.py`)
- 实现缓存类，支持首次填充任意长度（用于交叉注意力一次性预填）与后续逐 token 追加（用于自注意力）
- 添加 `frozen` 属性：冻结缓存可直接读取，不再进行投影或更新
- 内置设备检查与形状断言，防御常见错误

#### 2. 多层注意力模块改造 (`model/multi_head_attention.py`)
- `forward` 新增 `kv_cache` 参数，按三种路径处理：
  - `None` → 原始投影 + 分头（训练/无缓存推理）
  - `not frozen` → 投影当前 token，更新缓存，取出完整 K/V
  - `frozen` → 跳过投影，直接从缓存取 K/V
- 新增 `W_k`、`W_v` 方法用于外部预计算

#### 3. 解码器组件适配
- `DecoderBlock`：接受并传递 `self_cache`, `cross_cache`
- `Decoder`：接受 `self_caches`, `cross_caches`, `start_pos`，在逐步推理时指定位置编码偏移

#### 4. 位置编码扩展 (`model/positional_encoding.py`)
- `forward` 支持 `start_pos` 参数，从指定位置切片
- 保留 `get_position` 方法作为备选

#### 5. 推理入口重构 (`translate.py`)
- 统一 `translate` 接口，通过 `use_cache` 参数切换两种模式
- 实现 `_decode_with_cache`：仅自注意力使用缓存，交叉注意力每步重算
- 修复多项 Bug（输出层调用、位置编码偏移、设备等）
- 添加交互式模式选择

### 关键收获
- KV Cache 的本质是避免重复计算历史 K/V，将 O(n²) 降为 O(n)
- 位置编码在逐步推理中必须传入绝对位置（`start_pos`），否则会重复使用位置 0
- 交叉注意力缓存的预填充需与逐步追加逻辑区分，通过 `frozen` 实现隔离
