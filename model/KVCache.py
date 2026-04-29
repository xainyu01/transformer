import torch
from typing import Optional, Tuple

class KVCache:
    """自回归 KV 缓存，自动处理空缓存、设备和形状验证"""

    def __init__(self):
        self._k: Optional[torch.Tensor] = None
        self._v: Optional[torch.Tensor] = None
        self.frozen :Optional[bool]=None

    def __len__(self) -> int:
        """返回缓存的序列长度"""
        return 0 if self._k is None else self._k.size(2)  # (batch, heads, seq, dim)

    @property
    def is_empty(self) -> bool:
        return self._k is None

    @property
    def is_frozen(self):
        return self.frozen

    def set_frozen(self,frozen=None):
        self.frozen=frozen

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """
        new_k, new_v: 当前步的 Key / Value
                     形状 (batch, num_heads, 1, head_dim)
        """
        # 首次缓存，直接赋值并记录设备
        if self.is_empty:
            self._k = new_k
            self._v = new_v
            self._device = new_k.device
            return

        # 设备检查
        if new_k.device != self._device:
            raise RuntimeError(
                f"新 K 在 {new_k.device}，但缓存期望 {self._device}"
            )

        # 形状检查：确保除序列长度外的维度吻合
        assert new_k.dim() == 4, f"期望 4 维张量，但得到 {new_k.dim()} 维"
        assert self._k.shape[0:2] == new_k.shape[0:2], \
            f"batch/heads 维度不匹配: {self._k.shape} vs {new_k.shape}"
        assert self._k.shape[3] == new_k.shape[3], \
            f"head_dim 不匹配: {self._k.shape[3]} vs {new_k.shape[3]}"
        assert new_k.size(2) == 1, f"当前步的序列长度必须为 1，实际为 {new_k.size(2)}"

        # 在序列维度拼接（dim=2）
        self._k = torch.cat([self._k, new_k], dim=2)
        self._v = torch.cat([self._v, new_v], dim=2)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回当前所有缓存的 (K, V)"""
        if self.is_empty:
            raise RuntimeError("缓存为空，请先调用 update()")
        return self._k, self._v

    def clone(self) -> 'KVCache':
        """深复制缓存，供 batch beam search 使用（高级场景）"""
        new_cache = KVCache()
        if not self.is_empty:
            new_cache._k = self._k.clone()
            new_cache._v = self._v.clone()
            new_cache._device = self._device
        return new_cache