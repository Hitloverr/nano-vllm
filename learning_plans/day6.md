# Day 6: 工具函数与进阶特性

> **目标**: 理解支持功能和技术细节

---

## 1. 核心概念

### 1.1 工具函数的作用

工具函数为 nano-vllm 提供底层支持，主要包括：

```
┌─────────────────────────────────────────────────────────────┐
│                    工具函数架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  utils/context.py - 上下文管理                              │
│  ├── GlobalContext: 全局上下文单例                          │
│  ├── 在不同层之间传递配置和状态                              │
│  └── 管理 KV Cache 的全局引用                               │
│                                                             │
│  utils/loader.py - 模型加载器                               │
│  ├── 权重加载和转换                                         │
│  ├── 张量并行切分                                           │
│  └── 权重合并 (packed modules)                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 上下文管理的必要性

```
问题场景:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  模型层需要访问全局信息:                                     │
│                                                             │
│  Attention 层                                               │
│       │                                                     │
│       ├── 需要 KV Cache 引用 (存储在哪?)                     │
│       ├── 需要 block_size (每个块多大?)                      │
│       └── 需要 当前模式 (prefill or decode?)                 │
│                                                             │
│  问题: 如何在不传递大量参数的情况下共享这些信息?              │
│                                                             │
│  解决方案: 全局上下文 (Global Context)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 代码阅读：上下文管理

### 2.1 utils/context.py 核心结构

```python
from contextlib import contextmanager
from typing import Optional, Any, Dict

class GlobalContext:
    """
    全局上下文管理器

    使用单例模式，确保整个系统共享同一上下文
    """

    _instance: Optional['GlobalContext'] = None

    def __new__(cls) -> 'GlobalContext':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 配置参数
        self.block_size: int = 16

        # KV Cache 引用
        self.kv_cache: Dict[str, torch.Tensor] = {}

        # 当前运行模式
        self.is_prefill: bool = True

        # 其他全局状态
        self.max_seq_len: int = 0

        self._initialized = True

    @classmethod
    def get_instance(cls) -> 'GlobalContext':
        """获取全局上下文实例"""
        return cls()

    def set_kv_cache(self, layer_idx: int, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """设置指定层的 KV Cache"""
        self.kv_cache[f'{layer_idx}_k'] = k_cache
        self.kv_cache[f'{layer_idx}_v'] = v_cache

    def get_kv_cache(self, layer_idx: int) -> tuple:
        """获取指定层的 KV Cache"""
        k = self.kv_cache.get(f'{layer_idx}_k')
        v = self.kv_cache.get(f'{layer_idx}_v')
        return k, v


# 全局访问函数
def get_global_context() -> GlobalContext:
    return GlobalContext.get_instance()


def get_kv_cache(layer_idx: int) -> tuple:
    """便捷函数: 获取 KV Cache"""
    ctx = get_global_context()
    return ctx.get_kv_cache(layer_idx)


def get_block_size() -> int:
    """便捷函数: 获取 block size"""
    return get_global_context().block_size
```

### 2.2 上下文的使用场景

```python
# 在 attention.py 中使用上下文

class Qwen3Attention(nn.Module):

    def forward(self, hidden_states, position_ids, ...):
        # 获取全局配置
        block_size = get_block_size()

        # 获取 KV Cache
        k_cache, v_cache = get_kv_cache(self.layer_idx)

        if self.is_prefill:
            # Prefill: 写入 KV Cache
            self._write_to_cache(k, v, slot_mapping)
        else:
            # Decode: 从 KV Cache 读取
            output = flash_attn_with_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                block_tables=block_tables,
            )

        return output
```

### 2.3 上下文管理器模式

```python
@contextmanager
def set_prefill_mode():
    """设置 Prefill 模式的上下文管理器"""
    ctx = get_global_context()
    old_mode = ctx.is_prefill
    ctx.is_prefill = True
    try:
        yield
    finally:
        ctx.is_prefill = old_mode


@contextmanager
def set_decode_mode():
    """设置 Decode 模式的上下文管理器"""
    ctx = get_global_context()
    old_mode = ctx.is_prefill
    ctx.is_prefill = False
    try:
        yield
    finally:
        ctx.is_prefill = old_mode


# 使用示例
def run_model(hidden_states, ...):
    if is_prefill_phase:
        with set_prefill_mode():
            output = model(hidden_states, ...)
    else:
        with set_decode_mode():
            output = model(hidden_states, ...)
    return output
```

---

## 3. 代码阅读：模型加载器

### 3.1 utils/loader.py 核心结构

```python
import torch
from typing import Dict, Any
from collections import defaultdict

class ModelLoader:
    """
    模型加载器

    职责:
    1. 从 HuggingFace 格式加载权重
    2. 转换权重格式 (如果需要)
    3. 张量并行切分
    4. 权重合并 (packed modules)
    """

    # 需要合并的模块映射
    # Qwen3 中, q_proj, k_proj, v_proj 合并为 qkv_proj
    PACKED_MODULES_MAPPING = {
        'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
        'gate_up_proj': ['gate_proj', 'up_proj'],
    }

    def __init__(self, model_path: str, tp_size: int = 1):
        self.model_path = model_path
        self.tp_size = tp_size

    def load_weights(self, model: nn.Module) -> None:
        """
        加载权重到模型

        流程:
        1. 加载 HuggingFace 格式权重
        2. 合并 packed modules
        3. 张量并行切分
        4. 加载到模型
        """
        # 1. 加载原始权重
        state_dict = self._load_hf_weights()

        # 2. 合并 packed modules
        state_dict = self._pack_modules(state_dict)

        # 3. 张量并行切分
        if self.tp_size > 1:
            state_dict = self._split_weights_for_tp(state_dict)

        # 4. 加载到模型
        model.load_state_dict(state_dict, strict=False)

    def _load_hf_weights(self) -> Dict[str, torch.Tensor]:
        """加载 HuggingFace 格式的权重"""
        from safetensors import safe_open

        state_dict = {}

        # 遍历所有 safetensors 文件
        for file in self._get_weight_files():
            with safe_open(file, framework='pt', device='cpu') as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

        return state_dict

    def _pack_modules(self, state_dict: Dict) -> Dict:
        """
        合并 packed modules

        示例:
        model.layers.0.self_attn.q_proj.weight
        model.layers.0.self_attn.k_proj.weight
        model.layers.0.self_attn.v_proj.weight
        →
        model.layers.0.self_attn.qkv_proj.weight (concat)
        """
        new_state_dict = {}

        for name, tensor in state_dict.items():
            # 检查是否需要合并
            packed = False
            for packed_name, sub_modules in self.PACKED_MODULES_MAPPING.items():
                for sub_name in sub_modules:
                    if sub_name in name:
                        # 替换名称
                        new_name = name.replace(sub_name, packed_name)
                        packed = True
                        break
                if packed:
                    break

            if packed:
                # 累积到对应的 packed tensor
                if new_name not in new_state_dict:
                    new_state_dict[new_name] = []
                new_state_dict[new_name].append(tensor)
            else:
                new_state_dict[name] = tensor

        # 合并累积的 tensors
        for name, tensors in new_state_dict.items():
            if isinstance(tensors, list):
                new_state_dict[name] = torch.cat(tensors, dim=0)

        return new_state_dict

    def _split_weights_for_tp(self, state_dict: Dict) -> Dict:
        """
        张量并行切分权重

        Column Parallel: 沿 dim=0 切分
        Row Parallel: 沿 dim=1 切分
        """
        # 根据 TP rank 选择对应的权重切片
        tp_rank = get_tp_rank()

        new_state_dict = {}
        for name, tensor in state_dict.items():
            if self._is_column_parallel(name):
                # Column Parallel: 切分输出维度
                chunk_size = tensor.shape[0] // self.tp_size
                new_state_dict[name] = tensor[
                    tp_rank * chunk_size : (tp_rank + 1) * chunk_size
                ]
            elif self._is_row_parallel(name):
                # Row Parallel: 切分输入维度
                chunk_size = tensor.shape[1] // self.tp_size
                new_state_dict[name] = tensor[
                    :, tp_rank * chunk_size : (tp_rank + 1) * chunk_size
                ]
            else:
                # 不切分 (如 embedding, layernorm)
                new_state_dict[name] = tensor

        return new_state_dict
```

### 3.2 权重合并详解

```
┌─────────────────────────────────────────────────────────────┐
│                    权重合并过程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  HuggingFace 格式 (分开存储):                                │
│  model.layers.0.self_attn.q_proj.weight [num_heads*head_dim, hidden] │
│  model.layers.0.self_attn.k_proj.weight [num_kv_heads*head_dim, hidden] │
│  model.layers.0.self_attn.v_proj.weight [num_kv_heads*head_dim, hidden] │
│                                                             │
│  nano-vllm 格式 (合并存储):                                  │
│  model.layers.0.self_attn.qkv_proj.weight                  │
│  [num_heads*head_dim + 2*num_kv_heads*head_dim, hidden]    │
│                                                             │
│  合并过程:                                                  │
│  ┌────────────────┐                                         │
│  │   q_proj.weight │                                        │
│  ├────────────────┤                                         │
│  │   k_proj.weight │  →  concat(dim=0)  →  qkv_proj.weight  │
│  ├────────────────┤                                         │
│  │   v_proj.weight │                                        │
│  └────────────────┘                                         │
│                                                             │
│  优势:                                                      │
│  1. 一次矩阵乘法代替三次                                    │
│  2. 更好的内存局部性                                        │
│  3. 适合 FlashAttention                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 张量并行切分详解

```
┌─────────────────────────────────────────────────────────────┐
│                 张量并行权重切分                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Column Parallel (qkv_proj, gate_up_proj, lm_head):         │
│                                                             │
│  原始权重: [out_features, in_features]                      │
│  ┌─────────────────────────────────────┐                    │
│  │              Full Weight            │                    │
│  │  ┌─────────────┬─────────────┐      │                    │
│  │  │   GPU 0     │   GPU 1     │      │                    │
│  │  │ [0:O/2, :]  │ [O/2:O, :]  │      │                    │
│  │  └─────────────┴─────────────┘      │                    │
│  └─────────────────────────────────────┘                    │
│  切分维度: dim=0 (输出维度)                                 │
│                                                             │
│  Row Parallel (o_proj, down_proj):                          │
│                                                             │
│  原始权重: [out_features, in_features]                      │
│  ┌─────────────────────────────────────┐                    │
│  │              Full Weight            │                    │
│  │  ┌─────────────┬─────────────┐      │                    │
│  │  │ GPU 0       │ GPU 1       │      │                    │
│  │  │ [:, 0:I/2]  │ [:, I/2:I]  │      │                    │
│  │  └─────────────┴─────────────┘      │                    │
│  └─────────────────────────────────────┘                    │
│  切分维度: dim=1 (输入维度)                                 │
│                                                             │
│  Embedding, LayerNorm: 不切分 (所有 GPU 相同)               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. slot_mapping 详解

### 4.1 slot_mapping 的作用

```python
"""
slot_mapping 将每个 token 映射到 KV Cache 的物理存储位置

KV Cache 结构:
- 形状: [num_blocks, block_size, num_heads, head_dim]
- 每个 block 包含 block_size 个 slot
- slot_mapping 指定每个 token 写入/读取哪个 slot

计算方式:
slot = block_id * block_size + slot_in_block
"""

def compute_slot_mapping(
    block_tables: List[List[int]],  # 每个序列的 block 列表
    positions: List[int],            # 每个 token 的位置
    block_size: int,
) -> List[int]:
    """
    计算 slot_mapping

    示例:
    block_tables = [[0, 1, 2], [3, 4, 5]]  # 两个序列
    positions = [0, 1, 2, ..., 10]
    block_size = 4

    计算过程:
    position=0: block_idx=0, slot_in_block=0 → slot=0*4+0=0
    position=1: block_idx=0, slot_in_block=1 → slot=0*4+1=1
    position=4: block_idx=1, slot_in_block=0 → slot=1*4+0=4
    """
    slot_mapping = []

    for seq_idx, (block_table, pos) in enumerate(zip(block_tables, positions)):
        block_idx = pos // block_size
        slot_in_block = pos % block_size
        block_id = block_table[block_idx]
        slot = block_id * block_size + slot_in_block
        slot_mapping.append(slot)

    return slot_mapping
```

### 4.2 slot_mapping 在 Prefill 中的使用

```
┌─────────────────────────────────────────────────────────────┐
│               Prefill 阶段 slot_mapping                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入序列: "Hello World"                                     │
│  Token IDs: [1, 2, 3, 4, 5]                                 │
│  Block Table: [Block 7, Block 8]                            │
│                                                             │
│  KV Cache 写入位置:                                         │
│  ┌─────────────────────────────────────────┐               │
│  │ Token 1 (Hello) → Block 7, Slot 0      │               │
│  │ Token 2 ( )     → Block 7, Slot 1      │               │
│  │ Token 3 (W)     → Block 7, Slot 2      │               │
│  │ Token 4 (o)     → Block 7, Slot 3      │               │
│  │ Token 5 (rld)   → Block 8, Slot 0      │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
│  slot_mapping: [7*4+0, 7*4+1, 7*4+2, 7*4+3, 8*4+0]          │
│             = [28, 29, 30, 31, 32]                          │
│                                                             │
│  FlashAttention varlen 使用 slot_mapping:                   │
│  - 将 K, V 写入指定 slot                                    │
│  - 后续 decode 时可直接读取                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 前缀缓存完整流程

### 5.1 前缀缓存命中时的处理

```python
def handle_prefix_cache_hit(seq: Sequence, block_manager: BlockManager):
    """
    处理前缀缓存命中的序列

    流程:
    1. 检查前缀缓存命中
    2. 复用已缓存的 blocks
    3. 只计算未缓存的部分
    """

    # 1. 检查命中
    prefix_indices, cached_block_ids = block_manager.get_prefix_cache_hit(
        seq.prompt_token_ids
    )

    # 2. 设置序列的缓存信息
    seq.prefix_indices = prefix_indices
    seq.cached_block_ids = cached_block_ids

    # 3. 确定需要计算的 token 范围
    num_cached = len(prefix_indices)
    tokens_to_compute = seq.prompt_token_ids[num_cached:]

    if num_cached > 0:
        print(f"[Prefix Cache] Hit! Cached {num_cached} tokens")
        print(f"[Prefix Cache] Need to compute {len(tokens_to_compute)} tokens")

    return tokens_to_compute
```

### 5.2 前缀缓存命中时的数据准备

```
┌─────────────────────────────────────────────────────────────┐
│              前缀缓存命中时的 Prefill                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  请求: "你是一个AI助手。请回答: 什么是机器学习?"              │
│                                                             │
│  缓存状态:                                                  │
│  - "你是一个AI助手。" 已被缓存 (Block 5, ref_count=3)       │
│  - "请回答: " 未被缓存                                      │
│                                                             │
│  处理流程:                                                  │
│                                                             │
│  1. 检测缓存命中                                            │
│     prefix_indices = [0, 1, 2, 3, 4, 5, 6]  # 前7个token    │
│     cached_block_ids = [5, 6]                               │
│                                                             │
│  2. 设置 block_table                                        │
│     block_table = [5, 6, 10, 11]  # 共享 + 新分配           │
│                                                             │
│  3. 只计算未缓存的 tokens                                   │
│     input_ids = [8, 9, 10, ...]  # 跳过已缓存部分           │
│     position_ids = [7, 8, 9, ...]  # 从位置7开始            │
│                                                             │
│  4. slot_mapping 指向新分配的 blocks                        │
│     slot_mapping = [10*16+0, 10*16+1, ...]  # 写入新位置    │
│                                                             │
│  计算节省:                                                  │
│  - 原始: 计算 20 个 tokens                                  │
│  - 命中后: 只计算 13 个 tokens                              │
│  - 节省: 35%                                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 前缀缓存更新

```python
def update_prefix_cache(
    block_manager: BlockManager,
    seq: Sequence,
    new_token_ids: List[int],
):
    """
    更新前缀缓存

    在 Prefill 完成后，将新计算的 block 添加到缓存
    """
    # 获取新计算的 blocks
    new_block_start = len(seq.cached_block_ids)
    new_blocks = seq.block_table[new_block_start:]

    for i, block_id in enumerate(new_blocks):
        # 计算这个 block 的 hash
        block_start = (new_block_start + i) * block_manager.block_size
        block_tokens = new_token_ids[block_start:block_start + block_manager.block_size]

        # 累积 hash (包含之前所有 blocks)
        parent_hash = get_parent_hash(seq, new_block_start + i)
        block_hash = compute_block_hash(parent_hash, block_tokens)

        # 添加到缓存映射
        block_manager.hash_to_block_id[block_hash] = block_id

        print(f"[Prefix Cache] Added block {block_id} with hash {block_hash}")
```

---

## 6. 代码实践

### 6.1 调试上下文

```python
from utils.context import get_global_context, set_prefill_mode, set_decode_mode

# 检查全局上下文状态
ctx = get_global_context()
print(f"Block size: {ctx.block_size}")
print(f"Is prefill: {ctx.is_prefill}")
print(f"KV Cache keys: {list(ctx.kv_cache.keys())}")

# 使用上下文管理器
with set_prefill_mode():
    assert get_global_context().is_prefill == True
    # 执行 prefill 操作

with set_decode_mode():
    assert get_global_context().is_prefill == False
    # 执行 decode 操作
```

### 6.2 观察权重加载

```python
from utils.loader import ModelLoader

# 创建加载器
loader = ModelLoader(
    model_path="./models/Qwen3-0.6B",
    tp_size=1,
)

# 加载权重前
print("Loading weights...")

# 加载
loader.load_weights(model)

# 检查加载的权重
for name, param in model.named_parameters():
    if 'qkv_proj' in name:
        print(f"{name}: {param.shape}")
        # qkv_proj 合并了 q, k, v 的权重
        break
```

### 6.3 实践练习

**练习 1**: 实现简单的权重合并

```python
def pack_qkv_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    将分开的 q, k, v 权重合并

    输入:
    - state_dict['q_proj.weight']: [num_heads * head_dim, hidden]
    - state_dict['k_proj.weight']: [num_kv_heads * head_dim, hidden]
    - state_dict['v_proj.weight']: [num_kv_heads * head_dim, hidden]

    输出:
    - state_dict['qkv_proj.weight']: [(num_heads + 2*num_kv_heads) * head_dim, hidden]
    """
    q = state_dict.pop('q_proj.weight')
    k = state_dict.pop('k_proj.weight')
    v = state_dict.pop('v_proj.weight')

    state_dict['qkv_proj.weight'] = torch.cat([q, k, v], dim=0)
    return state_dict
```

**练习 2**: 实现 slot_mapping 计算

```python
def compute_slot_mapping(
    block_table: List[int],
    positions: List[int],
    block_size: int,
) -> List[int]:
    """
    计算 slot_mapping

    提示:
    1. 对于每个 position，找到对应的 block
    2. 计算 slot = block_id * block_size + (position % block_size)
    """
    slot_mapping = []
    for pos in positions:
        block_idx = pos // block_size
        slot_in_block = pos % block_size
        block_id = block_table[block_idx]
        slot = block_id * block_size + slot_in_block
        slot_mapping.append(slot)
    return slot_mapping

# 测试
block_table = [5, 10, 15]
positions = list(range(32))  # 32 个 token
block_size = 16

slot_mapping = compute_slot_mapping(block_table, positions, block_size)
print(f"Slot mapping: {slot_mapping}")
# 应该输出: [80, 81, ..., 95, 160, 161, ..., 175, 240, ...]
```

---

## 7. 调试技巧

### 7.1 添加详细日志

```python
# 在关键函数中添加日志

def load_weights(self, model):
    print(f"[Loader] Loading weights from {self.model_path}")
    print(f"[Loader] TP size: {self.tp_size}")

    state_dict = self._load_hf_weights()
    print(f"[Loader] Loaded {len(state_dict)} weight tensors")

    state_dict = self._pack_modules(state_dict)
    print(f"[Loader] After packing: {len(state_dict)} weight tensors")

    if self.tp_size > 1:
        state_dict = self._split_weights_for_tp(state_dict)
        print(f"[Loader] After TP split: rank {get_tp_rank()}")

    model.load_state_dict(state_dict, strict=False)
    print(f"[Loader] Weights loaded successfully")


def set_kv_cache(self, layer_idx, k_cache, v_cache):
    print(f"[Context] Setting KV cache for layer {layer_idx}")
    print(f"  K cache shape: {k_cache.shape}")
    print(f"  V cache shape: {v_cache.shape}")
    super().set_kv_cache(layer_idx, k_cache, v_cache)
```

### 7.2 验证权重加载

```python
def verify_weights(model, original_state_dict):
    """
    验证权重加载是否正确

    检查:
    1. 所有权重都已加载
    2. 数值没有异常 (NaN, Inf)
    3. 合并后的权重正确
    """
    model_state = model.state_dict()

    # 检查 NaN
    for name, param in model_state.items():
        if torch.isnan(param).any():
            print(f"WARNING: NaN found in {name}")
        if torch.isinf(param).any():
            print(f"WARNING: Inf found in {name}")

    # 检查 qkv_proj 合并
    for name, param in model_state.items():
        if 'qkv_proj' in name:
            expected_shape = param.shape
            print(f"{name}: {expected_shape}")

            # 验证合并是否正确
            if 'weight' in name:
                q_part = param[:num_heads * head_dim]
                # 验证 q 部分是否正确
```

---

## 8. 常见问题

### Q1: 为什么需要 packed modules？

```
传统方式 (分开):
┌─────────────────────────────────────┐
│ x → q_proj → Q                      │
│ x → k_proj → K  (3次独立矩阵乘法)   │
│ x → v_proj → V                      │
└─────────────────────────────────────┘

Packed 方式 (合并):
┌─────────────────────────────────────┐
│ x → qkv_proj → [Q, K, V] (1次合并) │
└─────────────────────────────────────┘

优势:
1. 减少 kernel 启动次数
2. 更好的内存访问局部性
3. 更适合 FlashAttention 的输入格式
```

### Q2: slot_mapping 在 GPU 上如何使用？

```python
# FlashAttention varlen 使用 slot_mapping

# 假设 input_ids = [token1, token2, token3, ...]
# 经过模型计算得到 K, V: [total_tokens, num_heads, head_dim]

# slot_mapping: [28, 29, 30, 31, 32, ...]
# 表示每个 token 的 K, V 应该写入哪个 slot

# 在 FlashAttention 内部:
# 将 K, V 按照 slot_mapping 写入 KV Cache
for i, slot in enumerate(slot_mapping):
    k_cache[slot] = k[i]
    v_cache[slot] = v[i]
```

### Q3: 如何处理多轮对话的前缀缓存？

```python
# 多轮对话示例

# 第一轮
prompt_1 = "你是一个AI助手。\n用户: 你好"
# → "你是一个AI助手。" 被缓存

# 第二轮
prompt_2 = "你是一个AI助手。\n用户: 你好\n助手: 你好！\n用户: 天气如何?"
# → "你是一个AI助手。\n用户: 你好\n助手: 你好！\n" 可能已被缓存
# → 只需要计算 "用户: 天气如何?"

# 实现要点:
# 1. 维护对话历史
# 2. 每次请求包含完整历史
# 3. Block Manager 自动检测和复用缓存
```

---

## 9. 知识检查点

### 基础问题

1. ❓ Global Context 的作用是什么？

2. ❓ 为什么需要将 q_proj, k_proj, v_proj 合并为 qkv_proj？

3. ❓ slot_mapping 的作用是什么？

### 进阶问题

4. ❓ 前缀缓存命中时，如何避免重复计算？

5. ❓ 张量并行时，不同类型的层如何切分权重？

6. ❓ 如何验证权重加载是否正确？

---

## 10. 代码走查清单

阅读工具函数时，关注：

**utils/context.py**:
- [ ] `GlobalContext`: 单例模式实现
- [ ] `set_kv_cache`, `get_kv_cache`: KV Cache 管理
- [ ] 上下文管理器: `set_prefill_mode`, `set_decode_mode`

**utils/loader.py**:
- [ ] `ModelLoader`: 加载流程
- [ ] `_pack_modules`: 权重合并逻辑
- [ ] `_split_weights_for_tp`: 张量并行切分

---

## 11. 扩展阅读

### 推荐资源

1. **HuggingFace safetensors 格式**
   - 理解模型权重存储格式

2. **Tensor Parallelism 实现细节**
   - Megatron-LM 的并行策略

3. **Prefix Caching 论文**
   - vLLM 的 Automatic Prefix Caching

### 相关代码

| 文件 | 核心类/函数 |
|------|------------|
| `utils/context.py` | `GlobalContext`, `get_global_context` |
| `utils/loader.py` | `ModelLoader`, `PACKED_MODULES_MAPPING` |
| `engine/model_runner.py` | `slot_mapping` 的使用 |

---

## 12. 下一步

完成 Day 6 后，你应该能够：
- ✅ 理解全局上下文的作用和使用方式
- ✅ 解释权重加载和合并的过程
- ✅ 描述 slot_mapping 的作用
- ✅ 理解前缀缓存的完整流程

**准备 Day 7**: 综合实践与深度理解
- 运行 Benchmark 分析性能
- 绘制完整数据流图

---

*预计学习时间: 2-3 小时*
