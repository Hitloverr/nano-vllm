# Day 3: KV Cache 块管理

> **目标**: 理解 Prefix Cache 和 Block 分配机制

---

## 1. 核心概念

### 1.1 KV Cache 基础

**什么是 KV Cache？**

在 Transformer 的自注意力计算中：

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

其中:
- Q (Query): 当前 token 的查询向量
- K (Key): 所有历史 token 的键向量
- V (Value): 所有历史 token 的值向量
```

**KV Cache 的作用**：
- 缓存每个 token 的 K 和 V 向量
- Decode 阶段只需计算新 token 的 Q、K、V
- 避免重复计算历史 token 的 K、V

```
┌─────────────────────────────────────────────────────────────┐
│                    KV Cache 工作原理                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Prefill 阶段:                                              │
│  ┌─────┬─────┬─────┬─────┐                                 │
│  │ T1  │ T2  │ T3  │ T4  │  ← 输入 tokens                   │
│  └──┬──┴──┬──┴──┬──┴──┬──┘                                 │
│     │     │     │     │                                     │
│     ▼     ▼     ▼     ▼                                     │
│  ┌─────┬─────┬─────┬─────┐                                 │
│  │ K1,V1│ K2,V2│ K3,V3│ K4,V4│  ← 计算并存入 KV Cache        │
│  └─────┴─────┴─────┴─────┘                                 │
│                                                             │
│  Decode 阶段 (生成 T5):                                      │
│  ┌─────┐                                                    │
│  │ T5  │  ← 只有新 token 需要计算                            │
│  └──┬──┘                                                    │
│     │                                                       │
│     ▼                                                       │
│  ┌─────┬─────┬─────┬─────┬─────┐                           │
│  │ K1,V1│ K2,V2│ K3,V3│ K4,V4│ K5,V5│  ← 复用 + 新增         │
│  └─────┴─────┴─────┴─────┴─────┘                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Block（块）的概念

**为什么需要 Block？**

传统方式的内存问题：
```
连续内存分配:
┌────────────────────────────────────────────┐
│ Seq1: ████████████████████ (已分配)        │
│ Seq2: ░░░░░░░░░░ (需要更多，但空间不连续)   │
│ Seq3: ████████ (已分配)                    │
└────────────────────────────────────────────┘
问题: 内存碎片，无法有效利用
```

**Block 方案（类似操作系统的分页）**：
```
Block 方式:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ B0  │ B1  │ B2  │ B3  │ B4  │ B5  │ B6  │ B7  │
│ S1  │ S2  │ S1  │ S3  │ S2  │ S1  │ S3  │ ... │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

Block 大小固定 (如 16 tokens):
- 可以非连续分配
- 支持内存共享
- 减少碎片
```

**Block 的结构**：
```python
class Block:
    block_id: int          # 块的唯一标识
    token_ids: List[int]   # 存储 token IDs (最多 block_size 个)
    ref_count: int         # 引用计数（用于共享）

    # KV Cache 实际存储在 GPU 上
    # Block 只是逻辑管理单元
```

### 1.3 Prefix Caching（前缀缓存）

**核心思想**：
- 多个请求可能有相同的 prompt 前缀
- 相同前缀的 KV Cache 可以共享
- 避免重复计算

```
┌─────────────────────────────────────────────────────────────┐
│                    Prefix Cache 示例                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  请求 1: "请翻译以下内容: Hello World"                        │
│          └── 前缀: "请翻译以下内容: "                         │
│                                                             │
│  请求 2: "请翻译以下内容: 你好世界"                           │
│          └── 前缀: "请翻译以下内容: " (相同!)                 │
│                                                             │
│  共享 Block:                                                │
│  ┌─────────────────────────────────┐                        │
│  │ "请翻译以下内容: "               │ ← Block 0,1,2          │
│  │ ref_count = 2                   │   被两个序列共享         │
│  └─────────────────────────────────┘                        │
│                                                             │
│  Seq1 专属: ┌────────────────┐                              │
│            │ "Hello World"   │                              │
│            └────────────────┘                               │
│                                                             │
│  Seq2 专属: ┌────────────────┐                              │
│            │ "你好世界"       │                              │
│            └────────────────┘                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 代码阅读：Block Manager

### 2.1 block_manager.py 核心结构

```python
class BlockManager:
    """管理 KV Cache 块的分配、释放和共享"""

    def __init__(self, config: Config):
        # 配置参数
        self.block_size = config.block_size      # 每个 block 存储的 token 数
        self.num_blocks = config.num_blocks      # 总 block 数量

        # 空闲块队列
        self.free_block_ids: Deque[int] = deque(
            range(self.num_blocks)
        )

        # 前缀缓存映射: token hash -> block_id
        self.hash_to_block_id: Dict[int, int] = {}

        # 引用计数: block_id -> count
        self.ref_count: Dict[int, int] = {}

        # Block 内容: block_id -> token_ids
        self.block_tables: Dict[int, List[int]] = {}
```

### 2.2 核心方法详解

**Block 分配**：

```python
def can_allocate(self, seq: Sequence) -> bool:
    """检查是否有足够的空闲 block 分配给序列"""
    # 计算需要的 block 数量
    num_required = self._get_num_required_blocks(seq)

    # 考虑前缀缓存命中
    num_cached = len(seq.prefix_indices) // self.block_size

    # 实际需要分配的 block 数
    num_new = num_required - num_cached

    return len(self.free_block_ids) >= num_new


def allocate(self, seq: Sequence) -> List[int]:
    """
    为序列分配 blocks
    返回: 分配的 block_id 列表
    """
    block_ids = []

    # 1. 处理前缀缓存命中的部分
    for cached_block_id in seq.cached_block_ids:
        block_ids.append(cached_block_id)
        self.ref_count[cached_block_id] += 1  # 增加引用计数

    # 2. 分配新的 blocks
    num_new_blocks = self._calc_new_blocks_needed(seq)
    for _ in range(num_new_blocks):
        block_id = self.free_block_ids.popleft()
        block_ids.append(block_id)
        self.ref_count[block_id] = 1

    return block_ids
```

**Block 释放**：

```python
def deallocate(self, seq: Sequence) -> None:
    """释放序列占用的 blocks"""
    for block_id in seq.block_table:
        self.ref_count[block_id] -= 1

        # 引用计数为 0 时，真正释放
        if self.ref_count[block_id] == 0:
            self.free_block_ids.append(block_id)
            del self.ref_count[block_id]

            # 清理缓存映射
            if block_id in self.hash_to_block_id.values():
                # 移除 hash 映射
                for k, v in list(self.hash_to_block_id.items()):
                    if v == block_id:
                        del self.hash_to_block_id[k]
```

**前缀缓存查询**：

```python
def get_prefix_cache_hit(
    self,
    token_ids: List[int]
) -> Tuple[List[int], List[int]]:
    """
    检查前缀缓存命中情况

    返回:
    - prefix_indices: 命中的 token 索引
    - cached_block_ids: 命中的 block IDs
    """
    prefix_indices = []
    cached_block_ids = []

    # 计算每个 block 的 hash
    for i in range(0, len(token_ids), self.block_size):
        block_tokens = token_ids[i:i + self.block_size]

        # 计算当前块的 hash (基于之前所有块的累积)
        block_hash = self._compute_hash(prefix_indices, block_tokens)

        # 检查是否命中缓存
        if block_hash in self.hash_to_block_id:
            cached_block_ids.append(self.hash_to_block_id[block_hash])
            prefix_indices.extend(range(i, i + len(block_tokens)))
        else:
            # 一旦未命中，后续不可能命中
            break

    return prefix_indices, cached_block_ids


def _compute_hash(
    self,
    parent_indices: List[int],
    token_ids: List[int]
) -> int:
    """
    计算块的哈希值
    哈希依赖于父块哈希和当前块内容
    """
    # 使用 parent hash 保证前缀完整性
    parent_hash = hash(tuple(parent_indices)) if parent_indices else 0
    content_hash = hash(tuple(token_ids))
    return hash((parent_hash, content_hash))
```

### 2.3 引用计数机制

```
引用计数工作流程:

初始状态:
┌──────────────────────────────────────────────────────────┐
│ Block 0: ref_count=0 (空闲)                              │
│ Block 1: ref_count=0 (空闲)                              │
│ Block 2: ref_count=0 (空闲)                              │
└──────────────────────────────────────────────────────────┘

分配给 Seq1:
┌──────────────────────────────────────────────────────────┐
│ Block 0: ref_count=1 ← Seq1                              │
│ Block 1: ref_count=1 ← Seq1                              │
│ Block 2: ref_count=0 (空闲)                              │
└──────────────────────────────────────────────────────────┘

Seq2 共享 Block 0,1 (前缀缓存命中):
┌──────────────────────────────────────────────────────────┐
│ Block 0: ref_count=2 ← Seq1, Seq2 (共享)                 │
│ Block 1: ref_count=2 ← Seq1, Seq2 (共享)                 │
│ Block 2: ref_count=1 ← Seq2                              │
└──────────────────────────────────────────────────────────┘

Seq1 完成，释放:
┌──────────────────────────────────────────────────────────┐
│ Block 0: ref_count=1 ← Seq2                              │
│ Block 1: ref_count=1 ← Seq2                              │
│ Block 2: ref_count=1 ← Seq2                              │
└──────────────────────────────────────────────────────────┘

Seq2 完成，释放:
┌──────────────────────────────────────────────────────────┐
│ Block 0: ref_count=0 (空闲)                              │
│ Block 1: ref_count=0 (空闲)                              │
│ Block 2: ref_count=0 (空闲)                              │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Hash 计算与缓存匹配

### 3.1 Hash 计算策略

```python
class BlockManager:

    def _compute_block_hash(
        self,
        parent_hash: int,
        token_ids: List[int],
    ) -> int:
        """
        计算块的哈希值

        关键点:
        1. 包含父块哈希 → 保证前缀完整性
        2. 包含当前块 token → 区分不同内容
        """
        # 方法 1: 简单哈希
        return hash((parent_hash, tuple(token_ids)))

        # 方法 2: 更安全的哈希 (防碰撞)
        # import hashlib
        # content = f"{parent_hash}:{token_ids}"
        # return int(hashlib.md5(content.encode()).hexdigest(), 16)
```

### 3.2 缓存匹配流程

```
┌─────────────────────────────────────────────────────────────┐
│                     前缀缓存匹配流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入: "请翻译以下内容: Hello"                                │
│  Tokens: [101, 234, 567, 890, 123, 456, 789]                │
│                                                             │
│  Step 1: 分块 (block_size = 4)                              │
│  ┌─────────────────┬─────────────────┐                      │
│  │ [101,234,567,890] │ [123,456,789,_] │                      │
│  │    Block 0       │    Block 1      │                      │
│  └─────────────────┴─────────────────┘                      │
│                                                             │
│  Step 2: 计算 Hash                                          │
│  Block 0: hash((), [101,234,567,890]) = H0                  │
│  Block 1: hash(H0, [123,456,789,_]) = H1                    │
│                                                             │
│  Step 3: 查找缓存                                           │
│  ┌─────────────────────────────────────────────┐           │
│  │ H0 → Block 5 (命中!)                        │           │
│  │ H1 → 未命中                                  │           │
│  └─────────────────────────────────────────────┘           │
│                                                             │
│  Step 4: 使用缓存 + 分配新块                                 │
│  ┌─────────────────┬─────────────────┐                      │
│  │  Block 5 (共享)  │  Block 10 (新)  │                      │
│  │   ref_count++   │   ref_count=1   │                      │
│  └─────────────────┴─────────────────┘                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 内存布局

### 4.1 GPU 内存结构

```python
# KV Cache 在 GPU 上的布局

# 形状: [num_blocks, block_size, num_heads, head_dim]
# 或: [num_blocks, num_heads, block_size, head_dim]

# 示例配置
num_layers = 24      # 模型层数
num_heads = 16       # 注意力头数
head_dim = 64        # 每个头的维度
block_size = 16      # 每个 block 的 token 数
num_blocks = 1024    # 总 block 数

# 每个 block 的内存
bytes_per_token = 2 * num_heads * head_dim * 2  # K + V, fp16
bytes_per_block = block_size * bytes_per_token
# = 16 * 2 * 16 * 64 * 2 = 65536 bytes = 64 KB per block per layer

# 总内存
total_memory = num_blocks * bytes_per_block * num_layers
# = 1024 * 64KB * 24 = 1.5 GB
```

### 4.2 Block Table 结构

```python
# 每个序列维护一个 block table

class Sequence:
    block_table: List[int]  # 该序列使用的 block IDs

# 示例
seq.block_table = [5, 5, 5, 10, 11, 12]
#                    ↑
#                 前 3 个 block 共享 Block 5 (前缀缓存)

# slot_mapping: 将 token 位置映射到 KV Cache 的物理位置
def get_slot_mapping(block_table: List[int], block_size: int):
    """生成 slot_mapping 用于 GPU 计算"""
    slot_mapping = []
    for block_idx, block_id in enumerate(block_table):
        for slot_in_block in range(block_size):
            slot = block_id * block_size + slot_in_block
            slot_mapping.append(slot)
    return slot_mapping
```

---

## 5. 代码实践

### 5.1 观察 Block 分配

```python
# 在 block_manager.py 中添加调试

def allocate(self, seq: Sequence) -> List[int]:
    print(f"[BlockManager] Allocating for seq {seq.seq_id}")
    print(f"  - Required blocks: {self._get_num_required_blocks(seq)}")
    print(f"  - Free blocks: {len(self.free_block_ids)}")

    block_ids = # ... 分配逻辑

    print(f"  - Allocated blocks: {block_ids}")
    return block_ids


def deallocate(self, seq: Sequence) -> None:
    print(f"[BlockManager] Deallocating for seq {seq.seq_id}")
    print(f"  - Blocks to release: {seq.block_table}")

    # ... 释放逻辑

    print(f"  - Free blocks after: {len(self.free_block_ids)}")
```

### 5.2 测试前缀缓存

```python
from nano_vllm import LLM, SamplingParams

llm = LLM(model="./models/Qwen3-0.6B")

# 相同前缀的请求
system_prompt = "你是一个有帮助的AI助手。请用简洁的语言回答问题。"

prompts = [
    f"{system_prompt} 用户问题1: 什么是机器学习?",
    f"{system_prompt} 用户问题2: 什么是深度学习?",
    f"{system_prompt} 用户问题3: 什么是自然语言处理?",
]

# 观察日志，确认前缀缓存命中
outputs = llm.generate(prompts, SamplingParams(max_tokens=50))
```

### 5.3 实践练习

**练习 1**: 计算内存需求

```python
def calculate_memory_requirements(
    model_name: str,
    max_model_len: int,
    block_size: int,
    gpu_memory_utilization: float,
):
    """
    计算需要的 GPU 内存和 block 数量

    提示:
    1. 获取模型配置 (num_layers, num_heads, head_dim)
    2. 计算每个 block 的内存
    3. 根据可用显存计算 block 数量
    """
    pass
```

**练习 2**: 实现简单的 Block 分配器

```python
class SimpleBlockAllocator:
    """简化版 Block 分配器"""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.allocated = {}  # seq_id -> block_ids

    def allocate(self, seq_id: int, num_tokens: int) -> List[int]:
        """分配 blocks"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            raise MemoryError("Not enough blocks")

        blocks = self.free_blocks[:num_blocks_needed]
        self.free_blocks = self.free_blocks[num_blocks_needed:]
        self.allocated[seq_id] = blocks
        return blocks

    def deallocate(self, seq_id: int) -> None:
        """释放 blocks"""
        if seq_id in self.allocated:
            self.free_blocks.extend(self.allocated[seq_id])
            del self.allocated[seq_id]
```

---

## 6. 性能优化

### 6.1 Block Size 选择

```
Block Size 对性能的影响:

┌─────────────────────────────────────────────────────────────┐
│ Block Size │ 优点                 │ 缺点                   │
├─────────────────────────────────────────────────────────────┤
│ 小 (8)     │ 更细粒度的内存管理    │ 更多的管理开销         │
│            │ 更好的共享效率        │ 更多的 hash 计算       │
├─────────────────────────────────────────────────────────────┤
│ 中 (16)    │ 平衡的选择           │ -                      │
│            │ (推荐)               │                        │
├─────────────────────────────────────────────────────────────┤
│ 大 (32)    │ 更少的管理开销       │ 更多的内存浪费         │
│            │ 更快的分配           │ 共享效率降低           │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 内存利用率优化

```python
# 配置建议

# 高吞吐场景
config = Config(
    gpu_memory_utilization=0.95,  # 最大化显存使用
    block_size=16,
)

# 低延迟场景 (预留空间避免抢占)
config = Config(
    gpu_memory_utilization=0.8,
    block_size=16,
)

# 长序列场景
config = Config(
    gpu_memory_utilization=0.9,
    block_size=32,  # 减少管理开销
    max_model_len=32768,
)
```

---

## 7. 关键问题解答

### Q1: Block 和 Slot 的区别是什么？

```
Block:
- 逻辑管理单元
- 包含 block_size 个 token 的空间
- 用于分配和共享管理

Slot:
- 物理存储位置
- 一个 slot 存储一个 token 的 K/V
- slot_mapping 指定每个 token 的存储位置

关系:
Block 0 包含 Slot 0-15 (假设 block_size=16)
Block 1 包含 Slot 16-31
...
```

### Q2: 前缀缓存如何保证正确性？

```
关键点:
1. Hash 链式依赖
   - 当前块的 hash 包含父块的 hash
   - 保证前缀的完整性

2. 内容验证
   - 存储 block_id -> token_ids 映射
   - 运行时验证内容一致

3. 引用计数
   - 共享 block 不会被过早释放
   - 最后一个使用者释放后才回收
```

### Q3: 为什么不用连续内存？

```
连续内存的问题:
1. 内存碎片
   - 长期运行后，空闲内存不连续
   - 无法分配大块连续内存

2. 无法共享
   - 每个序列独立分配
   - 相同前缀无法复用

3. 难以动态扩展
   - 需要预分配最大长度
   - 浪费内存

Block 方案的优势:
1. 按需分配，减少浪费
2. 支持非连续内存
3. 支持共享和复用
```

---

## 8. 流程图解

### 8.1 Block 分配流程

```
┌─────────────────────────────────────────────────────────────┐
│                     Block 分配流程                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Sequence 到达                                              │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────┐                                    │
│  │ 计算需要的 block 数量 │                                   │
│  └──────────┬──────────┘                                    │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │ 检查前缀缓存命中      │                                   │
│  └──────────┬──────────┘                                    │
│             │                                               │
│      ┌──────┴──────┐                                        │
│      │             │                                        │
│   命中           未命中                                      │
│      │             │                                        │
│      ▼             │                                        │
│  ┌─────────┐       │                                        │
│  │复用 blocks│       │                                        │
│  │ref_count++│       │                                        │
│  └────┬────┘       │                                        │
│       │             │                                        │
│       └──────┬──────┘                                        │
│              │                                              │
│              ▼                                              │
│  ┌─────────────────────┐                                    │
│  │ 分配新 blocks        │                                    │
│  │ (从 free_block_ids) │                                    │
│  └──────────┬──────────┘                                    │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │ 返回 block_table     │                                    │
│  └─────────────────────┘                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 内存共享示意图

```
┌─────────────────────────────────────────────────────────────┐
│                     内存共享示意                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  GPU Memory (KV Cache):                                     │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐        │
│  │ B0  │ B1  │ B2  │ B3  │ B4  │ B5  │ B6  │ B7  │        │
│  └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴─────┴─────┘        │
│     │     │     │     │     │     │                        │
│     │     │     │     │     │     └────── Seq3 专属        │
│     │     │     │     │     └────────── Seq2 专属          │
│     │     │     │     └──────────────── Seq1 专属          │
│     │     │     └────────────────────── 共享 (Seq1,2,3)    │
│     │     └────────────────────────── 共享 (Seq1,2,3)      │
│     └────────────────────────────────── 共享 (Seq1,2,3)    │
│                                                             │
│  Block Tables:                                              │
│  Seq1: [B0, B1, B2, B3]                                     │
│  Seq2: [B0, B1, B2, B4, B5]                                 │
│  Seq3: [B0, B1, B2, B6]                                     │
│                                                             │
│  引用计数:                                                   │
│  B0: 3, B1: 3, B2: 3, B3: 1, B4: 1, B5: 1, B6: 1           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. 知识检查点

### 基础问题

1. ❓ KV Cache 存储的是什么？为什么它能加速推理？

2. ❓ Block 是什么？为什么使用 Block 而不是连续内存？

3. ❓ 引用计数的作用是什么？

### 进阶问题

4. ❓ 前缀缓存的 Hash 计算为什么需要包含父块的 Hash？

5. ❓ Block Size 的选择需要考虑哪些因素？

6. ❓ 当 Block 不足时，Block Manager 会怎么处理？

---

## 10. 代码走查清单

阅读 `engine/block_manager.py` 时，关注：

- [ ] `__init__`: 初始化数据结构
- [ ] `can_allocate`: 资源检查逻辑
- [ ] `allocate`: 分配流程，包括缓存命中处理
- [ ] `deallocate`: 释放流程，引用计数管理
- [ ] `get_prefix_cache_hit`: 缓存匹配逻辑
- [ ] `_compute_hash`: Hash 计算

---

## 11. 扩展阅读

### 推荐论文

1. **vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention**
   - PagedAttention 的核心论文
   - Block 管理的理论基础

2. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
   - 理解 KV Cache 在注意力计算中的作用

### 相关代码

| 文件 | 关键类/函数 |
|------|------------|
| `engine/block_manager.py` | `BlockManager` |
| `engine/sequence.py` | `Sequence.block_table` |
| `engine/model_runner.py` | `slot_mapping` 的使用 |

---

## 12. 下一步

完成 Day 3 后，你应该能够：
- ✅ 解释 KV Cache 的作用和 Block 管理原理
- ✅ 理解前缀缓存的实现机制
- ✅ 描述 Block 分配和释放的流程
- ✅ 理解引用计数的作用

**准备 Day 4**: 模型运行器
- 深入理解 Prefill 和 Decode 的数据准备
- 学习 CUDA Graph 的捕获和回放

---

*预计学习时间: 2-3 小时*
