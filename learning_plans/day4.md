# Day 4: 模型运行器

> **目标**: 理解模型推理的执行流程

---

## 1. 核心概念

### 1.1 Model Runner 的职责

Model Runner 是推理引擎的核心执行组件，负责：

```
┌─────────────────────────────────────────────────────────────┐
│                   Model Runner 职责                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 数据准备                                                 │
│     ├── prepare_prefill(): Prefill 阶段数据                  │
│     └── prepare_decode(): Decode 阶段数据                    │
│                                                             │
│  2. 模型执行                                                 │
│     ├── run_prefill(): 执行 Prefill 推理                     │
│     └── run_decode(): 执行 Decode 推理                       │
│                                                             │
│  3. CUDA Graph 管理                                          │
│     ├── capture_cudagraph(): 捕获计算图                      │
│     └── replay_cudagraph(): 重放计算图                       │
│                                                             │
│  4. Tensor Parallel 通信                                     │
│     └── 与其他 GPU 进程协调                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Prefill vs Decode 的数据准备差异

```
┌─────────────────────────────────────────────────────────────┐
│              Prefill vs Decode 数据准备对比                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Prefill 阶段:                                              │
│  ┌─────────────────────────────────────────────┐           │
│  │ 输入: 多个序列的完整 prompt                  │           │
│  │ 特点:                                       │           │
│  │   - 每个序列长度可能不同                    │           │
│  │   - 使用 FlashAttention varlen             │           │
│  │   - 不使用 CUDA Graph (动态长度)            │           │
│  │                                             │           │
│  │ 数据:                                       │           │
│  │   - input_ids: [seq1_tokens, seq2_tokens]  │           │
│  │   - position_ids: 每个序列从 0 开始         │           │
│  │   - cu_seqlens: 累积序列长度               │           │
│  └─────────────────────────────────────────────┘           │
│                                                             │
│  Decode 阶段:                                               │
│  ┌─────────────────────────────────────────────┐           │
│  │ 输入: 每个序列的最新 token                  │           │
│  │ 特点:                                       │           │
│  │   - 所有序列长度相同 (1 token)             │           │
│  │   - 使用 FlashAttention with_kvcache       │           │
│  │   - 使用 CUDA Graph 优化                   │           │
│  │                                             │           │
│  │ 数据:                                       │           │
│  │   - input_ids: [batch_size, 1]             │           │
│  │   - position_ids: [batch_size] (各自位置)   │           │
│  │   - block_tables: KV Cache 位置映射        │           │
│  └─────────────────────────────────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 CUDA Graph 简介

**什么是 CUDA Graph？**

CUDA Graph 允许将一系列 GPU kernel 操作录制为计算图，然后一次性提交执行，减少 CPU-GPU 交互开销。

```
传统执行方式:
┌─────────────────────────────────────────────────────────────┐
│ CPU: ──kernel1──kernel2──kernel3──kernel4──kernel5──►       │
│ GPU:    ──exec1──│──exec2──│──exec3──│──exec4──│──exec5──►  │
│              ↑         ↑         ↑         ↑                │
│           启动开销   启动开销   启动开销   启动开销           │
└─────────────────────────────────────────────────────────────┘

CUDA Graph 执行:
┌─────────────────────────────────────────────────────────────┐
│ CPU: ──capture──│──────────────────────────────────────────►│
│ GPU:           ──graph execution (一次性)─────────────────► │
│                    ↑                                        │
│               只有一次启动开销                               │
└─────────────────────────────────────────────────────────────┘
```

**CUDA Graph 的限制**：
- 计算图结构必须固定
- 输入输出地址固定
- 只适用于固定 batch size 的 Decode 阶段

---

## 2. 代码阅读：Model Runner

### 2.1 model_runner.py 核心结构

```python
class ModelRunner:
    """模型推理执行器"""

    def __init__(self, model, config: Config):
        self.model = model           # 模型实例
        self.config = config

        # CUDA Graph 相关
        self.graphs: Dict[int, cudaGraph] = {}  # batch_size -> graph
        self.graph_buffers: Dict[int, Tensor] = {}  # 预分配的缓冲区

        # 预分配固定 batch size 的 buffer
        self.max_batch_size = 256
        self._allocate_graph_buffers()

    def run_model(
        self,
        scheduler_output: SchedulerOutput
    ) -> List[SamplerOutput]:
        """执行模型推理的主入口"""
        pass

    def prepare_prefill(
        self,
        sequences: List[Sequence]
    ) -> PrefillInput:
        """准备 Prefill 阶段的输入数据"""
        pass

    def prepare_decode(
        self,
        sequences: List[Sequence]
    ) -> DecodeInput:
        """准备 Decode 阶段的输入数据"""
        pass

    def capture_cudagraph(self, batch_size: int) -> None:
        """捕获指定 batch size 的 CUDA Graph"""
        pass

    def run_decode_with_graph(self, decode_input: DecodeInput) -> Tensor:
        """使用 CUDA Graph 执行 Decode"""
        pass
```

### 2.2 数据准备详解

**Prefill 数据准备**：

```python
def prepare_prefill(self, sequences: List[Sequence]) -> PrefillInput:
    """
    准备 Prefill 输入

    返回:
    - input_ids: 所有 token 拼接
    - position_ids: 每个 token 的位置
    - cu_seqlens: 累积序列长度 (用于 FlashAttention varlen)
    - slot_mapping: KV Cache 存储位置
    """
    input_ids = []
    position_ids = []
    cu_seqlens = [0]
    slot_mapping = []

    for seq in sequences:
        # 获取 prompt tokens
        tokens = seq.prompt_token_ids

        # 如果有前缀缓存命中，跳过已缓存的 tokens
        start_idx = len(seq.prefix_indices)
        tokens_to_process = tokens[start_idx:]

        # 添加到输入
        input_ids.extend(tokens_to_process)

        # 位置 ID (从缓存后的位置开始)
        positions = list(range(start_idx, len(tokens)))
        position_ids.extend(positions)

        # 累积序列长度
        cu_seqlens.append(cu_seqlens[-1] + len(tokens_to_process))

        # Slot mapping (KV Cache 写入位置)
        for i, block_id in enumerate(seq.block_table):
            block_start = i * self.config.block_size
            for j in range(self.config.block_size):
                slot_mapping.append(block_id * self.config.block_size + j)

    return PrefillInput(
        input_ids=torch.tensor(input_ids, device='cuda'),
        position_ids=torch.tensor(position_ids, device='cuda'),
        cu_seqlens=torch.tensor(cu_seqlens, device='cuda'),
        slot_mapping=torch.tensor(slot_mapping, device='cuda'),
    )
```

**Decode 数据准备**：

```python
def prepare_decode(self, sequences: List[Sequence]) -> DecodeInput:
    """
    准备 Decode 输入

    返回:
    - input_ids: 每个序列的最新 token [batch_size]
    - position_ids: 每个序列的当前位置 [batch_size]
    - block_tables: KV Cache 位置 [batch_size, max_blocks]
    - cache_slots: 当前 token 的缓存位置 [batch_size]
    """
    batch_size = len(sequences)

    input_ids = []
    position_ids = []
    block_tables = []
    cache_slots = []

    for seq in sequences:
        # 最新生成的 token
        input_ids.append(seq.output_token_ids[-1])

        # 当前位置 (prompt 长度 + 已生成长度 - 1)
        position_ids.append(len(seq.token_ids) - 1)

        # Block table
        block_tables.append(seq.block_table)

        # 当前 token 要写入的 cache slot
        block_idx = len(seq.token_ids) // self.config.block_size
        slot_in_block = len(seq.token_ids) % self.config.block_size
        cache_slots.append(seq.block_table[block_idx] * self.config.block_size + slot_in_block)

    return DecodeInput(
        input_ids=torch.tensor(input_ids, device='cuda'),
        position_ids=torch.tensor(position_ids, device='cuda'),
        block_tables=torch.tensor(block_tables, device='cuda'),
        cache_slots=torch.tensor(cache_slots, device='cuda'),
    )
```

### 2.3 CUDA Graph 捕获

```python
def capture_cudagraph(self, batch_size: int) -> None:
    """
    捕获指定 batch size 的 CUDA Graph

    注意:
    - 只能捕获固定 batch size 的计算
    - 需要预分配输入输出缓冲区
    - 捕获后可重复使用，无需重新捕获
    """
    # 1. 预分配固定大小的输入缓冲区
    input_ids = torch.zeros(batch_size, dtype=torch.long, device='cuda')
    position_ids = torch.zeros(batch_size, dtype=torch.long, device='cuda')
    block_tables = torch.zeros(
        batch_size, self.max_blocks_per_seq,
        dtype=torch.long, device='cuda'
    )

    # 2. 开始捕获
    torch.cuda.graph(self.graphs[batch_size])

    # 3. 执行模型前向传播 (会被记录到 graph 中)
    output = self.model.forward_decode(
        input_ids=input_ids,
        position_ids=position_ids,
        block_tables=block_tables,
    )

    # 4. 结束捕获
    self.graphs[batch_size].capture_end()

    # 5. 保存输出引用
    self.graph_outputs[batch_size] = output


def run_decode_with_graph(self, batch_size: int, decode_input: DecodeInput):
    """使用已捕获的 CUDA Graph 执行"""
    # 1. 复制输入到预分配缓冲区
    self.graph_buffers[batch_size]['input_ids'].copy_(decode_input.input_ids)
    self.graph_buffers[batch_size]['position_ids'].copy_(decode_input.position_ids)
    self.graph_buffers[batch_size]['block_tables'].copy_(decode_input.block_tables)

    # 2. 重放 Graph
    self.graphs[batch_size].replay()

    # 3. 返回输出 (直接使用预分配的输出缓冲区)
    return self.graph_outputs[batch_size]
```

### 2.4 运行模型

```python
def run_model(self, scheduler_output: SchedulerOutput) -> List[SamplerOutput]:
    """执行模型推理的主入口"""

    outputs = []

    # 1. 执行 Prefill (如果有)
    if scheduler_output.prefill_seqs:
        prefill_input = self.prepare_prefill(scheduler_output.prefill_seqs)

        # 直接执行，不使用 CUDA Graph
        hidden_states = self.model.forward_prefill(
            input_ids=prefill_input.input_ids,
            position_ids=prefill_input.position_ids,
            cu_seqlens=prefill_input.cu_seqlens,
            slot_mapping=prefill_input.slot_mapping,
        )

        # 采样
        for seq, hidden in zip(scheduler_output.prefill_seqs, hidden_states):
            next_token = self.sampler.sample(hidden, seq.sampling_params)
            outputs.append(SamplerOutput(seq.seq_id, next_token))

    # 2. 执行 Decode (如果有)
    if scheduler_output.decode_seqs:
        decode_input = self.prepare_decode(scheduler_output.decode_seqs)
        batch_size = len(scheduler_output.decode_seqs)

        # 选择合适的 CUDA Graph
        if batch_size in self.graphs:
            hidden_states = self.run_decode_with_graph(batch_size, decode_input)
        else:
            # 没有 captured graph，直接执行
            hidden_states = self.model.forward_decode(
                input_ids=decode_input.input_ids,
                position_ids=decode_input.position_ids,
                block_tables=decode_input.block_tables,
            )

        # 采样
        for seq, hidden in zip(scheduler_output.decode_seqs, hidden_states):
            next_token = self.sampler.sample(hidden, seq.sampling_params)
            outputs.append(SamplerOutput(seq.seq_id, next_token))

    return outputs
```

---

## 3. Tensor Parallel 通信

### 3.1 Tensor Parallel 原理

```
┌─────────────────────────────────────────────────────────────┐
│                   Tensor Parallel 示意图                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  单 GPU:                                                    │
│  ┌─────────────────────────────────────────┐               │
│  │ Linear(weight: [4096, 4096])            │               │
│  │ 完整的计算                              │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
│  2 GPU Tensor Parallel:                                     │
│  ┌───────────────────┐     ┌───────────────────┐           │
│  │ GPU 0             │     │ GPU 1             │           │
│  │ Linear(weight:    │     │ Linear(weight:    │           │
│  │   [2048, 4096])   │     │   [2048, 4096])   │           │
│  │ 上半部分计算       │     │ 下半部分计算       │           │
│  └─────────┬─────────┘     └─────────┬─────────┘           │
│            │                         │                      │
│            └───────────┬─────────────┘                      │
│                        │                                    │
│                        ▼                                    │
│                  All-Reduce                                 │
│                  (结果合并)                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 通信原语

```python
import torch.distributed as dist

def all_reduce(tensor: torch.Tensor) -> None:
    """
    All-Reduce: 所有 GPU 汇总结果并广播

    示例:
    GPU 0: [1, 2]    GPU 1: [3, 4]
    All-Reduce 后:
    GPU 0: [4, 6]    GPU 1: [4, 6]  # 求和
    """
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def all_gather(tensors: List[torch.Tensor]) -> None:
    """
    All-Gather: 收集所有 GPU 的张量

    示例:
    GPU 0: [A]    GPU 1: [B]
    All-Gather 后:
    GPU 0: [A, B]    GPU 1: [A, B]
    """
    dist.all_gather(tensors, tensors[dist.get_rank()])
```

### 3.3 Tensor Parallel 进程模型

```
┌─────────────────────────────────────────────────────────────┐
│                  Tensor Parallel 进程架构                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  主进程 (Rank 0):                                           │
│  ┌─────────────────────────────────────────────┐           │
│  │ - 接收用户请求                               │           │
│  │ - 运行 Scheduler                             │           │
│  │ - 准备输入数据                               │           │
│  │ - 发送指令给 Worker 进程                     │           │
│  │ - 收集输出                                   │           │
│  └─────────────────────────────────────────────┘           │
│                        │                                    │
│         ┌──────────────┼──────────────┐                    │
│         │              │              │                     │
│         ▼              ▼              ▼                     │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐              │
│  │ Worker 0  │  │ Worker 1  │  │ Worker 2  │              │
│  │ (Rank 0)  │  │ (Rank 1)  │  │ (Rank 2)  │              │
│  │           │  │           │  │           │              │
│  │ GPU 0     │  │ GPU 1     │  │ GPU 2     │              │
│  │ 模型分片  │  │ 模型分片  │  │ 模型分片  │              │
│  └───────────┘  └───────────┘  └───────────┘              │
│                                                             │
│  通信方式: NCCL (NVIDIA Collective Communications Library)  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 代码示例

```python
class TensorParallelContext:
    """Tensor Parallel 上下文管理"""

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank

    def run_model_parallel(self, model_input):
        """
        在多个 GPU 上并行执行模型

        流程:
        1. 主进程广播输入数据
        2. 各 Worker 并行计算
        3. All-Reduce 汇总结果
        """
        # 广播输入
        dist.broadcast(model_input.input_ids, src=0)
        dist.broadcast(model_input.position_ids, src=0)

        # 执行模型分片
        hidden_states = self.model.forward(model_input)

        # All-Reduce 汇总
        dist.all_reduce(hidden_states)

        return hidden_states
```

---

## 4. FlashAttention 集成

### 4.1 两种调用方式

```python
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

# Prefill 阶段: varlen (变长序列)
def forward_prefill(self, q, k, v, cu_seqlens, max_seqlen):
    """
    cu_seqlens: [0, len1, len1+len2, len1+len2+len3, ...]
    max_seqlen: 最大序列长度
    """
    output = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=1.0 / math.sqrt(head_dim),
    )
    return output


# Decode 阶段: with_kvcache (使用 KV Cache)
def forward_decode(self, q, k_cache, v_cache, block_tables, cache_seqlens):
    """
    q: 当前 token 的 query [batch, 1, heads, head_dim]
    k_cache, v_cache: KV Cache [num_blocks, block_size, heads, head_dim]
    block_tables: [batch, max_blocks] 每个 seq 的 block 位置
    cache_seqlens: [batch] 每个seq 的缓存长度
    """
    output = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_tables,
        cache_seqlens=cache_seqlens,
        softmax_scale=1.0 / math.sqrt(head_dim),
    )
    return output
```

### 4.2 KV Cache 访问模式

```
┌─────────────────────────────────────────────────────────────┐
│                  FlashAttention KV Cache 访问               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  KV Cache 布局: [num_blocks, block_size, heads, head_dim]   │
│                                                             │
│  Block Table (示例):                                        │
│  Seq 0: [Block 2, Block 5, Block 8]                        │
│  Seq 1: [Block 1, Block 3, Block 7]                        │
│                                                             │
│  访问 Seq 0 的 KV Cache:                                    │
│  ┌─────────┬─────────┬─────────┐                           │
│  │ Block 2 │ Block 5 │ Block 8 │                           │
│  │ T0-T15  │ T16-T31 │ T32-T47 │                           │
│  └─────────┴─────────┴─────────┘                           │
│                                                             │
│  FlashAttention 通过 block_table 直接访问:                  │
│  1. 根据 block_table 找到物理位置                           │
│  2. 使用 cache_seqlens 确定有效长度                         │
│  3. 高效计算注意力                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 代码实践

### 5.1 添加调试日志

```python
# 在 model_runner.py 中添加

def run_model(self, scheduler_output: SchedulerOutput):
    print(f"[ModelRunner] Prefill seqs: {len(scheduler_output.prefill_seqs)}")
    print(f"[ModelRunner] Decode seqs: {len(scheduler_output.decode_seqs)}")

    # Prefill
    if scheduler_output.prefill_seqs:
        prefill_input = self.prepare_prefill(scheduler_output.prefill_seqs)
        print(f"[ModelRunner] Prefill input shape: {prefill_input.input_ids.shape}")
        print(f"[ModelRunner] Prefill cu_seqlens: {prefill_input.cu_seqlens}")

    # Decode
    if scheduler_output.decode_seqs:
        decode_input = self.prepare_decode(scheduler_output.decode_seqs)
        print(f"[ModelRunner] Decode batch size: {len(scheduler_output.decode_seqs)}")
        print(f"[ModelRunner] Using CUDA Graph: {len(scheduler_output.decode_seqs) in self.graphs}")
```

### 5.2 观察 CUDA Graph 捕获

```python
# 捕获多个 batch size 的 CUDA Graph
for batch_size in [1, 2, 4, 8, 16, 32, 64]:
    print(f"Capturing CUDA Graph for batch_size={batch_size}")
    self.capture_cudagraph(batch_size)
    print(f"  Captured successfully")
```

### 5.3 实践练习

**练习 1**: 理解输入输出形状

```python
def analyze_shapes():
    """
    分析 Prefill 和 Decode 的输入输出形状

    提示:
    - Prefill: input_ids 是 1D (拼接所有序列)
    - Decode: input_ids 是 1D (batch_size)
    """
    # 模拟数据
    seqs = [
        Sequence(prompt="Hello", token_ids=[1, 2, 3, 4, 5]),
        Sequence(prompt="World", token_ids=[6, 7, 8, 9]),
    ]

    # 分析 prefill input
    # 分析 decode input
```

**练习 2**: 手动实现 CUDA Graph

```python
import torch

def simple_cuda_graph_example():
    """简单的 CUDA Graph 示例"""

    # 预分配缓冲区
    x = torch.zeros(100, device='cuda')
    y = torch.zeros(100, device='cuda')

    # 捕获
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y.copy_(x * 2 + 1)

    # 使用
    x.fill_(1.0)
    g.replay()
    print(y)  # 应该是 [3, 3, 3, ...]
```

---

## 6. 性能优化

### 6.1 CUDA Graph Batch Size 策略

```python
# 策略 1: 预捕获常用 batch size
def capture_common_batch_sizes(self):
    """捕获常用的 batch size"""
    common_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for bs in common_sizes:
        if bs <= self.max_batch_size:
            self.capture_cudagraph(bs)

# 策略 2: 动态捕获
def get_or_capture_graph(self, batch_size: int):
    """获取或捕获 CUDA Graph"""
    if batch_size not in self.graphs:
        self.capture_cudagraph(batch_size)
    return self.graphs[batch_size]

# 策略 3: 使用最接近的已捕获 batch size (padding)
def get_nearest_graph_batch_size(self, batch_size: int) -> int:
    """找到最近的已捕获 batch size"""
    available = sorted(self.graphs.keys())
    for size in available:
        if size >= batch_size:
            return size
    return available[-1]
```

### 6.2 内存优化

```python
# 预分配内存池
class MemoryPool:
    """预分配内存池，避免动态分配"""

    def __init__(self, max_batch_size: int, hidden_dim: int):
        # 预分配所有可能需要的缓冲区
        self.hidden_states = torch.zeros(
            max_batch_size, hidden_dim,
            device='cuda', dtype=torch.float16
        )
        self.logits = torch.zeros(
            max_batch_size, vocab_size,
            device='cuda', dtype=torch.float16
        )
```

---

## 7. 流程图解

### 7.1 完整推理流程

```
┌─────────────────────────────────────────────────────────────┐
│                    完整推理流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Scheduler.schedule()                                       │
│       │                                                     │
│       ▼                                                     │
│  SchedulerOutput                                            │
│  ├── prefill_seqs: [...]                                   │
│  └── decode_seqs: [...]                                    │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────┐           │
│  │           ModelRunner.run_model()            │           │
│  └─────────────────────────────────────────────┘           │
│       │                                                     │
│       ├─────────────────┬─────────────────┐                │
│       │                 │                 │                 │
│       ▼                 ▼                 ▼                 │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐            │
│  │Prefill? │      │Decode?  │      │ 两者都有 │            │
│  └────┬────┘      └────┬────┘      └────┬────┘            │
│       │                │                │                  │
│       ▼                ▼                ▼                  │
│  prepare_prefill  prepare_decode  分别执行                 │
│       │                │                                    │
│       ▼                ▼                                    │
│  model.forward    model.forward                             │
│  (varlen)         (with_kvcache)                            │
│       │                │                                    │
│       │                ▼                                    │
│       │         CUDA Graph?                                 │
│       │          ├── Yes: replay()                          │
│       │          └── No: forward()                          │
│       │                │                                    │
│       └────────────────┼────────────────┘                  │
│                        │                                    │
│                        ▼                                    │
│                  Sampler.sample()                           │
│                        │                                    │
│                        ▼                                    │
│              List[SamplerOutput]                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 CUDA Graph 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                  CUDA Graph 工作流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  初始化阶段 (只执行一次):                                    │
│  ┌─────────────────────────────────────────────┐           │
│  │ for batch_size in [1, 2, 4, 8, ...]:        │           │
│  │     1. 预分配输入输出缓冲区                  │           │
│  │     2. 开始捕获 (torch.cuda.graph)          │           │
│  │     3. 执行模型前向传播                      │           │
│  │     4. 结束捕获                              │           │
│  │     5. 保存 graph 和 buffers                 │           │
│  └─────────────────────────────────────────────┘           │
│                                                             │
│  推理阶段 (每次 decode):                                     │
│  ┌─────────────────────────────────────────────┐           │
│  │ 1. 确定 batch_size                           │           │
│  │ 2. 复制输入到预分配缓冲区                    │           │
│  │ 3. graph.replay()                            │           │
│  │ 4. 读取预分配缓冲区的输出                    │           │
│  └─────────────────────────────────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 关键问题解答

### Q1: 为什么 Prefill 不使用 CUDA Graph？

```
原因:
1. Prefill 的序列长度是动态的
   - 每个 prompt 长度不同
   - CUDA Graph 要求固定输入形状

2. FlashAttention varlen 需要动态参数
   - cu_seqlens 每次可能不同
   - max_seqlen 每次可能不同

3. Prefill 计算量大
   - kernel 执行时间长
   - 启动开销占比小
   - 优化收益不明显
```

### Q2: CUDA Graph 如何处理不同 batch size？

```
解决方案:

1. 预捕获多个 batch size
   - 捕获 [1, 2, 4, 8, 16, 32, ...] 的 graph

2. Padding 到最近的 batch size
   - 实际 batch=5 时，使用 batch=8 的 graph
   - 填充 dummy 数据

3. 动态捕获
   - 遇到新的 batch size 时捕获
   - 缓存以备后用

代码示例:
batch_size = 5
target_batch = 8  # 向上取整到已捕获的大小
padding = target_batch - batch_size

# 填充 dummy 数据
input_ids = torch.cat([
    real_input_ids,
    torch.zeros(padding, dtype=torch.long, device='cuda')
])
```

### Q3: Tensor Parallel 的通信开销如何？

```
通信开销分析:

每次前向传播的通信:
- 每个 Transformer 层: 1 次 All-Reduce
- All-Reduce 复杂度: O(hidden_dim)

优化策略:
1. Overlap 通信和计算
   - 在计算下一层时进行通信

2. 使用 NCCL 优化
   - NCCL 自动选择最优通信路径
   - Ring All-Reduce / Tree All-Reduce

3. 减少通信频率
   - Gradient accumulation (训练时)
   - 减少 All-Reduce 次数

实际影响:
- 2 GPU: ~5% 开销
- 4 GPU: ~10% 开销
- 8 GPU: ~15% 开销
```

---

## 9. 知识检查点

### 基础问题

1. ❓ Model Runner 的主要职责是什么？

2. ❓ Prefill 和 Decode 的数据准备有什么不同？

3. ❓ CUDA Graph 为什么能加速推理？

### 进阶问题

4. ❓ 为什么 Prefill 阶段不能使用 CUDA Graph？

5. ❓ Tensor Parallel 中 All-Reduce 的作用是什么？

6. ❓ 如何选择要捕获的 CUDA Graph batch size？

---

## 10. 代码走查清单

阅读 `engine/model_runner.py` 时，关注：

- [ ] `__init__`: CUDA Graph 缓冲区预分配
- [ ] `run_model`: 整体执行流程
- [ ] `prepare_prefill`: Prefill 数据准备细节
- [ ] `prepare_decode`: Decode 数据准备细节
- [ ] `capture_cudagraph`: CUDA Graph 捕获过程
- [ ] `run_decode_with_graph`: Graph 重放

---

## 11. 扩展阅读

### 推荐资源

1. **CUDA Graphs 官方文档**
   - 理解 CUDA Graph 的原理和限制

2. **FlashAttention GitHub**
   - `flash_attn_varlen_func` 和 `flash_attn_with_kvcache` 的使用

3. **NCCL Documentation**
   - Tensor Parallel 通信的实现细节

### 相关代码

| 文件 | 关键函数 |
|------|----------|
| `engine/model_runner.py` | `run_model`, `prepare_prefill`, `prepare_decode` |
| `models/qwen3.py` | `forward_prefill`, `forward_decode` |
| `layers/attention.py` | FlashAttention 调用 |

---

## 12. 下一步

完成 Day 4 后，你应该能够：
- ✅ 解释 Model Runner 的执行流程
- ✅ 理解 Prefill 和 Decode 的数据准备差异
- ✅ 描述 CUDA Graph 的捕获和使用过程
- ✅ 理解 Tensor Parallel 的通信机制

**准备 Day 5**: 模型实现与层定义
- 深入理解 Qwen3 模型架构
- 学习各层的实现细节

---

*预计学习时间: 2-3 小时*
