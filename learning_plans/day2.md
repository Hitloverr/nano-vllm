# Day 2: 调度器与序列管理

> **目标**: 理解请求调度和序列状态管理

---

## 1. 核心概念

### 1.1 Prefill vs Decode 深入理解

**Prefill（预填充）阶段**:
- 处理完整的 prompt（所有输入 token）
- 一次性计算所有 token 的 KV Cache
- 计算密集型，GPU 利用率高
- 输出：第一个生成的 token

**Decode（解码）阶段**:
- 每次只处理一个 token
- 利用 KV Cache 避免重复计算
- 内存带宽密集型
- 输出：下一个生成的 token

```
┌─────────────────────────────────────────────────────────────┐
│                     推理过程示意                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Prefill 阶段:                                              │
│  ┌───┬───┬───┬───┬───┐                                     │
│  │ 你 │ 好 │ ， │ 世 │ 界 │  ← 输入 prompt (并行处理)        │
│  └───┴───┴───┴───┴───┘                                     │
│         │                                                   │
│         ▼                                                   │
│  ┌───┐                                                     │
│  │ ! │  ← 输出第一个 token                                  │
│  └───┘                                                     │
│                                                             │
│  Decode 阶段 (迭代生成):                                     │
│  Step 1: [!] → [今]                                         │
│  Step 2: [今] → [天]                                        │
│  Step 3: [天] → [是]                                        │
│  ... 直到生成结束                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Continuous Batching（连续批处理）

传统批处理 vs 连续批处理：

```
传统批处理 (Static Batching):
┌────────────────────────────────────────┐
│ Batch 1: [Seq1■, Seq2■, Seq3■, Seq4■] │  等待最长的完成
│ 时间: ████████████████████████████████ │
└────────────────────────────────────────┘
问题: 短序列完成后 GPU 空闲

Continuous Batching:
┌────────────────────────────────────────┐
│ T1: [Seq1■, Seq2■, Seq3■, Seq4■]      │
│ T2: [Seq1■, Seq2■, Seq3■, Seq4■]      │
│ T3: [Seq1■, Seq2░, Seq3■, Seq4■]      │  Seq2 完成，加入 Seq5
│ T4: [Seq1■, Seq5■, Seq3■, Seq4■]      │
│ ...                                    │
└────────────────────────────────────────┘
优势: 动态加入新请求，提高 GPU 利用率
```

### 1.3 序列状态流转

```
         ┌──────────────┐
         │   WAITING    │  等待调度
         └──────┬───────┘
                │ scheduler.schedule()
                ▼
         ┌──────────────┐
         │   RUNNING    │  正在执行
         └──────┬───────┘
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
┌──────────────┐ ┌──────────────┐
│  FINISHED    │ │  (抢占)       │
│  STOPPED     │ │  回到 WAITING │
│  FINISHED    │ │  或被驱逐     │
│  LENGTH      │ │              │
└──────────────┘ └──────────────┘
```

---

## 2. 代码阅读：调度器

### 2.1 scheduler.py 核心结构

```python
class Scheduler:
    def __init__(self, config: Config):
        # 三个队列管理序列
        self.waiting: List[Sequence] = []   # 等待队列
        self.running: List[Sequence] = []   # 运行队列
        self.finished: List[Sequence] = []  # 完成队列

        # Block 管理器
        self.block_manager = BlockManager(config)

    def schedule(self) -> SchedulerOutput:
        """核心调度方法，决定执行哪些序列"""
        pass

    def update(self, outputs: List[SamplerOutput]) -> None:
        """更新序列状态，处理输出"""
        pass
```

### 2.2 调度策略详解

**核心调度逻辑** (`schedule()` 方法):

```python
def schedule(self) -> SchedulerOutput:
    # 1. 优先处理 Decode 阶段的序列
    if self.running:
        # 检查是否有足够资源继续运行
        # 可能触发抢占（Preemption）

    # 2. 从 waiting 队列调度新序列
    while self.waiting:
        seq = self.waiting[0]

        # 检查是否可以分配 block
        if self.block_manager.can_allocate(seq):
            # 分配资源，加入 running
            self.waiting.pop(0)
            self.running.append(seq)
        else:
            # 资源不足，尝试抢占
            break

    # 3. 决定执行 prefill 还是 decode
    if has_prefill_sequences:
        return prefill_output
    else:
        return decode_output
```

### 2.3 抢占机制（Preemption）

当显存不足时，系统会抢占部分正在运行的序列：

```
抢占策略:
┌─────────────────────────────────────────────────────────┐
│ 触发条件: 新请求无法获得足够的 block                       │
│                                                         │
│ 处理方式:                                                │
│ 1. 根据 LIFO (后进先出) 选择要抢占的序列                   │
│ 2. 将序列状态改回 WAITING                                │
│ 3. 释放其占用的 block (非共享部分)                        │
│ 4. 后续重新调度该序列                                    │
│                                                         │
│ 注意: 抢占会导致重复计算，是性能的权衡                      │
└─────────────────────────────────────────────────────────┘
```

**抢占代码逻辑**:

```python
def _preempt(self, seq: Sequence):
    """抢占一个序列"""

    # 方式 1: 重新计算 (Recompute)
    # - 释放所有 block
    # - 序列回到 WAITING 状态
    # - 下次调度时重新 prefill

    # 方式 2: Swap 到 CPU (如果实现了)
    # - 将 KV Cache 换出到 CPU 内存
    # - 后续换回 GPU

    # nano-vllm 使用方式 1 (更简单)
    self.block_manager.deallocate(seq)
    self.running.remove(seq)
    self.waiting.insert(0, seq)  # 放到队首，优先重新调度
```

---

## 3. 代码阅读：序列管理

### 3.1 sequence.py 核心结构

```python
class Sequence:
    """表示一个生成序列"""

    def __init__(self, prompt: str, token_ids: List[int]):
        self.prompt = prompt           # 原始 prompt
        self.prompt_token_ids = token_ids  # prompt token IDs
        self.output_token_ids: List[int] = []  # 生成的 token IDs

        # 状态
        self.status = SequenceStatus.WAITING
        self.block_table: List[int] = []  # 分配的 block IDs

        # 前缀缓存相关
        self.prefix_indices: List[int] = []  # 命中前缀缓存的 token 位置

    @property
    def token_ids(self) -> List[int]:
        """所有 token: prompt + output"""
        return self.prompt_token_ids + self.output_token_ids

    @property
    def len(self) -> int:
        """当前序列长度"""
        return len(self.token_ids)

    def is_finished(self) -> bool:
        """检查是否生成完成"""
        return self.status == SequenceStatus.FINISHED


class SequenceStatus(Enum):
    """序列状态枚举"""
    WAITING = auto()    # 等待调度
    RUNNING = auto()    # 正在执行
    FINISHED_STOPPED = auto()   # 遇到 stop token
    FINISHED_LENGTH = auto()    # 达到最大长度
    FINISHED_IGNORED = auto()   # 被忽略（异常）
```

### 3.2 序列生命周期

```python
# 1. 创建序列
seq = Sequence(prompt="你好", token_ids=[101, 234, 567])

# 2. 调度器分配资源
block_ids = block_manager.allocate(seq)
seq.block_table = block_ids
seq.status = SequenceStatus.RUNNING

# 3. 执行 Prefill
output_token = model_runner.run_prefill(seq)
seq.output_token_ids.append(output_token)

# 4. 执行 Decode (迭代)
while not seq.is_finished():
    output_token = model_runner.run_decode(seq)
    seq.output_token_ids.append(output_token)

    # 检查终止条件
    if output_token == eos_token:
        seq.status = SequenceStatus.FINISHED_STOPPED
    if len(seq.output_token_ids) >= max_tokens:
        seq.status = SequenceStatus.FINISHED_LENGTH

# 5. 完成处理
scheduler.finished.append(seq)
scheduler.running.remove(seq)
block_manager.deallocate(seq)
```

### 3.3 SequenceGroup 概念

```python
class SequenceGroup:
    """
    一组相关的序列，用于:
    - 支持 n > 1 的并行生成
    - 支持 beam search
    """

    def __init__(self, prompt: str, sampling_params: SamplingParams):
        self.prompt = prompt
        self.sampling_params = sampling_params

        # 创建 n 个序列
        self.seqs = [
            Sequence(prompt, token_ids)
            for _ in range(sampling_params.n)
        ]
```

---

## 4. 调度流程图解

### 4.1 完整调度流程

```
┌─────────────────────────────────────────────────────────────────┐
│                       调度器工作流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LLMEngine.step()                                               │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐                                            │
│  │ Scheduler.schedule()                                         │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────┐               │
│  │ 1. 处理 RUNNING 序列 (Decode)                │               │
│  │    - 检查资源是否足够                         │               │
│  │    - 必要时抢占部分序列                       │               │
│  └────────────────┬────────────────────────────┘               │
│                   │                                             │
│                   ▼                                             │
│  ┌─────────────────────────────────────────────┐               │
│  │ 2. 从 WAITING 调度新序列 (Prefill)           │               │
│  │    - 检查 block 是否足够                      │               │
│  │    - 尽可能多地调度新请求                     │               │
│  └────────────────┬────────────────────────────┘               │
│                   │                                             │
│                   ▼                                             │
│  ┌─────────────────────────────────────────────┐               │
│  │ 3. 返回 SchedulerOutput                      │               │
│  │    - prefill_seqs: 需要预填充的序列           │               │
│  │    - decode_seqs: 需要解码的序列              │               │
│  └────────────────┬────────────────────────────┘               │
│                   │                                             │
│                   ▼                                             │
│  ┌─────────────────────────────────────────────┐               │
│  │ 4. ModelRunner 执行推理                      │               │
│  │    - prepare_prefill() 或 prepare_decode()   │               │
│  │    - 模型前向传播                            │               │
│  │    - 采样生成 token                          │               │
│  └────────────────┬────────────────────────────┘               │
│                   │                                             │
│                   ▼                                             │
│  ┌─────────────────────────────────────────────┐               │
│  │ 5. Scheduler.update()                        │               │
│  │    - 更新序列状态                            │               │
│  │    - 处理完成的序列                          │               │
│  └─────────────────────────────────────────────┘               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 批处理示意图

```
时间线: ─────────────────────────────────────────────────►

T1: ┌────────────────────────────────────┐
    │ Prefill: [Req1, Req2]              │  处理新请求
    └────────────────────────────────────┘

T2: ┌────────────────────────────────────┐
    │ Decode: [Req1, Req2]               │  生成第 1 个 token
    └────────────────────────────────────┘

T3: ┌────────────────────────────────────┐
    │ Prefill: [Req3]                    │  新请求到达
    │ Decode: [Req1, Req2]               │  同时继续 decode
    └────────────────────────────────────┘

T4: ┌────────────────────────────────────┐
    │ Decode: [Req1, Req2, Req3]         │  Req3 加入 decode
    └────────────────────────────────────┘

T5: ┌────────────────────────────────────┐
    │ Decode: [Req2, Req3]               │  Req1 完成
    │ Prefill: [Req4]                    │  新请求加入
    └────────────────────────────────────┘
```

---

## 5. 代码实践

### 5.1 调试调度器

```python
# 在 scheduler.py 中添加日志

def schedule(self) -> SchedulerOutput:
    print(f"[Scheduler] Waiting: {len(self.waiting)}, "
          f"Running: {len(self.running)}, "
          f"Finished: {len(self.finished)}")

    # ... 原有逻辑 ...

    print(f"[Scheduler] Scheduling {len(prefill_seqs)} prefill, "
          f"{len(decode_seqs)} decode sequences")

    return output
```

### 5.2 观察抢占行为

```python
# 添加抢占日志

def _preempt(self, seq: Sequence):
    print(f"[Preemption] Evicting sequence {seq.seq_id}")
    print(f"  - Current length: {seq.len}")
    print(f"  - Output so far: {seq.output_text}")

    # ... 原有逻辑 ...
```

### 5.3 实践练习

**练习 1**: 添加调试信息运行推理

```python
from nano_vllm import LLM, SamplingParams

llm = LLM(model="./models/Qwen3-0.6B")

# 发送多个请求观察调度
prompts = ["你好"] * 5
outputs = llm.generate(prompts, SamplingParams(max_tokens=20))

# 观察日志输出，理解调度过程
```

**练习 2**: 触发抢占

```python
# 减小资源，更容易触发抢占
llm = LLM(
    model="./models/Qwen3-0.6B",
    gpu_memory_utilization=0.3,  # 较低的显存
    max_num_batched_tokens=1024,  # 较小的 batch
)

# 发送大量请求
prompts = ["请写一个很长的故事"] * 20
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
```

---

## 6. 关键问题解答

### Q1: 为什么 Decode 阶段可以 batch 多个序列？

**答案**:
```
Decode 阶段的每个序列:
- 只需要处理 1 个新 token
- 每个序列的计算量相同（处理 1 个 token）
- 可以并行处理多个序列的单 token

相比之下，Prefill 阶段:
- 每个序列的 prompt 长度不同
- 需要变长序列处理（FlashAttention varlen）
- 通常单独或分组处理
```

### Q2: 什么时候会触发抢占？

**答案**:
```
触发条件:
1. 新请求到达，需要分配 block
2. 当前空闲 block 不足
3. 无法通过等待释放获得足够资源

抢占选择:
- 选择最后加入 RUNNING 的序列（LIFO）
- 因为它们计算进度最短，重跑成本最低

避免抢占的配置:
- 增大 gpu_memory_utilization
- 减小 max_num_batched_tokens
- 限制并发请求数
```

### Q3: 调度器如何决定先运行 prefill 还是 decode？

**答案**:
```python
# 调度优先级
# 1. 优先保证 RUNNING 序列继续执行 (Decode)
# 2. 在资源允许时调度新序列 (Prefill)

# 特殊情况:
# - 如果 running 队列为空，执行 prefill
# - 如果资源不足，可能只执行 decode
# - 可以同时执行 prefill 和 decode（不同 batch）
```

---

## 7. 性能考量

### 7.1 调度对性能的影响

```
┌─────────────────────────────────────────────────────────────┐
│ 调度策略          │ 吞吐量  │ 延迟   │ 显存利用 │ 适用场景   │
├─────────────────────────────────────────────────────────────┤
│ FCFS (先来先服务)  │ 低      │ 低     │ 低       │ 单请求    │
│ SJF (短作业优先)   │ 高      │ 混合   │ 高       │ 批处理    │
│ Continuous Batching│ 最高    │ 中     │ 最高     │ 生产环境  │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 抢占的开销

```
抢占成本分析:
┌─────────────────────────────────────────────┐
│ 1. 已经计算的 KV Cache 被丢弃               │
│ 2. 重新调度时需要重新计算 prefill           │
│ 3. 增加了总计算量                           │
│                                             │
│ 但避免了:                                   │
│ - OOM 错误                                  │
│ - 请求完全失败                              │
│                                             │
│ 权衡: 牺牲部分吞吐量换取稳定性              │
└─────────────────────────────────────────────┘
```

---

## 8. 代码走查清单

阅读 `engine/scheduler.py` 时，关注以下关键点：

- [ ] `schedule()` 方法的整体流程
- [ ] `waiting`、`running`、`finished` 队列的管理
- [ ] `_preempt()` 抢占逻辑
- [ ] `update()` 如何更新序列状态
- [ ] `block_manager` 的调用时机

阅读 `engine/sequence.py` 时，关注：

- [ ] `Sequence` 类的属性
- [ ] `SequenceStatus` 枚举值
- [ ] 序列状态转换的条件
- [ ] `block_table` 的作用

---

## 9. 知识检查点

### 基础问题

1. ❓ Prefill 和 Decode 阶段分别处理什么输入？

2. ❓ Continuous Batching 相比传统批处理有什么优势？

3. ❓ 序列有哪几种状态？状态之间如何转换？

### 进阶问题

4. ❓ 调度器如何决定抢占哪个序列？为什么这样选择？

5. ❓ 为什么 prefill 和 decode 可以同时进行？

6. ❓ 如果不实现抢占机制，会发生什么问题？

---

## 10. 扩展阅读

### 推荐论文

1. **Orca: A Distributed Serving System for Transformer-Based Generative Models**
   - 提出 iteration-level scheduling
   - Continuous Batching 的理论基础

2. **vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention**
   - PagedAttention 和调度策略的结合

### 相关代码

| 文件 | 关键函数 |
|------|----------|
| `engine/scheduler.py` | `schedule()`, `_preempt()`, `update()` |
| `engine/sequence.py` | `Sequence` 类 |
| `engine/llm_engine.py` | `step()` - 调用调度器 |

---

## 11. 下一步

完成 Day 2 后，你应该能够：
- ✅ 解释 prefill 和 decode 的区别
- ✅ 理解 Continuous Batching 的工作原理
- ✅ 描述序列状态流转过程
- ✅ 理解抢占机制的触发条件

**准备 Day 3**: KV Cache 块管理
- 深入理解 Block 的分配和释放
- 学习 Prefix Caching 的实现

---

*预计学习时间: 2-3 小时*
