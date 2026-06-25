# Nano-vLLM 需求规格说明书

## 1. 文档信息

| 项目 | 内容 |
|------|------|
| 版本 | v1.0 |
| 作者 | Nano-vLLM Team |
| 日期 | 2025-04-17 |

---

## 2. 引言

### 2.1 目的

本文档详细描述 Nano-vLLM 系统的功能需求和非功能需求，作为开发、测试和验收的依据。

### 2.2 适用范围

本文档适用于 Nano-vLLM v0.2.0 版本的学习和实现。

---

## 3. 功能需求

### 3.1 推理引擎核心

#### FR-ENG-001: 离线批量推理

**需求描述**：系统应支持离线批量推理，能够处理多个输入提示并生成输出。

**输入**：
- 提示列表（字符串或 token ID 列表）
- 采样参数

**输出**：
- 生成的文本和 token ID 列表

**验收标准**：
- [ ] 支持字符串和 token ID 两种输入格式
- [ ] 支持批量处理多个提示
- [ ] 返回完整的生成结果

**接口定义**：

```python
def generate(
    self,
    prompts: list[str] | list[list[int]],
    sampling_params: SamplingParams | list[SamplingParams],
    use_tqdm: bool = True,
) -> list[dict]:
    """生成文本"""
    pass
```

---

#### FR-ENG-002: 序列状态管理

**需求描述**：系统应正确管理每个序列的状态，包括等待、运行、完成三种状态。

**状态定义**：

```python
class SequenceStatus(Enum):
    WAITING = auto()    # 等待调度
    RUNNING = auto()    # 正在执行
    FINISHED = auto()   # 执行完成
```

**验收标准**：
- [ ] 新序列初始状态为 WAITING
- [ ] 序列被调度后状态变为 RUNNING
- [ ] 序列生成完成后状态变为 FINISHED
- [ ] 状态转换正确，无遗漏

---

#### FR-ENG-003: 连续批处理调度

**需求描述**：系统应实现 Continuous Batching 调度策略，动态管理批次中的序列。

**调度规则**：

```
┌─────────────────────────────────────────────────────────────┐
│                    调度优先级规则                            │
├─────────────────────────────────────────────────────────────┤
│  1. Prefill 阶段优先于 Decode 阶段                           │
│  2. 遵循 max_num_seqs 和 max_num_batched_tokens 限制        │
│  3. 内存不足时触发抢占（Preemption）                          │
└─────────────────────────────────────────────────────────────┘
```

**验收标准**：
- [ ] Prefill 阶段正确调度等待队列中的序列
- [ ] Decode 阶段正确处理运行队列中的序列
- [ ] 资源不足时正确触发抢占机制
- [ ] 支持 iteration-level 调度

---

### 3.2 KV Cache 管理

#### FR-KV-001: KV Cache 块分配

**需求描述**：系统应使用分块方式管理 KV Cache，支持动态分配和释放。

**块参数**：
- 默认块大小：256 tokens
- 可配置块大小（256 的倍数）

**验收标准**：
- [ ] 支持按块分配 KV Cache
- [ ] 支持块的引用计数管理
- [ ] 支持块的动态释放
- [ ] 内存不足时正确拒绝分配

---

#### FR-KV-002: Prefix Caching

**需求描述**：系统应支持前缀缓存，避免重复计算相同前缀的 KV Cache。

**实现机制**：

```
┌─────────────────────────────────────────────────────────────┐
│                    Prefix Caching 流程                       │
├─────────────────────────────────────────────────────────────┤
│  1. 计算每个块的哈希值（基于 token IDs 和前一块哈希）          │
│  2. 查找哈希值匹配的已缓存块                                  │
│  3. 命中时复用 KV Cache，跳过计算                            │
│  4. 未命中时分配新块并计算 KV Cache                          │
└─────────────────────────────────────────────────────────────┘
```

**验收标准**：
- [ ] 正确计算块的哈希值
- [ ] 正确识别可复用的缓存块
- [ ] 缓存命中时跳过对应 token 的计算
- [ ] 缓存未命中时正确分配新块

---

#### FR-KV-003: Slot Mapping

**需求描述**：系统应维护 token 到 KV Cache 槽位的映射关系。

**映射结构**：
- 每个 token 映射到一个唯一的 slot
- Slot = block_id * block_size + offset

**验收标准**：
- [ ] Prefill 阶段正确生成 slot mapping
- [ ] Decode 阶段正确更新 slot mapping
- [ ] 支持缓存命中场景的 slot mapping

---

### 3.3 模型执行

#### FR-MOD-001: Prefill 执行

**需求描述**：系统应正确执行 Prefill 阶段，处理输入 prompt。

**输入准备**：
- input_ids: 未缓存的 token IDs
- positions: 对应的位置索引
- cu_seqlens_q/k: 累积序列长度
- slot_mapping: 槽位映射

**验收标准**：
- [ ] 正确处理多个变长序列
- [ ] 正确跳过已缓存的 tokens
- [ ] 正确调用 FlashAttention varlen API
- [ ] 支持 prefix cache 场景

---

#### FR-MOD-002: Decode 执行

**需求描述**：系统应正确执行 Decode 阶段，逐个生成输出 token。

**输入准备**：
- input_ids: 最后一个 token
- positions: 当前位置
- context_lens: 上下文长度
- block_tables: 块表

**验收标准**：
- [ ] 每次生成一个 token
- [ ] 正确更新 KV Cache
- [ ] 正确调用 FlashAttention with_kvcache API
- [ ] 支持 CUDA Graph 优化

---

#### FR-MOD-003: CUDA Graph 支持

**需求描述**：系统应支持 CUDA Graph 优化，减少 Decode 阶段的 kernel 启动开销。

**实现要求**：
- 预捕获多个 batch size 的图
- 动态选择合适的图执行
- 支持图变量更新

**验收标准**：
- [ ] 支持 eager 模式（enforce_eager=True）
- [ ] 正确捕获 CUDA Graph
- [ ] 正确选择和回放图
- [ ] 大 batch（>512）时使用 eager 模式

---

### 3.4 张量并行

#### FR-TP-001: 张量并行初始化

**需求描述**：系统应支持张量并行，将模型切分到多个 GPU。

**配置参数**：
- tensor_parallel_size: 并行数（1-8）

**验收标准**：
- [ ] 支持 1-8 GPU 并行
- [ ] 正确初始化 NCCL 进程组
- [ ] 正确设置每个 rank 的设备

---

#### FR-TP-002: 并行线性层

**需求描述**：系统应实现张量并行线性层。

**层类型**：
- ColumnParallelLinear: 输出维度切分
- RowParallelLinear: 输入维度切分
- QKVParallelLinear: Q/K/V 分别切分
- MergedColumnParallelLinear: 合并切分

**验收标准**：
- [ ] Column 正确切分输出维度
- [ ] Row 正确切分输入维度并 all-reduce
- [ ] QKV 正确加载权重
- [ ] 支持 weight_loader 机制

---

#### FR-TP-003: 进程间通信

**需求描述**：系统应支持张量并行进程间的通信。

**通信机制**：
- 主进程（rank 0）协调执行
- 使用共享内存传递参数
- 使用 Event 同步

**验收标准**：
- [ ] 主进程正确广播命令
- [ ] 子进程正确响应命令
- [ ] 进程组正确销毁

---

### 3.5 模型支持

#### FR-MDL-001: Qwen3 模型支持

**需求描述**：系统应支持 Qwen3 系列模型的推理。

**模型组件**：
- VocabParallelEmbedding
- Qwen3DecoderLayer
- RMSNorm
- RoPE
- FlashAttention
- ParallelLMHead

**验收标准**：
- [ ] 正确加载 Qwen3 模型权重
- [ ] 正确实现模型结构
- [ ] 支持 qk_norm
- [ ] 支持 weight tying

---

### 3.6 采样

#### FR-SMP-001: 温度采样

**需求描述**：系统应支持温度缩放采样。

**采样算法**：
- 温度缩放
- Gumbel-Max 采样（高效随机采样）

**验收标准**：
- [ ] 支持温度参数设置
- [ ] 正确实现 Gumbel-Max 采样
- [ ] 使用 torch.compile 优化

---

#### FR-SMP-002: 停止条件

**需求描述**：系统应支持多种停止条件。

**停止条件**：
- 遇到 EOS token
- 达到最大生成长度
- ignore_eos 模式

**验收标准**：
- [ ] 正确检测 EOS
- [ ] 正确限制最大长度
- [ ] 支持 ignore_eos 模式

---

## 4. 非功能需求

### 4.1 性能需求

| 编号 | 需求描述 | 指标 |
|------|---------|------|
| NFR-PERF-001 | 推理吞吐量 | ≥ vLLM 的 95% |
| NFR-PERF-002 | Prefix Cache 命中加速 | ≥ 2x |
| NFR-PERF-003 | CUDA Graph 加速 | ≥ 1.2x |
| NFR-PERF-004 | 最大批处理序列数 | 512 |
| NFR-PERF-005 | 最大批处理 tokens | 16384 |

### 4.2 可扩展性需求

| 编号 | 需求描述 | 指标 |
|------|---------|------|
| NFR-SCALE-001 | 最大 GPU 数 | 8 |
| NFR-SCALE-002 | 最大序列长度 | 取决于 GPU 内存 |
| NFR-SCALE-003 | 最大模型支持 | 取决于 GPU 内存 |

### 4.3 可用性需求

| 编号 | 需求描述 |
|------|---------|
| NFR-USE-001 | API 与 vLLM 兼容 |
| NFR-USE-002 | 代码可读性高，适合学习 |
| NFR-USE-003 | 提供完整示例代码 |

### 4.4 兼容性需求

| 编号 | 需求描述 |
|------|---------|
| NFR-COMP-001 | 支持 Python 3.10-3.12 |
| NFR-COMP-002 | 支持 PyTorch 2.4+ |
| NFR-COMP-003 | 支持 NVIDIA GPU |
| NFR-COMP-004 | 支持 Transformers 4.51+ |

---

## 5. 配置参数

### 5.1 引擎配置

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| model | str | 必填 | 模型路径 |
| max_num_batched_tokens | int | 16384 | 最大批处理 tokens |
| max_num_seqs | int | 512 | 最大批处理序列数 |
| max_model_len | int | 4096 | 最大序列长度 |
| gpu_memory_utilization | float | 0.9 | GPU 内存利用率 |
| tensor_parallel_size | int | 1 | 张量并行数 |
| enforce_eager | bool | False | 强制 eager 模式 |
| kvcache_block_size | int | 256 | KV Cache 块大小 |

### 5.2 采样参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| temperature | float | 1.0 | 采样温度 |
| max_tokens | int | 64 | 最大生成 tokens |
| ignore_eos | bool | False | 忽略 EOS |

---

## 6. 接口定义

### 6.1 LLM 类

```python
class LLM(LLMEngine):
    """推理引擎入口类"""
    pass
```

### 6.2 SamplingParams 类

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
```

### 6.3 生成接口

```python
def generate(
    prompts: list[str] | list[list[int]],
    sampling_params: SamplingParams | list[SamplingParams],
    use_tqdm: bool = True,
) -> list[dict]:
    """
    Args:
        prompts: 输入提示列表
        sampling_params: 采样参数
        use_tqdm: 是否显示进度条
    
    Returns:
        list[dict]: 每个元素包含 "text" 和 "token_ids"
    """
```

---

## 7. 数据结构

### 7.1 Sequence

```python
class Sequence:
    seq_id: int                    # 序列 ID
    status: SequenceStatus         # 状态
    token_ids: list[int]           # 所有 tokens
    num_prompt_tokens: int         # 输入 tokens 数
    num_cached_tokens: int         # 已缓存 tokens 数
    block_table: list[int]         # 块表
    temperature: float             # 采样温度
    max_tokens: int                # 最大生成长度
```

### 7.2 Block

```python
class Block:
    block_id: int                  # 块 ID
    ref_count: int                 # 引用计数
    hash: int                      # 哈希值
    token_ids: list[int]           # token IDs
```

### 7.3 Context

```python
@dataclass
class Context:
    is_prefill: bool               # 是否 prefill
    cu_seqlens_q: Tensor           # Q 累积序列长度
    cu_seqlens_k: Tensor           # K 累积序列长度
    max_seqlen_q: int              # Q 最大序列长度
    max_seqlen_k: int              # K 最大序列长度
    slot_mapping: Tensor           # 槽位映射
    context_lens: Tensor           # 上下文长度
    block_tables: Tensor           # 块表
```

---

## 8. 验收标准

### 8.1 功能验收

- [ ] 所有 FR-* 需求项通过测试
- [ ] example.py 运行成功
- [ ] bench.py 性能达标

### 8.2 性能验收

- [ ] 吞吐量达到 vLLM 的 95% 以上
- [ ] Prefix Cache 正确工作
- [ ] CUDA Graph 正确工作
- [ ] 张量并行正确工作

### 8.3 代码质量验收

- [ ] 代码可读性高
- [ ] 注释清晰
- [ ] 结构清晰
