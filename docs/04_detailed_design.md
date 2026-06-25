# Nano-vLLM 详细设计文档

## 1. 文档信息

| 项目 | 内容 |
|------|------|
| 版本 | v1.0 |
| 作者 | Nano-vLLM Team |
| 日期 | 2025-04-17 |

---

## 2. 目录结构

```
nano-vllm/
├── nanovllm/                      # 核心包
│   ├── __init__.py               # 包入口，导出 LLM, SamplingParams
│   ├── llm.py                    # LLM 入口类（继承 LLMEngine）
│   ├── config.py                 # 配置类
│   ├── sampling_params.py        # 采样参数类
│   │
│   ├── engine/                   # 推理引擎
│   │   ├── llm_engine.py         # 引擎核心
│   │   ├── scheduler.py          # 调度器
│   │   ├── model_runner.py       # 模型运行器
│   │   ├── sequence.py           # 序列管理
│   │   └── block_manager.py      # KV Cache 块管理
│   │
│   ├── models/                   # 模型实现
│   │   └── qwen3.py              # Qwen3 模型
│   │
│   ├── layers/                   # 网络层
│   │   ├── attention.py          # 注意力层
│   │   ├── linear.py             # 线性层（张量并行）
│   │   ├── layernorm.py          # RMSNorm
│   │   ├── rotary_embedding.py   # RoPE 位置编码
│   │   ├── sampler.py            # 采样器
│   │   ├── activation.py         # 激活函数
│   │   └── embed_head.py         # Embedding 和 LM Head
│   │
│   └── utils/                    # 工具函数
│       ├── context.py            # 上下文管理
│       └── loader.py             # 模型加载器
│
├── example.py                    # 使用示例
├── bench.py                      # 性能测试
├── pyproject.toml               # 项目配置
└── docs/                         # 文档目录
```

---

## 3. 配置模块

### 3.1 config.py

```python
import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    """引擎配置类"""
    model: str                           # 模型路径
    max_num_batched_tokens: int = 16384  # 最大批处理 tokens
    max_num_seqs: int = 512              # 最大批处理序列数
    max_model_len: int = 4096            # 最大序列长度
    gpu_memory_utilization: float = 0.9  # GPU 内存利用率
    tensor_parallel_size: int = 1        # 张量并行数
    enforce_eager: bool = False          # 强制 eager 模式
    hf_config: AutoConfig | None = None  # HuggingFace 配置
    eos: int = -1                        # EOS token ID
    kvcache_block_size: int = 256        # KV Cache 块大小
    num_kvcache_blocks: int = -1         # KV Cache 块数（运行时计算）

    def __post_init__(self):
        """初始化后验证"""
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, 
                                  self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
```

### 3.2 sampling_params.py

```python
from dataclasses import dataclass


@dataclass
class SamplingParams:
    """采样参数"""
    temperature: float = 1.0      # 采样温度
    max_tokens: int = 64          # 最大生成长度
    ignore_eos: bool = False      # 忽略 EOS

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
```

---

## 4. 引擎模块

### 4.1 sequence.py

```python
from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """序列状态枚举"""
    WAITING = auto()    # 等待调度
    RUNNING = auto()    # 正在执行
    FINISHED = auto()   # 执行完成


class Sequence:
    """
    序列类：表示单个推理序列
    
    属性:
        seq_id: 序列唯一标识
        status: 序列状态
        token_ids: 所有 token IDs（输入 + 输出）
        num_prompt_tokens: 输入 token 数
        num_cached_tokens: 已缓存的 token 数
        block_table: 块表，记录使用的块 ID
    """
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """获取第 i 块的 tokens"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        """追加生成的 token"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """序列化状态（用于进程间传递）"""
        return (self.num_tokens, self.num_prompt_tokens, 
                self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 
                               else self.last_token)

    def __setstate__(self, state):
        """反序列化状态"""
        (self.num_tokens, self.num_prompt_tokens, 
         self.num_cached_tokens, self.block_table) = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
```

### 4.2 block_manager.py

```python
from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    KV Cache 块
    
    属性:
        block_id: 块 ID
        ref_count: 引用计数（共享块时使用）
        hash: 块内容的哈希值
        token_ids: 块中的 token IDs
    """
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """更新块的哈希和内容"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块状态"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    块管理器：管理 KV Cache 块的分配和释放
    
    核心功能:
        1. 块分配和释放
        2. Prefix Caching（通过哈希匹配）
        3. 引用计数管理
    """
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """计算块的哈希值（用于 Prefix Caching）"""
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """分配一个块"""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """释放一个块"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否有足够的空闲块分配给序列"""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为序列分配块（支持 Prefix Caching）
        
        流程:
            1. 计算每个块的哈希值
            2. 检查是否存在哈希匹配的缓存块
            3. 命中时复用块，未命中时分配新块
        """
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # 只有满块才计算哈希
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            
            # 检查缓存命中
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            
            if cache_miss:
                # 缓存未命中，分配新块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # 块正在被使用，增加引用计数
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 块空闲，分配它
                    block = self._allocate_block(block_id)
            
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """释放序列的块"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """检查是否可以追加一个 token（可能需要新块）"""
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        处理 decode 阶段的块更新
        
        三种情况:
            1. 新块开始：分配新块
            2. 块填满：更新哈希
            3. 块未满：无需操作
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        
        if len(seq) % self.block_size == 1:
            # 新块开始
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # 块填满，更新哈希
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 块未满
            assert last_block.hash == -1
```

### 4.3 scheduler.py

```python
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    调度器：实现 Continuous Batching
    
    调度策略:
        1. Prefill 优先于 Decode
        2. 遵循 max_num_seqs 和 max_num_batched_tokens 限制
        3. 资源不足时触发抢占
    """
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, 
                                           config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """检查是否所有序列都已完成"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """添加序列到等待队列"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        调度序列
        
        Returns:
            (scheduled_seqs, is_prefill): 调度的序列列表和是否为 prefill 阶段
        """
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        # Prefill 阶段
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # 检查资源限制
            if (num_batched_tokens + len(seq) > self.max_num_batched_tokens or 
                not self.block_manager.can_allocate(seq)):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        
        if scheduled_seqs:
            return scheduled_seqs, True
        
        # Decode 阶段
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # 检查是否可以追加
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """抢占序列：释放资源，放回等待队列"""
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        """后处理：追加生成的 token，检查完成条件"""
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            # 检查是否完成
            if ((not seq.ignore_eos and token_id == self.eos) or 
                seq.num_completion_tokens == seq.max_tokens):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
```

### 4.4 model_runner.py

```python
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """
    模型运行器：执行模型推理
    
    核心功能:
        1. 模型初始化和加载
        2. KV Cache 分配
        3. Prefill/Decode 数据准备
        4. CUDA Graph 管理
        5. 张量并行通信
    """
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 初始化分布式
        dist.init_process_group("nccl", "tcp://localhost:2333", 
                                world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        
        # 设置默认 dtype 和 device
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        # 加载模型
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        
        # Warmup 和 KV Cache 分配
        self.warmup_model()
        self.allocate_kv_cache()
        
        # 捕获 CUDA Graph
        if not self.enforce_eager:
            self.capture_cudagraph()
        
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 张量并行：设置共享内存
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()  # 子进程进入循环

    def warmup_model(self):
        """预热模型"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, 
                       self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """分配 KV Cache"""
        config = self.config
        hf_config = config.hf_config
        
        # 计算可用内存
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        # 计算块大小和数量
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", 
                           hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = (2 * hf_config.num_hidden_layers * self.block_size * 
                       num_kv_heads * head_dim * hf_config.torch_dtype.itemsize)
        config.num_kvcache_blocks = int(
            total * config.gpu_memory_utilization - used - peak + current
        ) // block_bytes
        assert config.num_kvcache_blocks > 0
        
        # 分配 KV Cache
        self.kv_cache = torch.empty(
            2, hf_config.num_hidden_layers, config.num_kvcache_blocks,
            self.block_size, num_kv_heads, head_dim
        )
        
        # 将 KV Cache 引用分配到各层
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """准备块表"""
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) 
            for seq in seqs
        ]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, 
                                     pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """准备 Prefill 阶段的数据"""
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        
        for seq in seqs:
            seqlen = len(seq)
            # 只处理未缓存的 tokens
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            if not seq.block_table:  # warmup
                continue
            
            # 计算 slot_mapping
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        
        # Prefix cache 场景需要 block_tables
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        
        # 转换为 tensor
        input_ids = torch.tensor(input_ids, dtype=torch.int64, 
                                  pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, 
                                  pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, 
                                     pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, 
                                     pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, 
                                     pin_memory=True).cuda(non_blocking=True)
        
        # 设置上下文
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, 
                    max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """准备 Decode 阶段的数据"""
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        
        # 转换为 tensor
        input_ids = torch.tensor(input_ids, dtype=torch.int64, 
                                  pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, 
                                  pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, 
                                     pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, 
                                     pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        
        # 设置上下文
        set_context(False, slot_mapping=slot_mapping, 
                    context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """准备采样参数"""
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, 
                                     pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, 
                  is_prefill: bool):
        """运行模型"""
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 使用 CUDA Graph
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            
            # 更新输入
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            
            # 回放图
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """运行一步推理"""
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
        
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """捕获 CUDA Graph"""
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        
        # 创建固定大小的缓冲区
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        
        # 支持的 batch size
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 捕获每个 batch size 的图
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], 
                        context_lens=context_lens[:bs], 
                        block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
```

---

## 5. 模型层设计

### 5.1 attention.py

```python
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """Triton 内核：高效存储 KV Cache"""
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    """存储 KV Cache 到对应位置"""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0),
        k_cache, v_cache, slot_mapping, D
    )


class Attention(nn.Module):
    """
    注意力层
    
    使用 FlashAttention 进行高效计算
    """
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale  # 1/sqrt(head_dim)
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # 存储 KV Cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if context.is_prefill:
            # Prefill: 使用 varlen FlashAttention
            if context.block_tables is not None:  # Prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables
            )
        else:
            # Decode: 使用 with_kvcache FlashAttention
            o = flash_attn_with_kvcache(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True
            )
        return o
```

### 5.2 linear.py

```python
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """线性层基类"""
    def __init__(self, input_size, output_size, bias=False, tp_dim=None):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)


class ColumnParallelLinear(LinearBase):
    """
    列并行线性层：输出维度切分
    
    用于: QKV 投影、Gate/Up 投影
    """
    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param, loaded_weight):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(LinearBase):
    """
    行并行线性层：输入维度切分，输出 all-reduce
    
    用于: O 投影、Down 投影
    """
    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param, loaded_weight):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKV 并行线性层：分别切分 Q、K、V
    """
    def __init__(self, hidden_size, head_size, total_num_heads, 
                 total_num_kv_heads=None, bias=False):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param, loaded_weight, loaded_shard_id):
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
        
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    合并列并行线性层：用于 Gate + Up 投影
    """
    def __init__(self, input_size, output_sizes, bias=False):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param, loaded_weight, loaded_shard_id):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
```

### 5.3 sampler.py

```python
import torch
from torch import nn


class Sampler(nn.Module):
    """
    采样器
    
    使用 Gumbel-Max 技巧进行高效随机采样
    使用 torch.compile 优化
    """
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        采样
        
        Args:
            logits: [batch_size, vocab_size]
            temperatures: [batch_size]
        
        Returns:
            sampled_tokens: [batch_size]
        """
        # 温度缩放
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        
        # Softmax
        probs = torch.softmax(logits, dim=-1)
        
        # Gumbel-Max 采样
        # 原理: argmax(probs / exp(noise)) 等价于按概率采样
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        
        return sample_tokens
```

---

## 6. 工具模块

### 6.1 context.py

```python
from dataclasses import dataclass
import torch


@dataclass
class Context:
    """全局上下文：存储当前推理步骤的信息"""
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


# 全局上下文实例
_CONTEXT = Context()


def get_context():
    """获取当前上下文"""
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None,
                max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None,
                context_lens=None, block_tables=None):
    """设置上下文"""
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, slot_mapping,
        context_lens, block_tables
    )


def reset_context():
    """重置上下文"""
    global _CONTEXT
    _CONTEXT = Context()
```

### 6.2 loader.py

```python
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认权重加载器"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    加载模型权重
    
    支持合并权重（packed modules）的分片加载
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 检查是否为合并模块
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 普通权重
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
```

---

## 7. 关键算法

### 7.1 Prefix Caching 算法

```
算法: Prefix Cache 命中检测

输入: sequence 的 token_ids
输出: 命中的块数

1. h = -1  // 初始哈希
2. cache_miss = False
3. FOR each block i in sequence:
   a. token_ids = sequence.block(i)
   b. IF len(token_ids) == block_size:
         h = compute_hash(token_ids, h)  // 链式哈希
      ELSE:
         h = -1  // 最后一个不满的块不缓存
   c. block_id = hash_to_block_id.get(h, -1)
   d. IF block_id == -1 OR blocks[block_id].token_ids != token_ids:
         cache_miss = True
   e. IF cache_miss:
         分配新块
      ELSE:
         复用缓存块
         num_cached_tokens += block_size
```

### 7.2 Continuous Batching 算法

```
算法: Continuous Batching 调度

输入: waiting 队列, running 队列
输出: scheduled_seqs, is_prefill

// Phase 1: Prefill（优先）
1. WHILE waiting 非空 AND num_seqs < max_num_seqs:
   a. seq = waiting[0]
   b. IF 资源不足: BREAK
   c. 分配 KV Cache 块
   d. 移动到 running 队列
   e. 加入 scheduled_seqs

2. IF scheduled_seqs 非空:
      RETURN scheduled_seqs, True

// Phase 2: Decode
3. WHILE running 非空 AND num_seqs < max_num_seqs:
   a. seq = running.popleft()
   b. WHILE 资源不足:
         IF running 非空:
            preempt(running.pop())  // 抢占
         ELSE:
            preempt(seq)
            BREAK
   c. IF 可以追加:
         加入 scheduled_seqs

4. RETURN scheduled_seqs, False
```

### 7.3 Gumbel-Max 采样算法

```
算法: Gumbel-Max 采样

输入: logits [batch_size, vocab_size], temperatures [batch_size]
输出: sampled_tokens [batch_size]

原理:
  argmax(probs / G) 其中 G ~ Exp(1)
  等价于按概率采样

步骤:
1. logits = logits / temperatures  // 温度缩放
2. probs = softmax(logits)
3. noise = exponential(1)  // 生成指数分布噪声
4. scaled = probs / noise
5. tokens = argmax(scaled, dim=-1)

优点:
  - 可以使用 argmax 实现，不需要 CDF 求逆
  - 易于实现和优化
  - 与 torch.compile 兼容
```
