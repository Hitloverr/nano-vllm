# Nano-vLLM 测试用例文档

## 1. 文档信息

| 项目 | 内容 |
|------|------|
| 版本 | v1.0 |
| 作者 | Nano-vLLM Team |
| 日期 | 2025-04-17 |

---

## 2. 测试策略概览

```
┌─────────────────────────────────────────────────────────────┐
│                        测试金字塔                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                         /\                                  │
│                        /  \        E2E 测试                 │
│                       /    \       (推理流程)                │
│                      /──────\                               │
│                     /        \     集成测试                 │
│                    /          \    (模块交互)                │
│                   /────────────\                            │
│                  /              \  单元测试                 │
│                 /                \ (函数/类)                │
│                /──────────────────\                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 单元测试

### 3.1 测试 Sequence 类

#### TC-SEQ-001: Sequence 基本创建

```python
def test_sequence_creation():
    """测试 Sequence 基本创建"""
    token_ids = [1, 2, 3, 4, 5]
    seq = Sequence(token_ids)
    
    assert seq.num_tokens == 5
    assert seq.num_prompt_tokens == 5
    assert seq.status == SequenceStatus.WAITING
    assert seq.block_table == []
    assert seq.num_cached_tokens == 0
```

#### TC-SEQ-002: Sequence 块计算

```python
def test_sequence_block_calculation():
    """测试 Sequence 块数量计算"""
    # block_size = 256
    
    # 精确一个块
    seq = Sequence([1] * 256)
    assert seq.num_blocks == 1
    assert seq.last_block_num_tokens == 256
    
    # 一个块多一点
    seq = Sequence([1] * 300)
    assert seq.num_blocks == 2
    assert seq.last_block_num_tokens == 44
    
    # 多个块
    seq = Sequence([1] * 1000)
    assert seq.num_blocks == 4
    assert seq.last_block_num_tokens == 232
```

#### TC-SEQ-003: Sequence token 追加

```python
def test_sequence_append_token():
    """测试 Sequence token 追加"""
    seq = Sequence([1, 2, 3])
    assert seq.num_tokens == 3
    assert seq.last_token == 3
    
    seq.append_token(4)
    assert seq.num_tokens == 4
    assert seq.last_token == 4
    assert seq.completion_token_ids == [4]
```

#### TC-SEQ-004: Sequence 块访问

```python
def test_sequence_block_access():
    """测试 Sequence 块访问"""
    token_ids = list(range(600))  # 600 tokens, 3 blocks
    seq = Sequence(token_ids)
    seq.block_size = 256
    
    block0 = seq.block(0)
    assert len(block0) == 256
    assert block0 == token_ids[:256]
    
    block2 = seq.block(2)
    assert len(block2) == 88  # 600 - 512 = 88
    assert block2 == token_ids[512:]
```

---

### 3.2 测试 BlockManager 类

#### TC-BM-001: 块哈希计算

```python
def test_block_hash_computation():
    """测试块哈希计算"""
    token_ids = [1, 2, 3, 4, 5]
    
    # 相同输入产生相同哈希
    h1 = BlockManager.compute_hash(token_ids)
    h2 = BlockManager.compute_hash(token_ids)
    assert h1 == h2
    
    # 不同输入产生不同哈希
    h3 = BlockManager.compute_hash([5, 4, 3, 2, 1])
    assert h1 != h3
    
    # 链式哈希
    h4 = BlockManager.compute_hash(token_ids, prefix=h1)
    h5 = BlockManager.compute_hash(token_ids, prefix=h1)
    assert h4 == h5
    assert h4 != h1
```

#### TC-BM-002: 块分配

```python
def test_block_allocation():
    """测试块分配"""
    manager = BlockManager(num_blocks=100, block_size=256)
    
    # 创建序列
    token_ids = list(range(500))  # 2 blocks
    seq = Sequence(token_ids)
    
    # 检查可分配
    assert manager.can_allocate(seq) == True
    
    # 分配
    manager.allocate(seq)
    assert len(seq.block_table) == 2
    assert len(manager.free_block_ids) == 98
    assert len(manager.used_block_ids) == 2
```

#### TC-BM-003: 块释放

```python
def test_block_deallocation():
    """测试块释放"""
    manager = BlockManager(num_blocks=100, block_size=256)
    seq = Sequence(list(range(500)))
    
    manager.allocate(seq)
    assert len(manager.free_block_ids) == 98
    
    manager.deallocate(seq)
    assert len(manager.free_block_ids) == 100
    assert seq.block_table == []
```

#### TC-BM-004: Prefix Cache 命中

```python
def test_prefix_cache_hit():
    """测试 Prefix Cache 命中"""
    manager = BlockManager(num_blocks=100, block_size=256)
    
    # 第一个序列
    token_ids1 = [1] * 256 + [2] * 256 + [3] * 100  # 2.5 blocks
    seq1 = Sequence(token_ids1)
    manager.allocate(seq1)
    
    # 第二个序列（相同前缀）
    token_ids2 = [1] * 256 + [2] * 256 + [4] * 100  # 前 2 块相同
    seq2 = Sequence(token_ids2)
    manager.allocate(seq2)
    
    # 应该复用前 2 块
    assert seq2.num_cached_tokens == 512
    assert seq1.block_table[0] == seq2.block_table[0]
    assert seq1.block_table[1] == seq2.block_table[1]
    assert seq1.block_table[2] != seq2.block_table[2]
```

#### TC-BM-005: 块追加

```python
def test_block_append():
    """测试块追加"""
    manager = BlockManager(num_blocks=100, block_size=256)
    
    # 初始序列
    seq = Sequence([1] * 256)  # 1 block
    manager.allocate(seq)
    assert len(seq.block_table) == 1
    
    # 追加 tokens 直到需要新块
    for i in range(255):
        seq.append_token(2)
    
    # 还在同一个块
    manager.may_append(seq)
    assert len(seq.block_table) == 1
    
    # 再追加一个，需要新块
    seq.append_token(2)
    manager.may_append(seq)
    assert len(seq.block_table) == 2
```

---

### 3.3 测试 Scheduler 类

#### TC-SCHED-001: 序列添加

```python
def test_scheduler_add_sequence():
    """测试序列添加"""
    config = Config(model="/path/to/model")
    scheduler = Scheduler(config)
    
    seq = Sequence([1, 2, 3])
    scheduler.add(seq)
    
    assert len(scheduler.waiting) == 1
    assert scheduler.waiting[0] == seq
```

#### TC-SCHED-002: Prefill 调度

```python
def test_scheduler_prefill_scheduling():
    """测试 Prefill 调度"""
    config = Config(model="/path/to/model", max_num_seqs=2)
    scheduler = Scheduler(config)
    
    seq1 = Sequence([1] * 100)
    seq2 = Sequence([2] * 100)
    scheduler.add(seq1)
    scheduler.add(seq2)
    
    seqs, is_prefill = scheduler.schedule()
    
    assert is_prefill == True
    assert len(seqs) == 2
    assert seq1.status == SequenceStatus.RUNNING
    assert seq2.status == SequenceStatus.RUNNING
```

#### TC-SCHED-003: Decode 调度

```python
def test_scheduler_decode_scheduling():
    """测试 Decode 调度"""
    config = Config(model="/path/to/model", max_num_seqs=2)
    scheduler = Scheduler(config)
    
    # 先 prefill
    seq = Sequence([1] * 100)
    scheduler.add(seq)
    seqs, _ = scheduler.schedule()
    
    # 模拟生成了一个 token
    scheduler.postprocess(seqs, [42])
    
    # 下一次应该是 decode
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill == False
```

#### TC-SCHED-004: 抢占机制

```python
def test_scheduler_preemption():
    """测试抢占机制"""
    config = Config(model="/path/to/model", max_num_seqs=1)
    scheduler = Scheduler(config)
    
    seq1 = Sequence([1] * 100)
    seq2 = Sequence([2] * 100)
    
    # 调度第一个序列
    scheduler.add(seq1)
    scheduler.schedule()
    
    # 添加第二个序列
    scheduler.add(seq2)
    
    # 强制资源不足场景（需要 mock can_append 返回 False）
    # 抢占应该发生
    # ...
```

#### TC-SCHED-005: 序列完成

```python
def test_scheduler_sequence_completion():
    """测试序列完成"""
    config = Config(model="/path/to/model", eos=0)
    scheduler = Scheduler(config)
    
    seq = Sequence([1, 2, 3], SamplingParams(max_tokens=2))
    scheduler.add(seq)
    scheduler.schedule()
    
    # 生成两个 tokens
    scheduler.postprocess([seq], [4])
    assert seq.status == SequenceStatus.RUNNING
    
    scheduler.postprocess([seq], [5])
    assert seq.status == SequenceStatus.FINISHED
```

---

### 3.4 测试 Attention 层

#### TC-ATT-001: Prefill 注意力

```python
def test_attention_prefill():
    """测试 Prefill 阶段注意力"""
    num_heads = 8
    head_dim = 64
    scale = 1.0 / (head_dim ** 0.5)
    
    attn = Attention(num_heads, head_dim, scale, num_heads)
    
    # 模拟 prefill 输入
    batch_size = 4
    total_tokens = 100
    
    q = torch.randn(total_tokens, num_heads, head_dim)
    k = torch.randn(total_tokens, num_heads, head_dim)
    v = torch.randn(total_tokens, num_heads, head_dim)
    
    # 设置 context
    set_context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, 25, 50, 75, 100]),
        cu_seqlens_k=torch.tensor([0, 25, 50, 75, 100]),
        max_seqlen_q=25,
        max_seqlen_k=25,
        slot_mapping=torch.zeros(total_tokens, dtype=torch.int32)
    )
    
    output = attn(q, k, v)
    assert output.shape == (total_tokens, num_heads, head_dim)
    
    reset_context()
```

#### TC-ATT-002: Decode 注意力

```python
def test_attention_decode():
    """测试 Decode 阶段注意力"""
    num_heads = 8
    head_dim = 64
    scale = 1.0 / (head_dim ** 0.5)
    
    attn = Attention(num_heads, head_dim, scale, num_heads)
    
    # 模拟 KV Cache
    num_blocks = 10
    block_size = 256
    attn.k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
    attn.v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
    
    # 模拟 decode 输入（每个序列一个 token）
    batch_size = 4
    q = torch.randn(batch_size, num_heads, head_dim)
    k = torch.randn(batch_size, num_heads, head_dim)
    v = torch.randn(batch_size, num_heads, head_dim)
    
    # 设置 context
    set_context(
        is_prefill=False,
        slot_mapping=torch.tensor([0, 256, 512, 768]),
        context_lens=torch.tensor([100, 200, 300, 400]),
        block_tables=torch.tensor([[0], [1], [2], [3]])
    )
    
    output = attn(q, k, v)
    assert output.shape == (batch_size, num_heads, head_dim)
    
    reset_context()
```

---

### 3.5 测试 Linear 层

#### TC-LIN-001: ColumnParallelLinear

```python
def test_column_parallel_linear():
    """测试列并行线性层"""
    input_size = 1024
    output_size = 4096
    batch_size = 4
    
    layer = ColumnParallelLinear(input_size, output_size)
    x = torch.randn(batch_size, input_size)
    
    y = layer(x)
    
    # 输出维度应该是 output_size / world_size
    assert y.shape == (batch_size, output_size // dist.get_world_size())
```

#### TC-LIN-002: RowParallelLinear

```python
def test_row_parallel_linear():
    """测试行并行线性层"""
    input_size = 4096
    output_size = 1024
    batch_size = 4
    
    layer = RowParallelLinear(input_size, output_size)
    
    # 输入维度应该是 input_size / world_size
    x = torch.randn(batch_size, input_size // dist.get_world_size())
    y = layer(x)
    
    assert y.shape == (batch_size, output_size)
```

#### TC-LIN-003: QKVParallelLinear 权重加载

```python
def test_qkv_parallel_linear_weight_loader():
    """测试 QKV 并行线性层权重加载"""
    hidden_size = 1024
    head_size = 64
    num_heads = 16
    num_kv_heads = 4
    
    layer = QKVParallelLinear(hidden_size, head_size, num_heads, num_kv_heads)
    
    # 模拟权重加载
    q_weight = torch.randn(num_heads * head_size, hidden_size)
    k_weight = torch.randn(num_kv_heads * head_size, hidden_size)
    v_weight = torch.randn(num_kv_heads * head_size, hidden_size)
    
    layer.weight_loader(layer.weight, q_weight, "q")
    layer.weight_loader(layer.weight, k_weight, "k")
    layer.weight_loader(layer.weight, v_weight, "v")
    
    # 验证权重形状
    expected_size = (num_heads + 2 * num_kv_heads) * head_size // dist.get_world_size()
    assert layer.weight.shape[0] == expected_size
```

---

### 3.6 测试 Sampler

#### TC-SAMP-001: 基本采样

```python
def test_sampler_basic():
    """测试基本采样"""
    sampler = Sampler()
    
    batch_size = 4
    vocab_size = 1000
    
    logits = torch.randn(batch_size, vocab_size)
    temperatures = torch.ones(batch_size)
    
    tokens = sampler(logits, temperatures)
    
    assert tokens.shape == (batch_size,)
    assert (tokens >= 0).all() and (tokens < vocab_size).all()
```

#### TC-SAMP-002: 温度效果

```python
def test_sampler_temperature_effect():
    """测试温度效果"""
    sampler = Sampler()
    
    # 极端 logits（一个 token 概率很高）
    logits = torch.tensor([[0.0, 0.0, 10.0, 0.0]])  # token 2 概率最高
    
    # 低温度：应该几乎总是选 token 2
    low_temp = torch.tensor([0.1])
    tokens_low = [sampler(logits, low_temp).item() for _ in range(100)]
    assert tokens_low.count(2) > 95
    
    # 高温度：分布更均匀
    high_temp = torch.tensor([10.0])
    tokens_high = [sampler(logits, high_temp).item() for _ in range(100)]
    # 应该有更多多样性
    assert len(set(tokens_high)) > 1
```

---

## 4. 集成测试

### 4.1 端到端推理测试

#### TC-E2E-001: 单序列推理

```python
def test_single_sequence_inference():
    """测试单序列推理"""
    model_path = "/path/to/Qwen3-0.6B"
    llm = LLM(model_path, enforce_eager=True)
    
    prompt = "Hello, world!"
    sampling_params = SamplingParams(temperature=0.8, max_tokens=50)
    
    outputs = llm.generate([prompt], sampling_params)
    
    assert len(outputs) == 1
    assert "text" in outputs[0]
    assert "token_ids" in outputs[0]
    assert len(outputs[0]["token_ids"]) > 0
```

#### TC-E2E-002: 批量推理

```python
def test_batch_inference():
    """测试批量推理"""
    model_path = "/path/to/Qwen3-0.6B"
    llm = LLM(model_path, enforce_eager=True)
    
    prompts = [
        "What is AI?",
        "Explain quantum computing.",
        "Write a haiku.",
    ]
    sampling_params = SamplingParams(temperature=0.8, max_tokens=50)
    
    outputs = llm.generate(prompts, sampling_params)
    
    assert len(outputs) == len(prompts)
    for output in outputs:
        assert "text" in output
        assert len(output["text"]) > 0
```

#### TC-E2E-003: 变长输入

```python
def test_variable_length_input():
    """测试变长输入"""
    model_path = "/path/to/Qwen3-0.6B"
    llm = LLM(model_path, enforce_eager=True)
    
    # 变长输入
    prompts = [
        "Hi",                           # 短
        "Hello, how are you?",          # 中
        "This is a very long prompt" * 10,  # 长
    ]
    sampling_params = SamplingParams(temperature=0.8, max_tokens=20)
    
    outputs = llm.generate(prompts, sampling_params)
    
    assert len(outputs) == len(prompts)
```

---

### 4.2 Prefix Caching 测试

#### TC-PREFIX-001: 相同前缀复用

```python
def test_prefix_caching_reuse():
    """测试相同前缀复用"""
    model_path = "/path/to/Qwen3-0.6B"
    llm = LLM(model_path, enforce_eager=True)
    
    # 相同前缀
    prefix = "The capital of France is"
    prompts = [
        prefix + "?",
        prefix + " Paris.",
        prefix + " known for its culture.",
    ]
    
    # 第一次运行
    outputs1 = llm.generate(prompts, SamplingParams(max_tokens=10))
    
    # 第二次运行（应该使用缓存）
    outputs2 = llm.generate(prompts, SamplingParams(max_tokens=10))
    
    # 验证结果正确
    assert len(outputs1) == len(outputs2) == 3
```

---

### 4.3 张量并行测试

#### TC-TP-001: 单卡 vs 多卡

```python
def test_tensor_parallel_consistency():
    """测试张量并行一致性"""
    model_path = "/path/to/Qwen3-0.6B"
    
    # 单卡
    llm1 = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    
    # 多卡（假设有 2 个 GPU）
    llm2 = LLM(model_path, enforce_eager=True, tensor_parallel_size=2)
    
    prompt = "Hello"
    sampling_params = SamplingParams(temperature=0.1, max_tokens=20)
    
    # 结果应该相似（允许浮点误差）
    output1 = llm1.generate([prompt], sampling_params)[0]
    output2 = llm2.generate([prompt], sampling_params)[0]
    
    # 至少开头应该相似
    # 注意：由于浮点误差和采样随机性，完全一致不太可能
```

---

## 5. 性能测试

### 5.1 吞吐量测试

```python
def test_throughput():
    """测试吞吐量"""
    model_path = "/path/to/Qwen3-0.6B"
    llm = LLM(model_path, enforce_eager=False)
    
    num_requests = 256
    max_input_len = 1024
    max_output_len = 1024
    
    # 生成随机输入
    prompts = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_requests)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, max_tokens=randint(100, max_output_len))
        for _ in range(num_requests)
    ]
    
    # 计时
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.time() - start_time
    
    # 计算吞吐量
    total_tokens = sum(len(o["token_ids"]) for o in outputs)
    throughput = total_tokens / elapsed
    
    print(f"Throughput: {throughput:.2f} tokens/s")
    assert throughput > 1000  # 至少 1000 tokens/s
```

### 5.2 内存测试

```python
def test_memory_usage():
    """测试内存使用"""
    model_path = "/path/to/Qwen3-0.6B"
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    llm = LLM(model_path, gpu_memory_utilization=0.9)
    
    # 运行推理
    prompts = ["Test"] * 100
    llm.generate(prompts, SamplingParams(max_tokens=100))
    
    # 检查内存
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved: {reserved:.2f} GB")
```

---

## 6. 边界条件测试

### 6.1 空输入测试

```python
def test_empty_input():
    """测试空输入"""
    llm = LLM("/path/to/model", enforce_eager=True)
    
    # 空字符串
    outputs = llm.generate([""], SamplingParams(max_tokens=10))
    assert len(outputs) == 1
```

### 6.2 超长输入测试

```python
def test_long_input():
    """测试超长输入"""
    llm = LLM("/path/to/model", max_model_len=4096, enforce_eager=True)
    
    # 超过最大长度（应该被截断或报错）
    long_prompt = "Hello " * 2000  # ~10000 tokens
    
    # 应该优雅处理
    try:
        outputs = llm.generate([long_prompt], SamplingParams(max_tokens=10))
    except Exception as e:
        # 或者抛出清晰的错误
        assert "length" in str(e).lower()
```

### 6.3 极端温度测试

```python
def test_extreme_temperature():
    """测试极端温度"""
    llm = LLM("/path/to/model", enforce_eager=True)
    
    # 非常低的温度（接近 greedy）
    outputs = llm.generate(
        ["Hello"],
        SamplingParams(temperature=0.001, max_tokens=10)
    )
    assert len(outputs) == 1
    
    # 非常高的温度
    outputs = llm.generate(
        ["Hello"],
        SamplingParams(temperature=100.0, max_tokens=10)
    )
    assert len(outputs) == 1
```

---

## 7. 测试环境要求

### 7.1 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | 8GB VRAM | 24GB+ VRAM |
| CPU | 4 核 | 8+ 核 |
| 内存 | 16 GB | 32+ GB |

### 7.2 软件要求

| 软件 | 版本 |
|------|------|
| Python | 3.10 - 3.12 |
| PyTorch | 2.4+ |
| CUDA | 11.8+ |
| pytest | 7.0+ |

---

## 8. 测试覆盖率目标

| 模块 | 目标覆盖率 |
|------|-----------|
| engine/scheduler.py | > 90% |
| engine/block_manager.py | > 90% |
| engine/model_runner.py | > 85% |
| layers/attention.py | > 80% |
| layers/linear.py | > 85% |
| layers/sampler.py | > 90% |
| **总体** | **> 85%** |

---

## 9. 持续集成

### 9.1 CI 配置

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov
      
      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=nanovllm
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## 10. 测试执行命令

```bash
# 运行所有测试
pytest tests/

# 运行单元测试
pytest tests/unit/

# 运行集成测试
pytest tests/integration/

# 运行特定测试文件
pytest tests/unit/test_scheduler.py

# 运行特定测试用例
pytest tests/unit/test_scheduler.py::test_scheduler_prefill_scheduling

# 生成覆盖率报告
pytest tests/ --cov=nanovllm --cov-report=html
```
