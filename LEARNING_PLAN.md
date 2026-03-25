# nano-vllm 一周学习计划

> 通过本计划，你将完全掌握 nano-vllm 项目的核心原理和实现细节。

## 项目概述

nano-vllm 是一个轻量级的 vLLM 实现，约 1200 行 Python 代码，实现了与 vLLM 相当的推理速度。

### 核心特性
- 🚀 **Prefix Caching** - 前缀缓存，避免重复计算
- ⚡ **Tensor Parallelism** - 张量并行，支持多卡推理
- 🔥 **Torch Compile** - PyTorch 编译优化
- 📊 **CUDA Graph** - 减少 kernel 启动开销
- ⚡ **FlashAttention** - 高效注意力计算

---

## 学习计划

### Day 1: 环境搭建与整体架构

**目标**: 完成环境配置，理解项目整体架构

**学习内容**:
1. 理解 LLM 推理的基本流程
2. 安装依赖：`pip install -e .`（注意 CUDA 环境）
3. 下载模型：`huggingface-cli download Qwen/Qwen3-0.6B`
4. 运行示例：`python example.py` 验证环境

**代码阅读**:
- `llm.py` - 入口类，了解 API 设计
- `config.py` - 配置参数，理解各参数含义
- `sampling_params.py` - 采样参数

**重点理解**:
- `LLM.generate()` 的工作流程
- 配置参数的作用（max_num_batched_tokens, gpu_memory_utilization 等）

---

### Day 2: 调度器与序列管理

**目标**: 理解请求调度和序列状态管理

**学习内容**:
1. Prefill（预填充）阶段 vs Decode（解码）阶段的区别
2. Continuous Batching（连续批处理）策略

**代码阅读**:
- `engine/scheduler.py` - 调度器核心
- `engine/sequence.py` - 序列状态管理

**重点理解**:
- `Scheduler.schedule()` 如何决定先运行 prefill 还是 decode
- 序列状态流转（WAITING → RUNNING → FINISHED）
- 抢占机制（preemption）如何处理

**思考题**:
- 为什么 decode 阶段可以 batch 多个序列？
- 什么时候会触发抢占？

---

### Day 3: KV Cache 块管理

**目标**: 理解 Prefix Cache 和 Block 分配机制

**学习内容**:
1. KV Cache 的作用
2. Block 管理的原理
3. 前缀缓存如何实现

**代码阅读**:
- `engine/block_manager.py` - 块管理器

**重点理解**:
- Block 的分配和释放
- Hash 计算用于前缀匹配
- `can_allocate`, `allocate`, `deallocate` 的逻辑

**核心概念**:
```
Block Manager 维护:
- free_block_ids: 空闲块队列
- hash_to_block_id: 哈希到块ID的映射
- ref_count: 引用计数（共享块时使用）
```

---

### Day 4: 模型运行器

**目标**: 理解模型推理的执行流程

**学习内容**:
1. Prefill 和 Decode 的数据准备
2. CUDA Graph 的捕获和回放
3. 张量并行的通信机制

**代码阅读**:
- `engine/model_runner.py` - 模型运行器

**重点理解**:
- `prepare_prefill()` vs `prepare_decode()` 的区别
- `capture_cudagraph()` 如何优化 decode 阶段
- 主进程如何与 Tensor Parallel 进程通信

**关键函数**:
- `run_model()` - 根据是否 prefill 选择执行路径
- CUDA Graph 动态 batch size 选择逻辑

---

### Day 5: 模型实现与层定义

**目标**: 理解模型结构和各层的实现

**学习内容**:
1. Qwen3 模型架构
2. 各层的实现细节

**代码阅读**:
- `models/qwen3.py` - Qwen3 模型
- `layers/attention.py` - 注意力机制
- `layers/linear.py` - 张量并行线性层
- `layers/rotary_embedding.py` - RoPE 位置编码
- `layers/layernorm.py` - RMSNorm
- `layers/sampler.py` - 采样器

**重点理解**:
- FlashAttention 的调用方式（varlen vs with_kvcache）
- 张量并行线性层的切分方式（ColumnParallel, RowParallel）
- RMSNorm 的计算方式

---

### Day 6: 工具函数与进阶特性

**目标**: 理解支持功能和技术细节

**学习内容**:
1. 上下文管理机制
2. 模型加载器
3. 前缀缓存的完整流程

**代码阅读**:
- `utils/context.py` - 上下文管理
- `utils/loader.py` - 模型加载器

**重点理解**:
- Global context 如何在不同层之间传递
- 权重合并机制（packed_modules_mapping）
- slot_mapping 的作用

**问题思考**:
- 前缀缓存命中时，如何避免重复计算？
- Prefill 阶段如何处理已缓存的 tokens？

---

### Day 7: 综合实践与深度理解

**目标**: 融会贯通，完成 Benchmark 分析

**学习内容**:
1. 运行 `python bench.py` 理解性能
2. 分析各优化技术的作用

**实践任务**:
1. 尝试不同的配置参数，理解其影响
2. 绘制数据流图（从请求到输出的完整流程）
3. 分析 CUDA Graph 的工作原理

**代码阅读**:
- `bench.py` - 性能测试

**总结思考**:
- nano-vllm 与 vLLM 的核心区别
- 可以进一步优化的方向

---

## 学习检查清单

- [ ] Day 1: 能运行 example.py，理解 LLM API
- [ ] Day 2: 能解释 prefill vs decode 的区别
- [ ] Day 3: 能解释 block manager 的工作原理
- [ ] Day 4: 能解释模型运行器的执行流程
- [ ] Day 5: 能解释各层的作用和实现
- [ ] Day 6: 能解释上下文管理和模型加载
- [ ] Day 7: 能完整画出数据流图

---

## 关键文件索引

| 文件 | 作用 |
|------|------|
| `llm.py` | 入口类 |
| `config.py` | 配置参数 |
| `sampling_params.py` | 采样参数 |
| `engine/llm_engine.py` | 引擎核心 |
| `engine/scheduler.py` | 调度器 |
| `engine/model_runner.py` | 模型运行器 |
| `engine/sequence.py` | 序列管理 |
| `engine/block_manager.py` | 块管理 |
| `models/qwen3.py` | 模型定义 |
| `layers/attention.py` | 注意力 |
| `layers/linear.py` | 线性层 |
| `layers/sampler.py` | 采样器 |
| `utils/context.py` | 上下文 |
| `utils/loader.py` | 加载器 |

---

## 推荐学习资源

1. [vLLM 论文](https://arxiv.org/abs/2309.06180) - 了解 PagedAttention
2. [FlashAttention 论文](https://arxiv.org/abs/2205.14135) - 了解 IO 优化
3. [Tensor Parallelism 博客](https://pytorch.org/tensorparallel) - 了解张量并行