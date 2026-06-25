# Nano-vLLM 项目概述

## 1. 项目简介

### 1.1 项目背景

Nano-vLLM 是一个轻量级的 vLLM 实现，从零开始构建，旨在提供一个易于理解和学习的 LLM 推理引擎。项目使用约 1,200 行 Python 代码实现了与 vLLM 相当的推理性能，是学习 LLM 推理系统设计的绝佳资源。

### 1.2 项目定位

| 维度 | 描述 |
|------|------|
| **目标用户** | 想要理解 LLM 推理引擎内部原理的开发者、研究人员 |
| **核心价值** | 提供一个最小化但功能完整的推理引擎实现 |
| **代码规模** | ~1,200 行 Python 代码 |
| **性能水平** | 与 vLLM 相当的推理吞吐量 |

### 1.3 项目目标

```
┌─────────────────────────────────────────────────────────────┐
│                        项目目标                              │
├─────────────────────────────────────────────────────────────┤
│  1. 教育性：提供清晰、易读的代码，便于学习 LLM 推理原理      │
│  2. 完整性：实现生产级推理引擎的核心功能                     │
│  3. 高性能：达到与 vLLM 相当的推理吞吐量                     │
│  4. 可扩展：支持多种优化技术，便于进一步开发                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 核心特性

### 2.1 功能特性

| 特性 | 描述 | 实现文件 |
|------|------|----------|
| **Prefix Caching** | 前缀缓存，避免重复计算相同前缀 | `engine/block_manager.py` |
| **Tensor Parallelism** | 张量并行，支持多 GPU 推理 | `layers/linear.py`, `layers/embed_head.py` |
| **Torch Compile** | PyTorch 2.0 编译优化 | `layers/sampler.py`, `layers/layernorm.py` |
| **CUDA Graph** | 减少 kernel 启动开销 | `engine/model_runner.py` |
| **FlashAttention** | 高效注意力计算 | `layers/attention.py` |
| **Continuous Batching** | 连续批处理调度 | `engine/scheduler.py` |
| **PagedAttention** | 分页注意力机制 | `engine/block_manager.py` |

### 2.2 性能数据

**测试配置：**
- 硬件：RTX 4070 Laptop (8GB)
- 模型：Qwen3-0.6B
- 请求数：256 sequences
- 输入长度：100-1024 tokens 随机
- 输出长度：100-1024 tokens 随机

**性能对比：**

| 推理引擎 | 输出 Tokens | 时间 (s) | 吞吐量 (tokens/s) |
|---------|------------|----------|------------------|
| vLLM | 133,966 | 98.37 | 1,361.84 |
| Nano-vLLM | 133,966 | 93.41 | **1,434.13** |

---

## 3. 系统边界

### 3.1 功能范围

```
┌─────────────────────────────────────────────────────────────┐
│                     Nano-vLLM 功能范围                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ✅ 支持：                                                   │
│  ├── 离线批量推理                                            │
│  ├── 多 GPU 张量并行                                        │
│  ├── Prefix Caching                                         │
│  ├── CUDA Graph 优化                                        │
│  ├── FlashAttention 集成                                    │
│  ├── Continuous Batching                                    │
│  └── Qwen3 模型支持                                         │
│                                                             │
│  ❌ 不支持：                                                 │
│  ├── 在线服务 API                                           │
│  ├── 多模型同时加载                                         │
│  ├── 流式输出                                               │
│  ├── Beam Search                                            │
│  └── 非自回归解码                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 技术依赖

| 依赖 | 版本要求 | 用途 |
|------|---------|------|
| Python | 3.10 - 3.12 | 运行环境 |
| PyTorch | >= 2.4.0 | 深度学习框架 |
| Triton | >= 3.0.0 | GPU kernel 编写 |
| Transformers | >= 4.51.0 | 模型配置加载 |
| Flash-Attn | - | 高效注意力计算 |
| xxhash | - | 哈希计算（前缀缓存） |

---

## 4. 项目结构

```
nano-vllm/
├── nanovllm/                    # 核心包
│   ├── __init__.py             # 包入口
│   ├── llm.py                  # LLM 主入口类
│   ├── config.py               # 配置类
│   ├── sampling_params.py      # 采样参数
│   │
│   ├── engine/                 # 推理引擎
│   │   ├── llm_engine.py       # 引擎核心
│   │   ├── scheduler.py        # 调度器
│   │   ├── model_runner.py     # 模型运行器
│   │   ├── sequence.py         # 序列管理
│   │   └── block_manager.py    # KV Cache 块管理
│   │
│   ├── models/                 # 模型实现
│   │   └── qwen3.py            # Qwen3 模型
│   │
│   ├── layers/                 # 网络层
│   │   ├── attention.py        # 注意力层
│   │   ├── linear.py           # 线性层（张量并行）
│   │   ├── layernorm.py        # RMSNorm
│   │   ├── rotary_embedding.py # RoPE 位置编码
│   │   ├── sampler.py          # 采样器
│   │   ├── activation.py       # 激活函数
│   │   └── embed_head.py       # Embedding 和 LM Head
│   │
│   └── utils/                  # 工具函数
│       ├── context.py          # 上下文管理
│       └── loader.py           # 模型加载器
│
├── example.py                   # 使用示例
├── bench.py                     # 性能测试
├── pyproject.toml              # 项目配置
└── docs/                        # 文档目录
```

---

## 5. 术语定义

| 术语 | 英文 | 定义 |
|------|------|------|
| Prefill | Prefill Phase | 预填充阶段，处理输入 prompt，生成初始 KV Cache |
| Decode | Decode Phase | 解码阶段，逐个生成输出 token |
| KV Cache | Key-Value Cache | 注意力层的缓存，避免重复计算 |
| Block | KV Cache Block | KV Cache 的分块单位，默认 256 tokens |
| Continuous Batching | Continuous Batching | 连续批处理，动态调整批次中的序列 |
| Prefix Caching | Prefix Caching | 缓存相同前缀的 KV Cache，避免重复计算 |
| Tensor Parallelism | Tensor Parallelism | 张量并行，将模型层切分到多 GPU |
| CUDA Graph | CUDA Graph | CUDA 图，减少 kernel 启动开销 |
| FlashAttention | FlashAttention | 高效注意力计算算法 |
| Slot Mapping | Slot Mapping | token 到 KV Cache 位置的映射 |

---

## 6. 快速开始

### 6.1 安装

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

### 6.2 模型下载

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

### 6.3 基本使用

```python
from nanovllm import LLM, SamplingParams

# 初始化
llm = LLM("/path/to/model", enforce_eager=True, tensor_parallel_size=1)

# 设置采样参数
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

# 生成
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)

# 输出结果
print(outputs[0]["text"])
```

---

## 7. 里程碑

| 版本 | 目标 | 状态 |
|------|------|------|
| v0.1.0 | 基础推理功能 | ✅ 完成 |
| v0.2.0 | Tensor Parallelism 支持 | ✅ 完成 |
| v0.3.0 | Prefix Caching | ✅ 完成 |
| v0.4.0 | CUDA Graph 优化 | ✅ 完成 |
| 未来 | 更多模型支持、流式输出 | 🚧 规划中 |

---

## 8. 学习路径

建议按以下顺序阅读源码：

```
1. 入口层
   └── llm.py → config.py → sampling_params.py

2. 引擎层
   └── engine/llm_engine.py
   └── engine/scheduler.py
   └── engine/sequence.py
   └── engine/block_manager.py
   └── engine/model_runner.py

3. 模型层
   └── models/qwen3.py
   └── layers/attention.py
   └── layers/linear.py
   └── layers/rotary_embedding.py
   └── layers/sampler.py

4. 工具层
   └── utils/context.py
   └── utils/loader.py
```

---

## 9. 参考资料

1. [vLLM 论文](https://arxiv.org/abs/2309.06180) - PagedAttention 原理
2. [FlashAttention 论文](https://arxiv.org/abs/2205.14135) - IO 优化注意力
3. [Tensor Parallelism](https://pytorch.org/tensorparallel) - PyTorch 张量并行
4. [Qwen3 模型](https://huggingface.co/Qwen/Qwen3-0.6B) - 支持的模型
