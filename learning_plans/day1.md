# Day 1: 环境搭建与整体架构

> **目标**: 完成环境配置，理解项目整体架构

---

## 1. LLM 推理基础

### 1.1 LLM 推理的两个阶段

LLM 推理分为两个核心阶段：

| 阶段 | 名称 | 输入 | 输出 | 特点 |
|------|------|------|------|------|
| **Prefill** | 预填充 | 完整 prompt | 第一个 token | 计算密集，可并行处理所有 token |
| **Decode** | 解码 | 当前 token | 下一个 token | 内存密集，逐个生成 token |

```
Prefill 阶段:
Prompt: "什么是人工智能？" → 一次性计算所有 token 的 KV Cache
Output: 第一个生成的 token

Decode 阶段:
Step 1: 输入第一个生成 token → 输出第二个 token
Step 2: 输入第二个 token → 输出第三个 token
...以此类推直到生成结束
```

### 1.2 KV Cache 的作用

- **KV Cache** 存储注意力计算中的 Key 和 Value 矩阵
- 避免 Decode 阶段重复计算历史 token 的 K、V
- 是 LLM 推理优化的核心数据结构

### 1.3 核心优化技术概览

```
nano-vllm 的五大优化技术:

1. Prefix Caching (前缀缓存)
   └── 相同前缀的请求共享 KV Cache

2. Tensor Parallelism (张量并行)
   └── 多 GPU 并行计算模型层

3. Torch Compile (编译优化)
   └── PyTorch 2.0 编译器优化

4. CUDA Graph (图捕获)
   └── 减少 kernel 启动开销

5. FlashAttention (闪存注意力)
   └── GPU 内存访问优化
```

---

## 2. 环境搭建

### 2.1 系统要求

```bash
# 必需
- Python >= 3.9
- CUDA >= 11.8
- PyTorch >= 2.1

# 推荐
- NVIDIA GPU with >= 8GB VRAM (用于 0.6B 模型)
- GPU with >= 16GB VRAM (用于更大模型或多卡)
```

### 2.2 安装步骤

```bash
# Step 1: 克隆项目（如果还没有）
git clone https://github.com/GpuMist/nano-vllm.git
cd nano-vllm

# Step 2: 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate   # Windows

# Step 3: 安装依赖
pip install -e .

# Step 4: 验证安装
python -c "import nano_vllm; print('安装成功!')"
```

### 2.3 下载模型

```bash
# 安装 huggingface-cli
pip install huggingface_hub

# 下载 Qwen3-0.6B 模型（推荐新手）
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B

# 或设置环境变量自动下载
export HF_HOME=./models  # 模型将下载到此目录
```

### 2.4 验证环境

```bash
# 运行示例程序
python example.py

# 预期输出示例:
# Prompt: 你好
# Output: 你好！很高兴见到你。有什么我可以帮助你的吗？
```

---

## 3. 项目结构

### 3.1 目录结构

```
nano-vllm/
├── llm.py              # 入口类 LLM
├── config.py           # 配置参数
├── sampling_params.py  # 采样参数
├── example.py          # 使用示例
├── bench.py            # 性能测试
│
├── engine/             # 核心引擎
│   ├── llm_engine.py   # 引擎主类
│   ├── scheduler.py    # 调度器
│   ├── model_runner.py # 模型运行器
│   ├── sequence.py     # 序列管理
│   └── block_manager.py# 块管理器
│
├── models/             # 模型定义
│   └── qwen3.py        # Qwen3 模型
│
├── layers/             # 层定义
│   ├── attention.py    # 注意力层
│   ├── linear.py       # 线性层
│   ├── layernorm.py    # 归一化层
│   ├── rotary_embedding.py  # RoPE
│   └── sampler.py      # 采样器
│
└── utils/              # 工具函数
    ├── context.py      # 上下文管理
    └── loader.py       # 模型加载
```

### 3.2 调用链路

```
用户代码
    │
    ▼
llm.py: LLM.generate()
    │
    ▼
engine/llm_engine.py: LLMEngine.step()
    │
    ├──▶ scheduler.py: Scheduler.schedule()  → 决定执行哪些请求
    │
    ├──▶ model_runner.py: ModelRunner.run()  → 执行模型推理
    │
    └──▶ block_manager.py: BlockManager      → 管理 KV Cache
```

---

## 4. 代码阅读：入口类

### 4.1 llm.py - 主入口

**核心类**: `LLM`

**主要方法**:
- `__init__()`: 初始化引擎
- `generate()`: 生成文本
- `_run_engine()`: 运行引擎直到完成

**代码要点**:
```python
class LLM:
    def __init__(
        self,
        model: str,                    # 模型路径
        tensor_parallel_size: int = 1, # 张量并行数
        gpu_memory_utilization: float = 0.9,  # GPU 内存利用率
        max_num_batched_tokens: int = 32768,  # 最大批处理 token 数
    ):
        # 初始化引擎配置
        # 启动 Tensor Parallel 进程
        # 初始化 LLMEngine
        pass

    def generate(
        self,
        prompts: List[str],            # 输入提示词列表
        sampling_params: SamplingParams = None,  # 采样参数
        use_tqdm: bool = True,         # 是否显示进度条
    ) -> List[RequestOutput]:
        # 为每个 prompt 创建 Sequence
        # 调度并执行引擎
        # 收集并返回结果
        pass
```

### 4.2 config.py - 配置参数

**核心类**: `Config`

**关键参数说明**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_num_batched_tokens` | 32768 | 单次批处理最大 token 数 |
| `gpu_memory_utilization` | 0.9 | GPU 显存利用率上限 |
| `block_size` | 16 | 每个 block 存储的 token 数 |
| `max_model_len` | 8192 | 最大序列长度 |
| `tensor_parallel_size` | 1 | 张量并行 GPU 数量 |

**参数影响**:
```
max_num_batched_tokens ↑
├── 优点: 更大的 batch，更高的吞吐
└── 缺点: 需要更多显存

gpu_memory_utilization ↑
├── 优点: 更充分利用显存
└── 缺点: 可能导致 OOM

block_size ↑
├── 优点: 更好的内存连续性
└── 缺点: 内存碎片增加
```

### 4.3 sampling_params.py - 采样参数

**核心类**: `SamplingParams`

**参数说明**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n` | 1 | 每个 prompt 生成的序列数 |
| `max_tokens` | 128 | 最大生成 token 数 |
| `temperature` | 1.0 | 温度参数，控制随机性 |
| `top_p` | 1.0 | nucleus sampling 参数 |
| `top_k` | -1 | top-k sampling 参数 |
| `stop` | None | 停止词列表 |

**采样策略**:
```
temperature:
├── = 0: 贪婪采样（确定性）
├── < 1: 更确定的输出
└── > 1: 更随机的输出

top_p (nucleus sampling):
├── = 1.0: 不限制
└── < 1.0: 只从概率累计达到 top_p 的 token 中采样

top_k:
├── = -1: 不限制
└── > 0: 只从概率最高的 k 个 token 中采样
```

---

## 5. 实践练习

### 5.1 基础练习

**练习 1**: 运行 example.py，观察输出

```bash
python example.py
```

**练习 2**: 修改采样参数，观察效果

```python
from nano_vllm import LLM, SamplingParams

llm = LLM(model="./models/Qwen3-0.6B")

# 尝试不同的采样参数
params1 = SamplingParams(temperature=0.0, max_tokens=50)  # 确定性输出
params2 = SamplingParams(temperature=1.5, max_tokens=50)  # 更随机
params3 = SamplingParams(top_p=0.9, max_tokens=50)        # nucleus sampling

outputs = llm.generate(["你好"], sampling_params=params1)
print(outputs[0].outputs[0].text)
```

### 5.2 进阶练习

**练习 3**: 分析 LLM 类的初始化流程

```python
# 在 llm.py 中添加调试信息
# 追踪 __init__ 的执行顺序
```

**练习 4**: 理解配置参数的影响

```python
# 尝试不同的配置
llm = LLM(
    model="./models/Qwen3-0.6B",
    gpu_memory_utilization=0.5,  # 降低显存使用
    max_num_batched_tokens=16384,  # 减小 batch 大小
)
```

---

## 6. 知识检查点

完成以下问题以验证学习效果：

### 基础问题

1. ❓ Prefill 阶段和 Decode 阶段的主要区别是什么？

2. ❓ KV Cache 的作用是什么？为什么它能加速推理？

3. ❓ `max_num_batched_tokens` 参数如何影响系统性能？

### 进阶问题

4. ❓ 如果你有两个 GPU，如何配置 Tensor Parallel？

5. ❓ temperature=0 和 temperature=1 的输出有什么区别？

6. ❓ nano-vllm 的入口类 LLM 的主要职责是什么？

---

## 7. 常见问题

### Q1: CUDA out of memory 怎么办？

```python
# 方案 1: 降低 GPU 内存利用率
llm = LLM(model="...", gpu_memory_utilization=0.7)

# 方案 2: 减小 batch 大小
llm = LLM(model="...", max_num_batched_tokens=16384)

# 方案 3: 使用更小的模型
```

### Q2: 模型下载太慢怎么办？

```bash
# 使用镜像站
export HF_ENDPOINT=https://hf-mirror.com

# 或使用 modelscope
pip install modelscope
modelscope download --model Qwen/Qwen3-0.6B
```

### Q3: 如何验证 CUDA 环境正确？

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

---

## 8. 扩展阅读

### 推荐资源

1. **vLLM 论文**: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
   - 理解 PagedAttention 的核心思想

2. **LLM 推理入门**: [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
   - 可视化理解 Transformer 架构

3. **KV Cache 详解**: [KV Cache explained](https://medium.com/@joaolages/kv-cache-explained-9a4b2c50a8ae)
   - 深入理解 KV Cache 机制

### 代码位置速查

| 概念 | 文件 | 关键类/函数 |
|------|------|------------|
| 入口 API | `llm.py` | `LLM` |
| 配置 | `config.py` | `Config` |
| 采样参数 | `sampling_params.py` | `SamplingParams` |
| 引擎 | `engine/llm_engine.py` | `LLMEngine` |

---

## 9. 下一步

完成 Day 1 后，你应该能够：
- ✅ 成功运行 nano-vllm 示例
- ✅ 理解 LLM 推理的基本流程
- ✅ 知道如何调整基本的配置参数

**准备 Day 2**: 调度器与序列管理
- 深入理解 Prefill vs Decode 的调度策略
- 学习 Continuous Batching 的实现

---

*预计学习时间: 2-3 小时*
