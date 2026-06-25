# 用 nano-vllm 熟悉 vLLM 的需求清单

这份文档不是普通学习计划，而是一组可以逐个实现、验证和复盘的小需求。目标是通过 nano-vllm 的简化代码路径，摸清 vLLM 的核心能力：离线推理、采样、continuous batching、KV cache / prefix cache、PagedAttention 思想、CUDA Graph、张量并行、服务化接口、观测与性能分析，以及 vLLM 中更完整但 nano-vllm 尚未覆盖的高级功能。

建议做法：每个需求都先读关联源码，再写最小实现或实验脚本，最后记录现象、指标和你对 vLLM 机制的解释。

## 需求 1：补全采样参数，做一个可对比的生成实验台

### 背景

nano-vllm 当前的 `SamplingParams` 只有 `temperature`、`max_tokens`、`ignore_eos`，而 vLLM 的常用推理体验强依赖采样控制。先从采样入手，可以熟悉从 API 参数到 logits 处理再到输出 token 的完整链路。

### 要做什么

- 为 `SamplingParams` 增加 `top_p`、`top_k`、`seed`、`stop_token_ids`、`min_tokens` 中至少 3 个参数。
- 修改 `layers/sampler.py`，让采样逻辑支持新增参数。
- 写一个脚本，例如 `examples/sampling_playground.py`，对同一 prompt 分别跑低温、高温、top-p、top-k 等配置。
- 输出每组配置的文本、token ids、耗时和 tokens/s。

### 验收标准

- 相同 seed 下的输出可复现。
- `max_tokens`、`ignore_eos`、`stop_token_ids` 的停止行为清晰可测。
- 文档中能解释 `LLM.generate -> LLMEngine.step -> ModelRunner.run -> Sampler.forward` 的参数流动。

### 重点源码

- `nanovllm/sampling_params.py`
- `nanovllm/engine/sequence.py`
- `nanovllm/engine/model_runner.py`
- `nanovllm/layers/sampler.py`

### 对应 vLLM 功能

- Offline inference
- SamplingParams
- logits processors / sampling controls

## 需求 2：实现流式输出，理解 prefill 与 decode 的节奏

### 背景

vLLM 在线服务最常见的体验是 streaming。nano-vllm 的 `generate` 当前等待序列完成后统一返回，适合做离线推理，但不利于观察 decode 阶段每一步的调度。

### 要做什么

- 增加一个 `LLM.generate_stream()` 或独立脚本，按 token 或按文本增量 yield 输出。
- 每次 `step()` 后返回本轮完成的新 token，而不是只返回 finished sequence。
- 为每个请求输出 `seq_id`、新增 token、累计 token 数、是否 finished。
- 写一个 demo，同时提交多个 prompt，观察不同序列交错 decode 的过程。

### 验收标准

- 能看到 prefill 后，每轮 decode 为多个 running sequence 各生成 1 个 token。
- 任意一个序列结束后，不影响其他序列继续生成。
- 能解释为什么 decode 阶段天然适合 continuous batching。

### 重点源码

- `nanovllm/engine/llm_engine.py`
- `nanovllm/engine/scheduler.py`
- `nanovllm/engine/sequence.py`

### 对应 vLLM 功能

- Online serving
- Streaming
- Continuous batching

## 需求 3：给调度器加可观测事件，画出请求生命周期

### 背景

vLLM 的吞吐优势来自调度器和 KV cache 管理。只看最终输出，很难理解 waiting、running、finished 之间的状态转换，也看不到 preemption 什么时候发生。

### 要做什么

- 在 `Scheduler` 中增加轻量事件记录，例如 `request_added`、`prefill_scheduled`、`decode_scheduled`、`preempted`、`finished`。
- 每个事件记录时间、seq_id、队列长度、block 使用量、是否 prefix cache 命中。
- 写一个脚本生成压力场景：短 prompt、长 prompt、共享前缀 prompt、超长输出混合提交。
- 将事件导出为 JSONL 或 Markdown 表格。

### 验收标准

- 能从事件日志复原每个请求的生命周期。
- 能观察到 `max_num_seqs`、`max_num_batched_tokens` 调小时对调度的影响。
- 能说明 prefill 优先策略与 decode batch 的权衡。

### 重点源码

- `nanovllm/engine/scheduler.py`
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/llm_engine.py`

### 对应 vLLM 功能

- Scheduler
- Request lifecycle
- Continuous batching
- Preemption

## 需求 4：做 KV Cache / Prefix Cache 命中率实验

### 背景

PagedAttention 和 KV cache 是 vLLM 的核心。nano-vllm 已经有 block manager 和 prefix cache，适合做一个小型实验来理解块分配、引用计数、hash 命中和缓存复用。

### 要做什么

- 为 `BlockManager` 增加统计信息：分配次数、释放次数、cache hit blocks、cache miss blocks、ref_count 分布。
- 构造三类 prompt：
  - 完全无共享前缀。
  - 多请求共享长 system prompt。
  - 只有部分 block 对齐的共享前缀。
- 对比三类 prompt 的 prefill tokens、耗时和 cache 命中率。
- 画一张 block table 示例图，解释 prompt token 如何映射到 KV cache block。

### 验收标准

- 能证明共享 block_size 对齐前缀时，`num_cached_tokens` 会增加。
- 能解释为什么未满 block 不会被稳定缓存。
- 能说明 `hash_to_block_id`、`ref_count`、`free_block_ids` 的职责。

### 重点源码

- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/sequence.py`
- `nanovllm/engine/model_runner.py`
- `nanovllm/layers/attention.py`

### 对应 vLLM 功能

- KV cache
- PagedAttention
- Automatic prefix caching

## 需求 5：增加一个 OpenAI 兼容的最小 HTTP 服务

### 背景

vLLM 很重要的一部分价值是直接作为 OpenAI-compatible server 对外提供服务。nano-vllm 目前只有 Python API，补一个最小服务层可以帮你理解引擎与服务接口的边界。

### 要做什么

- 使用 FastAPI 或项目已有偏好的轻量框架，增加 `/v1/completions` 接口。
- 请求字段至少支持 `model`、`prompt`、`max_tokens`、`temperature`、`stream`。
- 非流式返回 OpenAI 风格 JSON。
- 流式返回 Server-Sent Events，每个 chunk 携带 delta text。
- 写一个 curl 示例和一个 Python client 示例。

### 验收标准

- 可以用 curl 调用非流式和流式接口。
- 多个并发请求能进入同一个 engine 调度，而不是每个请求单独加载模型。
- 能说明 HTTP 层、tokenizer、scheduler、model runner 的职责边界。

### 重点源码

- `nanovllm/llm.py`
- `nanovllm/engine/llm_engine.py`
- 可新增 `nanovllm/entrypoints/openai_api_server.py`

### 对应 vLLM 功能

- OpenAI-compatible server
- Online serving
- Streaming responses

## 需求 6：做性能基准矩阵，解释吞吐与延迟的来源

### 背景

只跑一次 `bench.py` 很难真正理解 vLLM 的性能模型。要熟悉 vLLM，需要能把 batch size、输入长度、输出长度、KV cache 容量、CUDA Graph、prefix cache 与吞吐/延迟联系起来。

### 要做什么

- 扩展 `bench.py`，支持参数矩阵：
  - `num_requests`
  - prompt length range
  - output length range
  - `max_num_seqs`
  - `max_num_batched_tokens`
  - `enforce_eager`
  - `gpu_memory_utilization`
- 输出 TTFT、TPOT、总吞吐、prefill 吞吐、decode 吞吐。
- 对比 `enforce_eager=True/False`，观察 CUDA Graph 的收益。
- 对比有共享前缀和无共享前缀的 benchmark。

### 验收标准

- 能产出一张 Markdown 表格或 CSV。
- 能解释吞吐提升来自 prefill、decode、CUDA Graph、prefix cache 中的哪一部分。
- 能指出某些配置为什么会 OOM、触发 preemption 或吞吐下降。

### 重点源码

- `bench.py`
- `nanovllm/config.py`
- `nanovllm/engine/model_runner.py`
- `nanovllm/engine/scheduler.py`

### 对应 vLLM 功能

- Benchmarking
- Profiling
- CUDA Graph
- Optimization and tuning

## 需求 7：跑通张量并行，并记录跨进程执行路径

### 背景

nano-vllm 已经有 `tensor_parallel_size`、NCCL、shared memory 和并行 linear layer。把单卡路径和多卡路径对照起来，可以理解 vLLM 的并行推理基础。

### 要做什么

- 在多 GPU 环境中运行 `tensor_parallel_size=2`。
- 为 rank 0 和 worker rank 增加简洁日志：进程启动、shared memory 读写、barrier、run、exit。
- 记录各 rank 的权重切分方式和通信点。
- 对比 TP=1 与 TP=2 的显存占用、吞吐和启动时间。

### 验收标准

- 能说明 rank 0 如何把 `run` 指令广播给其他 rank。
- 能说明 ColumnParallelLinear 与 RowParallelLinear 的切分和 all-reduce 位置。
- 能解释多卡不一定更快的原因：通信、模型大小、batch 规模和启动开销。

### 重点源码

- `nanovllm/engine/model_runner.py`
- `nanovllm/layers/linear.py`
- `nanovllm/utils/loader.py`

### 对应 vLLM 功能

- Tensor parallelism
- Distributed inference
- Parallelism and scaling

## 需求 8：实现一个最小 LoRA 推理接口

### 背景

vLLM 支持 LoRA adapter，是生产推理常见功能。nano-vllm 当前没有 LoRA，做一个最小版本可以帮你理解模型权重加载、线性层扩展和请求级 adapter 选择。

### 要做什么

- 先只支持 Qwen3 模型中的部分 linear 层，例如 q_proj、v_proj 或 o_proj。
- 定义一个 LoRA adapter 加载格式，读取 A/B 矩阵和 scaling。
- 支持请求级参数 `lora_name`，没有传时使用 base model。
- 做两个 adapter 切换实验，确认同一 prompt 输出不同。

### 验收标准

- base model 和 LoRA model 可以在同一进程内切换。
- LoRA 权重不破坏原始权重，关闭 adapter 后输出恢复 base 行为。
- 能解释 vLLM 为什么需要 adapter 管理、缓存和 batch 内混合 adapter 处理。

### 重点源码

- `nanovllm/models/qwen3.py`
- `nanovllm/layers/linear.py`
- `nanovllm/utils/loader.py`
- 可新增 `nanovllm/lora/`

### 对应 vLLM 功能

- LoRA adapters
- Request-level model customization

## 需求 9：增加结构化输出的最小约束解码

### 背景

vLLM 支持 structured outputs。完整实现会比较复杂，但可以先实现一个极简 logits processor，限制模型只能生成 JSON 所需的 token 子集或固定格式字段。

### 要做什么

- 增加 `SamplingParams.allowed_token_ids` 或 `logits_processor`。
- 在采样前对 logits 做 mask。
- 写一个 demo，让模型只生成 `{`、`}`、`"`、数字、逗号、冒号、指定字段名相关 token。
- 记录这种简单 token mask 的局限：tokenizer 粒度、中文/英文 token、无法保证完整语法。

### 验收标准

- 可以通过 logits mask 禁止不允许的 token。
- 能解释约束解码发生在模型 forward 之后、sample 之前。
- 能说清楚“token 级约束”和“语法级约束”的区别。

### 重点源码

- `nanovllm/sampling_params.py`
- `nanovllm/layers/sampler.py`
- `nanovllm/engine/model_runner.py`

### 对应 vLLM 功能

- Structured outputs
- Custom logits processors
- Guided decoding

## 需求 10：设计 speculative decoding 的实验原型

### 背景

speculative decoding 是 vLLM 的重要高级优化。nano-vllm 不一定适合马上完整实现，但可以先做原型设计，理解 draft model、target model、accept/reject 的关系。

### 要做什么

- 写一份设计文档：如何在 nano-vllm 中接入 draft model。
- 明确需要修改哪些接口：`ModelRunner.run`、`Scheduler.schedule`、`Sequence.append_token`、采样器。
- 做一个伪实现：draft model 用规则或随机 token 生成候选，target model 验证 1 到 N 个 token。
- 记录 acceptance rate、额外 forward 次数、理论收益。

### 验收标准

- 能画出普通 decode 与 speculative decode 的时序差异。
- 能解释什么时候 speculative decoding 会变慢。
- 能列出完整实现必须解决的问题：KV cache 回滚、多个 token append、batch 对齐、采样一致性。

### 重点源码

- `nanovllm/engine/llm_engine.py`
- `nanovllm/engine/scheduler.py`
- `nanovllm/engine/model_runner.py`
- `nanovllm/engine/sequence.py`

### 对应 vLLM 功能

- Speculative decoding
- Multi-token generation
- Advanced scheduler behavior

## 推荐完成顺序

| 顺序 | 需求 | 主要收获 |
| --- | --- | --- |
| 1 | 采样实验台 | 熟悉最短 API 到 token 输出链路 |
| 2 | 流式输出 | 看懂 prefill/decode 和 continuous batching |
| 3 | 调度事件 | 掌握请求生命周期和 preemption |
| 4 | KV Cache 实验 | 理解 PagedAttention 和 prefix cache |
| 5 | 性能基准矩阵 | 建立吞吐、延迟、显存的直觉 |
| 6 | HTTP 服务 | 理解 vLLM 作为服务的边界 |
| 7 | 张量并行 | 熟悉多进程、多卡和权重切分 |
| 8 | LoRA | 接触生产推理的 adapter 管理 |
| 9 | 结构化输出 | 理解 logits processor 和约束解码 |
| 10 | speculative decoding | 进入 vLLM 高级优化设计 |

## 每个需求的复盘模板

完成一个需求后，建议在单独的日志中回答这些问题：

- 我改了哪些文件？每个文件负责什么？
- 请求从进入 `LLM.generate` 到产出 token，经过了哪些对象？
- prefill 阶段和 decode 阶段分别做了什么？
- KV cache block 是什么时候分配、复用、释放的？
- 这个需求对应 vLLM 的哪个生产功能？
- 与真正 vLLM 相比，nano-vllm 的简化点在哪里？
- 我观察到的性能指标是什么？瓶颈可能在哪里？

## 参考入口

- vLLM 官方文档：https://docs.vllm.ai/en/latest/
- vLLM offline inference：https://docs.vllm.ai/en/latest/serving/offline_inference.html
- vLLM OpenAI-compatible server：https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- vLLM Automatic Prefix Caching：https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html
- vLLM Speculative Decoding：https://docs.vllm.ai/en/latest/features/spec_decode.html
- vLLM Structured Outputs：https://docs.vllm.ai/en/latest/features/structured_outputs.html
- vLLM LoRA：https://docs.vllm.ai/en/latest/features/lora.html

