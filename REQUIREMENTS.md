# nano-vllm 深度实践需求文档

> 目标：通过实现以下需求，彻底熟悉 vLLM 推理引擎的核心功能与设计理念。
> 每个需求由浅入深，覆盖调度、内存管理、采样、模型支持、性能优化、工程化等关键模块。

---

## 需求一：扩展模型支持 —— 接入 Llama 模型

**涉及模块：** `models/`, `layers/`, `utils/loader.py`

**背景：** 当前 nano-vllm 仅支持 Qwen3 架构。vLLM 生产环境支持数十种模型，理解模型接入流程是掌握推理引擎的关键第一步。

**任务：**
1. 在 `models/` 下新建 `llama.py`，参照 `qwen3.py` 实现完整的 LlamaForCausalLM
   - Llama 与 Qwen3 的核心差异：
     - Llama 没有 Q/K normalization（Qwen3 在无 bias 时会做 QK Norm）
     - Llama 的 MLP 使用 `gate_proj + up_proj + down_proj`（与 Qwen3 相同，可复用 `MergedColumnParallelLinear`）
     - Llama 的 `o_proj` 没有 `all_reduce`（非 TP 场景），TP 场景下同 Qwen3
     - 注意 `tie_word_embeddings` 的处理（Llama 通常不绑定）
2. 在 `utils/loader.py` 中注册 Llama 的参数映射（只需 `q_proj/k_proj/v_proj → qkv_proj`，`gate_proj/up_proj → gate_up_proj`）
3. 验证：加载 `meta-llama/Llama-3.2-1B`（或 HuggingFace 上任意小 Llama 模型），跑通 `example.py`，确保输出合理

**学习目标：**
- 理解不同模型架构如何适配统一的 Engine 层
- 掌握 Tensor Parallel 下的权重分片策略（ColumnParallel / RowParallel）
- 理解 packed module 映射机制

---

## 需求二：完善采样策略 —— Top-K / Top-P / 重复惩罚

**涉及模块：** `layers/sampler.py`, `sampling_params.py`, `engine/llm_engine.py`

**背景：** 当前 Sampler 仅实现了最简单的 temperature + Gumbel-max 采样。真实场景需要更丰富的解码策略来控制生成质量。

**任务：**
1. 扩展 `SamplingParams`，新增参数：
   - `top_k: int = -1`（-1 表示禁用）
   - `top_p: float = 1.0`（1.0 表示禁用）
   - `repetition_penalty: float = 1.0`（1.0 表示不惩罚）
   - `frequency_penalty: float = 0.0`
   - `presence_penalty: float = 0.0`
   - `seed: int | None = None`（用于可复现采样）
2. 重写 `Sampler.forward`：
   - 先应用 repetition_penalty：对已生成的 token 降低其 logit 值
   - 再应用 frequency_penalty / presence_penalty
   - 然后做 top-k 过滤（保留概率最高的 k 个 token）
   - 接着做 top-p 过滤（保留累积概率刚好超过 p 的最小 token 集合）
   - 最后做 temperature + 采样
3. **关键细节**：repetition_penalty 应该只对已生成的 token 施加惩罚，需要在 `Sequence` 对象中维护 token 出现计数。考虑将 `prompt_token_ids` 和 `output_token_ids` 合并处理。
4. 验证：设计对比实验，证明 top_p=0.9 比 top_p=1.0 生成质量更好（更少的重复/乱码），repetition_penalty=1.2 能有效抑制重复。

**学习目标：**
- 深入理解 logits 后处理管线
- 掌握贪心解码 vs 随机采样的权衡
- 理解各惩罚项对生成行为的影响

---

## 需求三：实现分块预填充（Chunked Prefill）

**涉及模块：** `engine/scheduler.py`, `engine/model_runner.py`, `engine/sequence.py`

**背景：** 当前调度器的策略是"先完成所有 prefill，再 decode"。当一条长 prompt 到达时，所有 decode 序列都必须等待，导致 **TTFT（Time To First Token）延迟抖动**。Chunked Prefill 将长 prompt 拆分为多个小块，与 decode 交替执行。

**任务：**
1. 修改 `Scheduler.schedule()` 的调度逻辑：
   - 引入 `max_num_prefill_tokens` 参数（每个 step 最多 prefill 的 token 数）
   - 当一个 WAITING 序列的 prompt 过长时，只取前 `max_num_prefill_tokens` 个 token 做 prefill
   - 该序列 prefill 完一个 chunk 后留在 WAITING 队列（而非进入 RUNNING），等待下一次被调度
   - 当该序列的所有 prompt token 都被 prefill 完成后，才进入 RUNNING 状态
2. 修改 `Sequence` 类：
   - 增加 `num_prefilled_tokens: int = 0` 字段，记录已 prefill 的 token 数
   - 区分"完全 prefilled"和"部分 prefilled"两种中间状态
3. 修改 `ModelRunner.prepare_prefill()`：
   - 确保对部分 prefilled 的序列，正确设置位置编码（position 从 num_computed_tokens 开始）
   - 正确处理 KV cache slot 分配
4. 验证：构造一条 4096 token 的长 prompt + 多条 16 token 的短 prompt，对比改造前后的 TTFT 分布

**学习目标：**
- 理解推理系统的首 token 延迟优化
- 掌握调度器如何平衡吞吐与延迟
- 体会"将大任务拆分为小任务"在实时系统中的重要性

---

## 需求四：KV Cache CPU 卸载与 Swap

**涉及模块：** `engine/block_manager.py`, `engine/scheduler.py`, `config.py`

**背景：** 当前 preemption 策略是"丢弃最新序列，重新计算"。这在内存紧张时浪费了已计算的 KV Cache。生产 vLLM 支持将 KV block 从 GPU 交换到 CPU，preempt 时先尝试 swap 而非直接丢弃。

**任务：**
1. 在 `BlockManager` 中增加 CPU cache 支持：
   - 分配一段 CPU memory 作为 swap 空间（`num_swap_blocks` 通过配置或自动计算）
   - 实现 `swap_out(block_id) -> int`：将 GPU block 复制到 CPU，返回 swap 索引
   - 实现 `swap_in(swap_index) -> int`：将 CPU block 复制回 GPU，返回 GPU block_id
   - 维护 swap 空间的空闲/占用位图
2. 修改 `Scheduler` 的 preemption 逻辑：
   - Preempt 时优先 swap 而非 recompute
   - 只有当 swap 空间也满了才回退到 recompute
3. 修改 `Sequence` 状态机：
   - 增加 `PREEMPTED` 状态，区分"swap 后等待恢复"和"recompute 后等待恢复"
   - 恢复时优先从 CPU swap 回来（D2H → H2D 传输）
4. 验证：在接近显存极限的条件下运行，观察 swap 是否发生、swap 后恢复是否正确

**学习目标：**
- 理解异构内存管理（GPU ↔ CPU）在推理系统中的应用
- 掌握 preemption 的分级策略（swap → recompute）
- 体会内存管理对系统吞吐的影响

---

## 需求五：实现流式输出（Streaming）

**涉及模块：** `engine/llm_engine.py`, `llm.py`

**背景：** 当前 `generate()` 是同步阻塞的，等全部 token 生成完毕后一次性返回。实际应用（如聊天机器人）需要逐 token 返回以改善用户体验。

**任务：**
1. 为 `LLMEngine` 添加 `generate_stream()` 方法，改为生成器：
   ```python
   def generate_stream(self, prompts, sampling_params) -> Iterator[list[TokenUpdate]]:
       # 每步 step() 后 yield 新生成的 token
   ```
2. 定义 `TokenUpdate` 数据结构：
   - `seq_index: int` — 属于哪个输入 prompt
   - `token_id: int` — 新生成的 token
   - `token_text: str` — 解码后的文本
   - `is_finished: bool` — 该序列是否已结束
   - `finish_reason: str | None` — 结束原因（"stop" / "length" / "abort"）
3. 在 `LLM` 类中暴露 `generate_stream()` 接口
4. 编写 `example_stream.py`，演示逐 token 打印效果
5. **扩展（可选）**：使用 `asyncio` 实现异步版本的 `agenerate_stream()`

**学习目标：**
- 理解生成器/异步迭代器在推理引擎中的应用
- 掌握如何在不破坏调度循环的前提下实现流式输出
- 体验服务层与引擎层的解耦设计

---

## 需求六：性能剖析与 CUDA Graph 深度理解

**涉及模块：** `engine/model_runner.py`, `layers/attention.py`

**背景：** nano-vllm 宣称性能接近 vLLM，但你真的理解为什么快吗？这个需求要求你用性能分析工具深入理解每一微秒的耗时。

**任务：**
1. **Profile 工具集成**：
   - 使用 `torch.profiler` 对一次完整的 `generate()` 调用进行 profiling
   - 导出 Chrome Trace，在 `chrome://tracing` 中可视化
   - 标注出以下阶段的时间占比：tokenization → prefill → decode → sampling → postprocess
2. **CUDA Graph 对比实验**：
   - 分别测量 `enforce_eager=True` 和 `enforce_eager=False` 的吞吐差异
   - 统计 CUDA Graph capture 耗时 vs 收益（decode 阶段节省了多少 kernel launch overhead）
   - 实验不同 batch size 下 CUDA Graph 的收益变化曲线（batch_size=1/4/16/64/256）
3. **FlashAttention 对比**：
   - 临时将 FlashAttention 替换为 PyTorch 原生 scaled_dot_product_attention（用环境变量控制）
   - 对比 prefill 和 decode 阶段的耗时差异
4. **瓶颈分析**：
   - 分析 prefill 阶段是 compute-bound 还是 memory-bound
   - 分析 decode 阶段是 compute-bound 还是 memory-bound
   - 基于分析结论，提出至少一条针对性的优化建议
5. **编写性能分析报告**（保存为 `BENCHMARK_REPORT.md`）：
   - 包含上述所有实验的图表/数据
   - 给出各模块耗时占比的饼图（用 ASCII art 或表格）
   - 总结 nano-vllm 的性能关键路径

**学习目标：**
- 掌握 PyTorch profiling 工具链
- 深刻理解 CUDA Graph 的工作机制及其适用场景
- 学会区分 compute-bound vs memory-bound，指导后续优化方向
- 培养数据驱动的性能调优思维

---

## 需求七：构建 OpenAI 兼容 API 服务

**涉及模块：** 新建 `server/` 目录

**背景：** vLLM 的一大亮点是开箱即用的 OpenAI 兼容 API。这个需求让你从零搭建一个轻量级 HTTP 服务。

**任务：**
1. 搭建 FastAPI 服务（`server/api.py`）：
   - `POST /v1/chat/completions` — Chat Completion 接口
   - `POST /v1/completions` — Text Completion 接口
   - `GET /v1/models` — 模型列表
   - `GET /health` — 健康检查
2. 实现请求/响应模型（`server/protocol.py`）：
   - ChatCompletionRequest / ChatCompletionResponse
   - 支持 `messages` 格式，自动应用 Chat Template（用 tokenizer 内置的 `apply_chat_template`）
   - 支持 `stream: bool` 参数（配合需求五的流式输出）
3. 实现 `server/engine_client.py`：
   - 将 `LLMEngine` 封装为异步客户端
   - 管理请求队列，支持并发请求
   - 处理取消/超时
4. 编写 `server/__main__.py` 启动脚本：
   ```bash
   python -m server --model Qwen/Qwen3-0.6B --port 8000
   ```
5. 验证：使用 `curl` 或 OpenAI Python SDK 测试所有接口
   ```bash
   curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"qwen3","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
   ```

**学习目标：**
- 理解推理引擎如何与服务层解耦
- 掌握异步并发请求处理
- 了解 OpenAI API 协议的请求/响应规范

---

## 需求八：高级 Prefix Caching —— 跨请求共享优化

**涉及模块：** `engine/block_manager.py`, `engine/model_runner.py`

**背景：** 当前 BlockManager 已实现基础的 hash-based prefix caching。但在实际 LLM 应用中，system prompt 往往跨请求完全一致，现有的链式 hash 方案在以下场景不够健壮：
- 不同请求共用相同 system prompt 前缀，但中间 token 不同
- 前缀缓存命中后，cache_miss 标志导致后续块全部重新分配（即使某些块可以命中）

**任务：**
1. 改进 `BlockManager.allocate()` 的缓存逻辑：
   - 移除"一旦 cache_miss 则后续全部 miss"的限制
   - 改为每个 block 独立判断：如果 hash 命中且 token_ids 完全匹配 → 共享；否则分配新块
   - **关键问题**：当块被多个不同序列引用且内容不同时，hash 冲突如何解决？考虑引入"先匹配 hash，再逐 token 验证"的机制
2. 增加缓存统计：
   - `cache_hit_rate`：命中的 block 数 / 总请求的 block 数
   - `cache_saved_tokens`：因缓存避免重新计算的 token 数
   - 在日志中输出缓存命中率
3. 实现缓存驱逐（Eviction）：
   - 当空闲块不足时，驱逐引用计数为 1 的块（未被其他序列共享）
   - 驱逐策略：LRU（Least Recently Used）
   - 被驱逐的块：如果对应序列仍在运行，标记为"需重新计算"
4. 编写 `test_prefix_cache.py`：
   - 构造 100 条共享相同 system prompt（~500 tokens）的请求
   - 对比启用/禁用前缀缓存的 prefill 耗时差异
   - 验证跨请求的缓存命中率

**学习目标：**
- 深入理解 KV Cache 块管理的哈希链与引用计数
- 掌握缓存驱逐策略（LRU）在推理系统中的应用
- 理解生产环境中 system prompt 对缓存命中的重要性

---

## 学习路径建议

| 顺序 | 需求 | 难度 | 预计耗时 | 核心收获 |
|------|------|------|----------|----------|
| 1 | 需求二：采样策略 | ⭐⭐ | 3-4h | 理解 logits 后处理管线 |
| 2 | 需求一：Llama 模型 | ⭐⭐⭐ | 4-6h | 掌握模型接入与 TP 权重分片 |
| 3 | 需求五：流式输出 | ⭐⭐ | 2-3h | 理解生成器模式与引擎解耦 |
| 4 | 需求三：Chunked Prefill | ⭐⭐⭐⭐ | 6-8h | 掌握调度器核心优化 |
| 5 | 需求四：KV Cache Swap | ⭐⭐⭐⭐ | 5-7h | 理解异构内存管理 |
| 6 | 需求六：性能剖析 | ⭐⭐⭐ | 4-6h | 培养性能分析能力 |
| 7 | 需求八：Prefix Caching | ⭐⭐⭐⭐ | 6-8h | 深入块管理和缓存策略 |
| 8 | 需求七：API 服务 | ⭐⭐⭐ | 4-6h | 掌握推理即服务的工程化 |

> **建议：** 按顺序完成需求 1-5 即可覆盖 nano-vllm 的核心模块（模型层→采样层→调度层→内存层→服务层）。
> 需求 6-8 适合作为进阶挑战，分别侧重性能调优、缓存优化和生产化部署。
