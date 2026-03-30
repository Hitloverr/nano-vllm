# Day 5: 模型实现与层定义

> **目标**: 理解模型结构和各层的实现细节

---

## 1. Qwen3 模型架构概览

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Qwen3 模型架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入: input_ids [batch, seq_len]                           │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────────────┐           │
│  │           Embedding Layer                    │           │
│  │   vocab_size × hidden_dim                   │           │
│  └─────────────────────────────────────────────┘           │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────────────┐           │
│  │         Transformer Block × N               │           │
│  │  ┌─────────────────────────────────────┐   │           │
│  │  │         RMSNorm                      │   │           │
│  │  └────────────────┬────────────────────┘   │           │
│  │                   ▼                         │   │           │
│  │  ┌─────────────────────────────────────┐   │           │
│  │  │    Attention (with RoPE)            │   │           │
│  │  │    QKV → FlashAttention → Output    │   │           │
│  │  └────────────────┬────────────────────┘   │           │
│  │                   │ + Residual             │   │           │
│  │                   ▼                         │   │           │
│  │  ┌─────────────────────────────────────┐   │           │
│  │  │         RMSNorm                      │   │           │
│  │  └────────────────┬────────────────────┘   │           │
│  │                   ▼                         │   │           │
│  │  ┌─────────────────────────────────────┐   │           │
│  │  │         MLP (FFN)                    │   │           │
│  │  │  Gate → Up → Down                    │   │           │
│  │  └────────────────┬────────────────────┘   │           │
│  │                   │ + Residual             │   │           │
│  │                   ▼                         │   │           │
│  └─────────────────────────────────────────────┘           │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────────────┐           │
│  │         Final RMSNorm                       │           │
│  └─────────────────────────────────────────────┘           │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────────────┐           │
│  │         LM Head                             │           │
│  │   hidden_dim → vocab_size                  │           │
│  └─────────────────────────────────────────────┘           │
│         │                                                   │
│         ▼                                                   │
│  输出: logits [batch, seq_len, vocab_size]                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 模型配置参数

```python
# Qwen3-0.6B 配置示例
config = {
    "hidden_size": 896,           # 隐藏层维度
    "intermediate_size": 4864,    # MLP 中间层维度
    "num_attention_heads": 14,    # 注意力头数
    "num_hidden_layers": 24,      # Transformer 层数
    "num_key_value_heads": 2,     # KV 头数 (GQA)
    "vocab_size": 151936,         # 词表大小
    "max_position_embeddings": 32768,  # 最大位置编码
    "rms_norm_eps": 1e-6,         # RMSNorm epsilon
    "rope_theta": 1000000.0,      # RoPE base
}
```

### 1.3 Grouped Query Attention (GQA)

```
┌─────────────────────────────────────────────────────────────┐
│              Grouped Query Attention 示意图                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Multi-Head Attention (MHA):                                │
│  Heads: 14 Q, 14 K, 14 V                                    │
│  ┌───┬───┬───┬───┬───┬───┬───┐                             │
│  │Q0 │Q1 │Q2 │Q3 │Q4 │Q5 │...│  每个 Q 有独立的 K, V        │
│  │K0 │K1 │K2 │K3 │K4 │K5 │...│                             │
│  │V0 │V1 │V2 │V3 │V4 │V5 │...│                             │
│  └───┴───┴───┴───┴───┴───┴───┘                             │
│                                                             │
│  Grouped Query Attention (GQA):                             │
│  Heads: 14 Q, 2 K, 2 V (groups = 7)                         │
│  ┌───┬───┬───┬───┬───┬───┬───┐                             │
│  │Q0 │Q1 │Q2 │Q3 │Q4 │Q5 │...│  多个 Q 共享同一个 K, V      │
│  │   K0    │   K1    │  ... │  减少 KV Cache 大小          │
│  │   V0    │   V1    │  ... │                             │
│  └───┴───┴───┴───┴───┴───┴───┘                             │
│                                                             │
│  KV Cache 节省: 14/2 = 7x                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 代码阅读：模型定义

### 2.1 models/qwen3.py 核心结构

```python
class Qwen3(nn.Module):
    """Qwen3 模型实现"""

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Embedding 层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer 层
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

        # 最终归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM Head (输出投影)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,  # Prefill
        block_tables: Optional[torch.Tensor] = None,  # Decode
        **kwargs,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: 输入 token IDs
            position_ids: 位置 IDs
            cu_seqlens: Prefill 时的累积序列长度
            block_tables: Decode 时的 KV Cache 块表
        """
        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # 通过所有 Transformer 层
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                block_tables=block_tables,
            )

        # 最终归一化
        hidden_states = self.norm(hidden_states)

        # LM Head
        logits = self.lm_head(hidden_states)

        return logits
```

### 2.2 Transformer 层实现

```python
class Qwen3DecoderLayer(nn.Module):
    """单个 Transformer 解码层"""

    def __init__(self, config: ModelConfig):
        super().__init__()

        # 注意力前的归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 自注意力
        self.self_attn = Qwen3Attention(config)

        # MLP 前的归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MLP
        self.mlp = Qwen3MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        前向传播 (带残差连接)
        """
        residual = hidden_states

        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
```

### 2.3 MLP 层实现

```python
class Qwen3MLP(nn.Module):
    """
    SwiGLU MLP 实现

    结构: Gate → Up → Down
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        # Gate 投影
        self.gate_proj = RowParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        # Up 投影
        self.up_proj = RowParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        # Down 投影
        self.down_proj = ColumnParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

        # 激活函数
        self.act_fn = nn.SiLU()  # Swish 激活

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: down(silu(gate(x)) * up(x))
        """
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)

        # SwiGLU: gate * up (element-wise)
        hidden = self.act_fn(gate_output) * up_output

        # 投影回原维度
        output = self.down_proj(hidden)

        return output
```

---

## 3. 代码阅读：注意力层

### 3.1 layers/attention.py 核心结构

```python
class Qwen3Attention(nn.Module):
    """Qwen3 注意力层实现"""

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # QKV 投影
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )

        # Output 投影
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        # RoPE 位置编码
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Prefill: 使用 FlashAttention varlen
        Decode: 使用 FlashAttention with_kvcache
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 计算 Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 重塑为 [batch, seq, heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # 应用 RoPE 位置编码
        q, k = self.rotary_emb(q, k, position_ids)

        # 根据 prefill 或 decode 选择不同的注意力计算
        if cu_seqlens is not None:
            # Prefill: varlen attention
            output = self._attention_prefill(q, k, v, cu_seqlens)
        else:
            # Decode: with kvcache
            output = self._attention_decode(q, k, v, block_tables)

        # 重塑并投影
        output = output.view(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output
```

### 3.2 FlashAttention 调用

```python
def _attention_prefill(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """
    Prefill 阶段: 使用 FlashAttention varlen

    特点:
    - 处理变长序列
    - 写入 KV Cache
    - 支持前缀缓存跳过
    """
    from flash_attn import flash_attn_varlen_func

    # 将 Q, K, V 展平为 1D
    q = q.view(-1, self.num_heads, self.head_dim)
    k = k.view(-1, self.num_kv_heads, self.head_dim)
    v = v.view(-1, self.num_kv_heads, self.head_dim)

    # 计算 max_seqlen
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    # 调用 FlashAttention
    output = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=1.0 / math.sqrt(self.head_dim),
        causal=True,  # 因果注意力
    )

    return output


def _attention_decode(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_tables: torch.Tensor,
) -> torch.Tensor:
    """
    Decode 阶段: 使用 FlashAttention with_kvcache

    特点:
    - 使用 KV Cache
    - 支持 GQA
    - 高效的内存访问
    """
    from flash_attn import flash_attn_with_kvcache

    # 获取 KV Cache 引用
    k_cache = get_global_kv_cache('k')
    v_cache = get_global_kv_cache('v')

    # 当前序列长度
    cache_seqlens = position_ids + 1

    # 调用 FlashAttention
    output = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_tables,
        cache_seqlens=cache_seqlens,
        softmax_scale=1.0 / math.sqrt(self.head_dim),
    )

    return output
```

---

## 4. 代码阅读：线性层

### 4.1 layers/linear.py 张量并行线性层

```python
class ColumnParallelLinear(nn.Module):
    """
    列并行线性层

    输出维度按列切分到多个 GPU

    示例: hidden_size=896, vocab_size=151936, tp=2
    - GPU 0: 处理 vocab [0, 75968)
    - GPU 1: 处理 vocab [75968, 151936)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_size: int = 1,
    ):
        super().__init__()

        self.tp_size = tp_size
        self.out_features_per_partition = out_features // tp_size

        # 只存储当前 GPU 负责的部分
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_partition)
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        输入: [batch, seq, in_features]
        输出: [batch, seq, out_features_per_partition]

        后续需要 All-Gather 汇总所有 GPU 的输出
        """
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """
    行并行线性层

    输入维度按行切分到多个 GPU

    示例: hidden_size=896, intermediate_size=4864, tp=2
    - GPU 0: 处理 input [0, 2432)
    - GPU 1: 处理 input [2432, 4864)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp_size: int = 1,
    ):
        super().__init__()

        self.tp_size = tp_size
        self.in_features_per_partition = in_features // tp_size

        # 只存储当前 GPU 负责的部分
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )

        if bias:
            # bias 只在一个 GPU 上初始化，其他为 0
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        输入: [batch, seq, in_features_per_partition]
        输出: [batch, seq, out_features]

        需要 All-Reduce 汇总所有 GPU 的结果
        """
        output = F.linear(x, self.weight)
        output = all_reduce(output)  # Tensor Parallel 通信
        if self.bias is not None:
            output = output + self.bias
        return output
```

### 4.2 张量并行切分示意图

```
┌─────────────────────────────────────────────────────────────┐
│                 张量并行线性层切分示意                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Column Parallel (如 LM Head):                              │
│  ┌─────────────────────────────────────────┐               │
│  │          Weight [vocab, hidden]         │               │
│  │  ┌─────────────────┬─────────────────┐  │               │
│  │  │   GPU 0         │   GPU 1         │  │               │
│  │  │ [0:V/2, :]      │ [V/2:V, :]      │  │               │
│  │  └─────────────────┴─────────────────┘  │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
│  计算流程:                                                  │
│  x @ W0^T → output_0 (partial)                             │
│  x @ W1^T → output_1 (partial)                             │
│  All-Gather → [output_0, output_1] = full output           │
│                                                             │
│  Row Parallel (如 MLP gate_proj):                           │
│  ┌─────────────────────────────────────────┐               │
│  │          Weight [inter, hidden]         │               │
│  │  ┌─────────────────┬─────────────────┐  │               │
│  │  │   GPU 0         │   GPU 1         │  │               │
│  │  │ [:I/2, 0:H/2]   │ [:I/2, H/2:H]   │  │               │
│  │  │                 │                 │  │               │
│  │  │ [I/2:I, 0:H/2]  │ [I/2:I, H/2:H]  │  │               │
│  │  └─────────────────┴─────────────────┘  │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
│  计算流程:                                                  │
│  x_0 @ W0 → output_0 (partial)                             │
│  x_1 @ W1 → output_1 (partial)                             │
│  All-Reduce → output_0 + output_1 = full output            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 代码阅读：其他层

### 5.1 layers/layernorm.py - RMSNorm

```python
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    与 LayerNorm 的区别:
    - 不需要计算均值
    - 不需要减去均值
    - 计算更简单，速度更快
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm 计算:

        1. 计算 RMS: sqrt(mean(x^2) + eps)
        2. 归一化: x / RMS
        3. 缩放: x * weight
        """
        # 计算 RMS
        variance = x.pow(2).mean(-1, keepdim=True)
        rms = torch.sqrt(variance + self.eps)

        # 归一化并缩放
        output = (x / rms) * self.weight

        return output

    @torch.compile
    def forward_compiled(self, x: torch.Tensor) -> torch.Tensor:
        """使用 torch.compile 优化"""
        return self.forward(x)
```

### 5.2 layers/rotary_embedding.py - RoPE

```python
class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)

    原理:
    - 将位置信息编码为旋转矩阵
    - 应用到 Q 和 K 上
    - 相对位置编码，泛化性好
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 1000000.0,
    ):
        super().__init__()

        # 计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # 预计算 cos 和 sin
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        """预计算位置编码"""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)

        # [seq_len, dim/2]
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用旋转位置编码

        Args:
            q: [batch, seq, heads, head_dim]
            k: [batch, seq, kv_heads, head_dim]
            position_ids: [batch, seq]

        Returns:
            旋转后的 q, k
        """
        # 获取对应位置的 cos, sin
        cos = self.cos_cached[position_ids]  # [batch, seq, dim/2]
        sin = self.sin_cached[position_ids]

        # 扩展维度以匹配 heads
        cos = cos.unsqueeze(2)  # [batch, seq, 1, dim/2]
        sin = sin.unsqueeze(2)

        # 应用旋转
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    旋转操作

    将 x 分为两半，交换并取负
    [x1, x2] -> [-x2, x1]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

### 5.3 layers/sampler.py - 采样器

```python
class Sampler(nn.Module):
    """
    Token 采样器

    支持:
    - Greedy (贪婪)
    - Temperature (温度)
    - Top-k
    - Top-p (Nucleus)
    """

    def forward(
        self,
        logits: torch.Tensor,  # [batch, vocab_size]
        sampling_params: SamplingParams,
    ) -> torch.Tensor:
        """
        从 logits 采样下一个 token
        """
        # 应用温度
        if sampling_params.temperature > 0:
            logits = logits / sampling_params.temperature

        # Top-k 过滤
        if sampling_params.top_k > 0:
            top_k = min(sampling_params.top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            min_value = values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_value,
                torch.full_like(logits, float('-inf')),
                logits,
            )

        # Top-p 过滤
        if sampling_params.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 移除累积概率超过 top_p 的 token
            sorted_indices_to_remove = cum_probs > sampling_params.top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # 采样
        if sampling_params.temperature == 0:
            # Greedy
            return torch.argmax(logits, dim=-1)
        else:
            # 随机采样
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

---

## 6. 代码实践

### 6.1 分析模型参数

```python
from models.qwen3 import Qwen3

model = Qwen3(config)

# 统计参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"总参数: {total_params / 1e6:.2f}M")
print(f"可训练参数: {trainable_params / 1e6:.2f}M")

# 分析各层参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, {param.numel() / 1e6:.4f}M")
```

### 6.2 观察 RoPE 效果

```python
import torch
import matplotlib.pyplot as plt
from layers.rotary_embedding import RotaryEmbedding

rope = RotaryEmbedding(dim=64, max_position_embeddings=128)

# 比较不同位置的编码相似度
positions = torch.arange(128)
cos = rope.cos_cached[position_ids]

# 计算位置间的相似度
similarity = torch.mm(cos, cos.T)

# 可视化
plt.imshow(similarity.numpy(), cmap='viridis')
plt.colorbar()
plt.title('RoPE Position Similarity')
plt.xlabel('Position')
plt.ylabel('Position')
plt.show()
```

### 6.3 实践练习

**练习 1**: 实现 GQA 的 K/V 复制

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将 K/V 复制以匹配 Q 的头数

    Args:
        hidden_states: [batch, seq, kv_heads, head_dim]
        n_rep: 复制次数 (num_heads // num_kv_heads)

    Returns:
        [batch, seq, num_heads, head_dim]
    """
    if n_rep == 1:
        return hidden_states

    batch, seq, kv_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :]
    hidden_states = hidden_states.expand(batch, seq, kv_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, seq, kv_heads * n_rep, head_dim)
```

**练习 2**: 实现简单的注意力计算

```python
def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    缩放点积注意力 (无 FlashAttention 版本)

    用于理解注意力计算原理
    """
    d_k = q.size(-1)

    # 计算注意力分数
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # 应用 mask (因果注意力)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax
    attn_weights = F.softmax(scores, dim=-1)

    # 加权求和
    output = torch.matmul(attn_weights, v)

    return output
```

---

## 7. 性能考量

### 7.1 GQA 的优势

```
┌─────────────────────────────────────────────────────────────┐
│                    GQA 优势分析                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Qwen3-0.6B 配置:                                           │
│  - num_attention_heads = 14                                 │
│  - num_key_value_heads = 2                                  │
│  - KV Cache 节省: 14/2 = 7x                                 │
│                                                             │
│  内存占用对比 (seq_len=4096, head_dim=64, layers=24):        │
│                                                             │
│  MHA:                                                       │
│  4096 × 14 × 64 × 2 (K+V) × 24 × 2 bytes = 343 MB          │
│                                                             │
│  GQA:                                                       │
│  4096 × 2 × 64 × 2 × 24 × 2 bytes = 49 MB                  │
│                                                             │
│  节省: 294 MB (85%)                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 RoPE vs Learnable Position Embedding

```
┌─────────────────────────────────────────────────────────────┐
│               位置编码对比                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Learnable Position Embedding:                              │
│  - 需要学习参数                                             │
│  - 外推能力差 (超过训练长度效果下降)                         │
│  - 额外内存占用                                             │
│                                                             │
│  RoPE:                                                      │
│  - 无需学习参数                                             │
│  - 相对位置编码，外推能力强                                 │
│  - 通过旋转实现，计算高效                                   │
│  - 长序列表现更好                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 关键问题解答

### Q1: 为什么使用 RMSNorm 而不是 LayerNorm？

```
LayerNorm:
  y = (x - mean) / sqrt(var + eps) * weight + bias

RMSNorm:
  y = x / sqrt(mean(x^2) + eps) * weight

RMSNorm 优势:
1. 计算更简单 (不需要均值和方差)
2. 不需要 bias 参数
3. 在 LLM 中效果相当或更好
4. 计算速度更快
```

### Q2: FlashAttention 如何优化内存访问？

```
传统注意力:
1. 计算 Q @ K^T → [batch, heads, seq, seq]
2. Softmax
3. @ V → [batch, heads, seq, head_dim]

问题: 中间结果 [seq, seq] 占用大量内存

FlashAttention:
1. 分块计算 (tiling)
2. 不保存完整注意力矩阵
3. 在 SRAM 中完成 Softmax
4. 只保存最终输出

内存: O(seq^2) → O(seq)
速度: 减少 HBM 访问，更快
```

### Q3: SwiGLU 相比普通 MLP 有什么优势？

```
普通 MLP:
  output = down(activation(up(x)))

SwiGLU:
  output = down(silu(gate(x)) * up(x))

优势:
1. 门控机制提供更好的表达能力
2. SiLU 激活平滑且非单调
3. 在 LLM 中效果更好
4. 参数量增加，但性能提升明显
```

---

## 9. 知识检查点

### 基础问题

1. ❓ Qwen3 模型的主要组件有哪些？

2. ❓ GQA 是什么？它如何减少内存占用？

3. ❓ RMSNorm 和 LayerNorm 有什么区别？

### 进阶问题

4. ❓ RoPE 如何实现相对位置编码？

5. ❓ FlashAttention varlen 和 with_kvcache 有什么区别？

6. ❓ Tensor Parallel 的 Column 和 Row 切分有什么不同？

---

## 10. 代码走查清单

阅读模型相关文件时，关注：

**models/qwen3.py**:
- [ ] `Qwen3.__init__`: 模型组件初始化
- [ ] `Qwen3.forward`: 整体前向传播流程
- [ ] `Qwen3DecoderLayer`: 单层结构

**layers/attention.py**:
- [ ] `Qwen3Attention`: QKV 投影和注意力计算
- [ ] `_attention_prefill`: varlen 调用
- [ ] `_attention_decode`: with_kvcache 调用

**layers/linear.py**:
- [ ] `ColumnParallelLinear`: 列切分
- [ ] `RowParallelLinear`: 行切分和 All-Reduce

**layers/rotary_embedding.py**:
- [ ] `RotaryEmbedding`: 频率计算和旋转应用

**layers/sampler.py**:
- [ ] `Sampler`: 各种采样策略的实现

---

## 11. 扩展阅读

### 推荐论文

1. **RoFormer: Enhanced Transformer with Rotary Position Embedding**
   - RoPE 的原始论文

2. **FlashAttention: Fast and Memory-Efficient Exact Attention**
   - FlashAttention 的原理

3. **GQA: Training Generalized Multi-Query Transformer Models**
   - Grouped Query Attention 论文

4. **GLU Variants Improve Transformer**
   - SwiGLU 的来源

### 相关代码

| 文件 | 核心类/函数 |
|------|------------|
| `models/qwen3.py` | `Qwen3`, `Qwen3DecoderLayer`, `Qwen3MLP` |
| `layers/attention.py` | `Qwen3Attention` |
| `layers/linear.py` | `ColumnParallelLinear`, `RowParallelLinear` |
| `layers/layernorm.py` | `RMSNorm` |
| `layers/rotary_embedding.py` | `RotaryEmbedding` |
| `layers/sampler.py` | `Sampler` |

---

## 12. 下一步

完成 Day 5 后，你应该能够：
- ✅ 解释 Qwen3 模型的整体架构
- ✅ 理解各层的实现细节
- ✅ 描述 GQA 的工作原理
- ✅ 理解 RoPE 位置编码

**准备 Day 6**: 工具函数与进阶特性
- 深入理解上下文管理机制
- 学习模型加载器

---

*预计学习时间: 3-4 小时*
