import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
    """
    将 Python 代码编译成高效的内核
    融合多个操作，减少内存访问
    自动优化计算图
    """
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # logits:(batch_size, vocab_size) temperature:(batch_size, 1) 广播
        # logits / temperature => 温度缩放，温度越大，输出的随机性越大（抹平了差异）
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        """
        用随机噪声 + 确定性 argmax = 随机采样
        # 步骤 1: 生成指数分布噪声
        noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)

        # 步骤 2: 概率除以噪声
        scaled = probs.div_(noise)

        # 步骤 3: 取最大值的索引
        sample_tokens = scaled.argmax(dim=-1)

        假设 probs = [0.7, 0.2, 0.1]，三个 token
        传统采样：
        70% 概率选 token 0
        20% 概率选 token 1
        10% 概率选 token 2

        Gumbel-Max 采样：
        生成噪声: noise = [0.5, 2.0, 0.1]  (指数分布随机)
        
        scaled = probs / noise
                = [0.7/0.5, 0.2/2.0, 0.1/0.1]
                = [1.4, 0.1, 1.0]
        
        argmax → 选 token 0（值为 1.4 最大）
        """
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
