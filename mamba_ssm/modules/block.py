# Copyright (c) 2024, Tri Dao, Albert Gu.
from typing import Optional

import torch
from torch import nn, Tensor

from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn


class Block(nn.Module):
    def __init__(
        """
        dim: 模型基准维度d_model，是所有模块的输入输出维度;
        mixer_cls：混合器类，是Block的核心特征提取模块，由create_block根据层索引动态选择（Mamba/注意力）
        mlp_cls: MLP类，是Block的辅助特征提取模块，由create_block根据层索引动态选择（前馈网络/恒等映射）
        norm_cls：归一化类，create_block由rms_norm参数控制，RMSNorm比LayerNorm更高效
        fused_add_norm：是否启用Add+Norm融合操作（Triton内核加速）
        residual_in_fp32: 是与否将残差张量以float32存储
        """
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim) # 混合器前的归一化层
        self.mixer = mixer_cls(dim) # 实例化混合器（Mamba/MHA）
        # Block不关心mixer_cls/mlp_cls的具体实现细节，只要求其满足输入输出维度为dim
        # 可选MLP分支：有MLP则新增norm2，无则置为None
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        # 融合操作的断言检查：确保依赖正确
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs
    ): # hidden_states: (B, L, D),上一层Block的加工后输出
    # residual: (B, L, D),上一层Block的残差输出
    # inference_params: 推理参数
    # mixer_kwargs: 传递给混合器的其他关键字参数，如Mamba的dt_rank等
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states # 残差加法，无历史残差则直接用当前的hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype)) # 将残差张量转为归一化层的精度（避免精度不匹配)
            if self.residual_in_fp32: # 残差精度优化
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        # 混合器加工：归一化的张量送入Mamba/MHA，得到新的hidden_states
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if self.mlp is not None:
            if not self.fused_add_norm:
                # 残差加法：混合器结果+上一步的residual
                residual = hidden_states + residual
                # 用norm2做二次归一化
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            # MLP加工，得到最终的hidden_states
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    “”“
    非融合模式核心流程（有 MLP）上一层输出 → 残差加法 → norm → Mamba/MHA → 残差加法 → norm2 → GatedMLP → 输出（新 hidden_states + 更新后 residual）
    融合模式：是性能优化版
    ”“”
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

# MambaLMHeadModel → 包含MixerModel（骨干） → MixerModel堆叠N个Block → 每个Block由create_block动态创建 → Block封装Mamba/MHA+MLP+Norm
“”“
create_block：根据配置（Mamba1/Mamba2 / 注意力、是否启用 MLP），生成 Block 的组件类（mixer_cls/mlp_cls/norm_cls），实例化 Block；
MixerModel：将 N 个 Block 堆叠成nn.ModuleList，初始化 Token Embedding 和最终归一化层，负责跨 Block 的残差传递和缓存管理；
MambaLMHeadModel：在 MixerModel 基础上添加 LM Head，实现从特征到 Token 概率的映射，支持文本生成。
Mamba 的所有序列特征加工，最终都落到每个 Block 的 forward 方法中，Block 是 Mamba 的「特征加工原子」。

残差：除了第一个block以外，每个block的残差=上一个Block的输出特征+上一个Block传递的历史残差，解决深层模型梯度消失问题
”“”