# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    def __init__(
        self,
        d_model, # 模型特征维度/每个token的特征向量分量数/通道数/mamba模块的输入输出维度，维度越大每个Token能表达的信息越丰富，通常取512、768、1024等
        d_state=16, # SSM内部状态维度h_t，控制SSM的记忆容量和表达能力，较大的d_state能捕捉更复杂的序列模式，但计算开销也更大，通常取16、32等
        d_conv=4, # 卷积核大小（控制因果卷积的滑动窗口大小），控制局部特征提取的范围
        expand=2, # 特征维度扩展因子，self.d_inner = int(self.expand * self.d_model)
        dt_rank="auto", # dt的低秩表示维度，默认是d_model/16，由输入特征动态生成，先生成低维dt表示，再通过线性映射恢复到d_inner维度
        dt_min=0.001, # dt的最小值，越小状态更新越慢
        dt_max=0.1, # dt的最大值，越大状态更新越慢
        dt_init="random",
        dt_scale=1.0, # 控制dt初始化的波动范围
        dt_init_floor=1e-4, # dt初始化下限，防止dt无限接近0，避免模型一开始就完全遗忘
        ### 选择性SSM核心：输入不同，dt不同，模型的记忆/遗忘策略不同，dt = Softplus(Linear(x)+bias)
        conv_bias=True, # 卷积层是否加偏置向量，如果加会使卷积对序列特征的局部提取更灵活
        bias=False, # 线性层是否加偏置 Mamba 的内部特征维度已经通过expand扩大，且有卷积、SSM 的非线性变换，不加偏置也能保证拟合能力，同时减少参数数量，提升训练 / 推理速度；这是作者实验后的最优选择。
        use_fast_path=True,  # Fused kernel options 调用mamba_inner_fn—— 这是一个融合了「卷积 + SSM + 投影」的 CUDA 内核，把多个步骤合并成一个，大幅提升训练 / 推理速度；
        layer_idx=None, # 用于多层mamba堆叠时，在inference cache中区分不同层的状态
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model # 输入/输出维度
        self.d_state = d_state # SSM内部状态维度（B、C矩阵）
        self.d_conv = d_conv 
        self.expand = expand # 扩展因子，d_inner = expand×d_model，让中间层有更多参数学习特征；
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank # dt 的低维表示维度（默认是 d_model/16），平衡计算量和表达能力。
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        # 输入维度映射为扩展维度，拆成两个分支：特征提取和门控（筛选有用特征）
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, # 卷积的输入输出维度都是d_inner=expand×d_model
            out_channels=self.d_inner, # 卷积的输入输出维度都是d_inner=expand×d_model
            bias=conv_bias, # 卷积层是否加偏置
            kernel_size=d_conv, # 卷积核大小
            groups=self.d_inner, # 深度可分离卷积，每个通道独立卷积
            padding=d_conv - 1, # 因为是因果卷积，padding放在左侧，保证输出长度和输入长度一致
            **factory_kwargs,
        ) # 深度可分离卷积，逐通道卷积提取局部特征

        self.activation = "silu" # 非线性激活函数
        self.act = nn.SiLU() 

        self.x_proj = nn.Linear( # 将卷积输出映射为dt，B，C三个部分，dt是状态更新速度，B是输入->状态投影，C是状态->输出投影，用于后续的SSM计算
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)  # dt的线性映射，dt_rank维度的低秩表示映射到d_inner维度

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization SSM基础衰减矩阵A
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous() # 得到(d_inner, d_state)形状的A矩阵，每行是1到d_state的整数序列
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log) # 注册为可训练的模型参数
        self.A_log._no_weight_decay = True

        # D "skip" parameter 注册为形状d_inner的可训练参数，表示SSM的直接跳跃连接权重
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # 线性层，将扩展后的中间维度d_inner投影回原始模型维度d_model
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None): # inference_params: 表示是否为推理模式
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        训练模式：inference_params=None（默认），处理整段序列，无状态缓存，追求批量计算效率；
        推理模式：inference_params≠None（在线部署时用），单步处理序列，缓存卷积状态conv_state和 SSM 状态ssm_state，避免重复计算，实现低延迟实时决策（对应在线部署场景）。
        """
        batch, seqlen, dim = hidden_states.shape 

        conv_state, ssm_state = None, None # conv_state: 卷积缓存状态，形状 (B, d_inner, d_conv),存储前 d_conv 个 Token 的 x 分支特征；ssm_state: SSM缓存状态，形状 (B, d_inner, d_state),存储上一步的 h_{t-1}；
        if inference_params is not None:# 推理时的状态缓存容器（自定义类，包含 seqlen_offset、conv_state、ssm_state 等）
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch) # 推理时取出卷积缓存状态和SSM缓存状态
            if inference_params.seqlen_offset > 0: # 非首次推理调用
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state) # 单步处理当前时间步特征（而非整段序列），实现「实时感知 - 单步决策」
                # step函数会原地更新conv_state和ssm_state，下一次推理直接复用更新后的状态，无需重新计算历史，大幅降低推理延迟（Mamba 在线部署高效的关键）
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state) 还原为初始化的正数值矩阵，取负数后得到负数矩阵。A 的形状始终为 (d_inner, d_state)，每个通道对应一个 d_state 维的衰减向量，控制该通道的基础遗忘速度
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        # 开启快速计算条件： 三个条件同时满足 ——use_fast_path=True（开启快速路径）、存在融合卷积函数causal_conv1d_fn、训练模式（inference_params=None）；
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            ) # 内部会自动完成「卷积→x_proj→dt 计算→SSM 扫描→门控融合→输出投影」的全流程
        else: # 推理模式下分步执行
            x, z = xz.chunk(2, dim=1) #维度从1开始
            # Compute short convolution
            if conv_state is not None: 
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W) # 对当前 x（推理时为单步，L=1）做左侧零填充，保证填充后长度为 d_conv，避免序列长度不足的报错；
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen]) 
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
            # 卷积后 x 的形状仍为 **(B, d_inner, L)**，维度不变，仅特征被加工为包含短程依赖的表示。
            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d) 将 (B, d_inner, L)→(B×L, d_inner)，合并批次和时间步，特征维度 d_inner 作为最后一维（适配线性层x_proj的输入要求）
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1) # 输出维度是**(B×L, dt_rank+2d_state)**
            dt = self.dt_proj.weight @ dt.t() # dt_low 转置为 (dt_rank, B×L)，与 dt_proj 权重 (d_inner, dt_rank) 相乘，得到 (d_inner, B×L)；
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen) # 将 (d_inner, B×L)→(B, d_inner, L)，得到最终的动态 dt 张量，形状 **(B, d_inner, L)**—— 每个批次、每个通道、每个时间步都有专属的 dt 值，完美实现「数据依赖的动态遗忘速度」
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None, # 推理模式（ssm_state≠None）设为 True，返回「加工后特征 y + 最后一个 SSM 状态 last_state」；训练模式设为 False，仅返回 y；
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y) # 将 d_inner→d_model，输出out形状为 **(B, L, D)，与输入hidden_states形状完全一致 **；
        return out

    def step(self, hidden_states, conv_state, ssm_state): # 推理模式，用于在线实时决策
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D) # squeeze(B,1,D)→(B,D)，去掉长度为 1 的序列维度（L=1 无意义);in_proj将 D→2d_inner，输出xz形状为 **(B, 2d_inner)**
        x, z = xz.chunk(2, dim=-1)  # (B D) (B, d_inner)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W) 挤走旧特征，为新特征腾出位置
            conv_state[:, :, -1] = x # 将当前单 Token 的 x 分支特征（B, d_inner）写入 conv_state 的最后一维（空出的位置），完成卷积窗口的更新；
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D) 将卷积权重从 (d_inner,1,d_conv)→(d_inner,d_conv)，去掉无意义的维度 1；卷积状态与权重逐元素相乘（B, d_inner, d_conv）×（d_inner, d_conv）=（B, d_inner, d_conv）屏；沿卷积窗口维度求和，得到单步卷积结果，形状为 (B, d_inner)（对应卷积的「加权求和」核心逻辑）
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype) # 得到加工后的x特征
        else: # 将「状态滚动 + 写入新特征 + 卷积计算 + 激活」合并为一步
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # 生成（d_inner, d_state）的负数衰减矩阵，保证 SSM 状态自然衰减

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype)) # F.softplus激活：将 dt 映射为正实数（B, d_inner），满足「动态时间步必须为正」的核心要求
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A)) # (B d_inner d_state) 计算离散化的状态转移矩阵 dA（对应 h_t = dA * h_{t-1} + dB * x_t 中的 dA）
            dB = torch.einsum("bd,bn->bdn", dt, B) # (B d_inner d_state) 计算离散化的输入投影矩阵 dB（对应 h_t = dA * h_{t-1} + dB * x_t 中的 dB）
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB) # 更新 SSM 状态 h_t = dA * h_{t-1} + dB * x_t，ssm_state * dA：逐元素相乘
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y) # self.out_proj(y)将 y（B, d_inner）投影为（B, D）（D=d_model），还原模型基准维度
        return out.unsqueeze(1), conv_state, ssm_state # 将（B,D）→（B,1,D），恢复序列长度维度（L=1），与输入形状一致

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ) # 卷积状态缓存的初始化：初始值为全 0：推理开始前没有任何历史 Token，卷积窗口的初始状态为 0，第一次调用step时，第一个 Token 的 x 特征会覆盖最后一维的位置；
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            ) # SSM状态缓存的初始化：初始值为全 0：推理开始前没有任何历史 Token，SSM 的记忆状态初始为 0，第一次调用 step 时，SSM 会根据第一个 Token 的 x 特征和动态 dt 更新状态；
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
