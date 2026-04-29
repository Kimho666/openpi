import torch
from torch import nn
from torch.nn import functional as F
from functools import partial


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_weight_per_group_absmax(w, n_bits=8, group_size=128):
    # w: (out_features, in_features)
    q_max = 2 ** (n_bits - 1) - 1
    in_features = w.shape[-1]
    for start in range(0, in_features, group_size):
        end = min(start + group_size, in_features)
        # 直接对原张量切片操作，不使用中间变量！！！
        scales = w[:, start:end].abs().max(dim=-1, keepdim=True)[0]
        scales.clamp_(min=1e-5).div_(q_max)
        # 原地修改原权重，必须这样写才生效
        w[:, start:end].div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_group_absmax(t, n_bits=8, group_size=128):
    # t: (..., hidden_dim), group along hidden_dim
    t_shape = t.shape
    # 展平为 2D 做分组量化
    t_2d = t.reshape(-1, t_shape[-1])
    q_max = 2 ** (n_bits - 1) - 1
    hidden_dim = t_2d.shape[-1]
    
    for start in range(0, hidden_dim, group_size):
        end = min(start + group_size, hidden_dim)
        # 直接原地修改，不使用中间变量
        scales = t_2d[:, start:end].abs().max(dim=-1, keepdim=True)[0]
        scales.clamp_(min=1e-5).div_(q_max)
        t_2d[:, start:end].div_(scales).round_().mul_(scales)
    
    # ✅ 关键修复：必须把 2D 还原回原始形状！！！
    t = t_2d.view(t_shape)
    return t

class W8A8Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
        n_bits=8,
        group_size=128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.group_size = group_size

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=self.n_bits)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=self.n_bits)
        elif act_quant == "per_group":
            self.act_quant_name = "per_group"
            self.act_quant = partial(
                quantize_activation_per_group_absmax,
                n_bits=self.n_bits,
                group_size=self.group_size,
            )
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_output=False,
        n_bits=8,
        group_size=128,
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            n_bits=n_bits,
            group_size=group_size,
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=n_bits
            )  # use 8-bit integer for weight
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=n_bits
            )
        elif weight_quant == "per_group":
            new_module.weight = quantize_weight_per_group_absmax(
                module.weight, n_bits=n_bits, group_size=group_size
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"


def quantize_vla_like(
    model,
    weight_quant="per_group",
    act_quant="per_group",
    quantize_bmm_input=False,
    n_bits=8,
    group_size=128,
):
    from transformers.models.gemma.modeling_gemma import (
        GemmaAttention,
        GemmaMLP,
    )

    for name, m in model.paligemma_with_expert.paligemma.model.language_model.named_modules():
        if isinstance(m, (GemmaMLP)):
            m.gate_proj = W8A8Linear.from_float(
                m.gate_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                n_bits=n_bits,
                group_size=group_size,
            )
            m.up_proj = W8A8Linear.from_float(
                m.up_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                n_bits=n_bits,
                group_size=group_size,
            )
            m.down_proj = W8A8Linear.from_float(
                m.down_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                n_bits=n_bits,
                group_size=group_size,
            )
        elif isinstance(m, (GemmaAttention)):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                n_bits=n_bits,
                group_size=group_size,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                n_bits=n_bits,
                group_size=group_size,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                n_bits=n_bits,
                group_size=group_size,
            )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                n_bits=n_bits,
                group_size=group_size,
            )
    return model
