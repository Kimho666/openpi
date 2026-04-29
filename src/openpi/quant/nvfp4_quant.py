"""
NVFP4 pseudo-quantization for VLA models.

NVFP4 (E2M1) format:
  - 4-bit values: 1 sign + 2 exponent + 1 mantissa bits
  - Representable positive values: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
  - Per-16-element block E4M3 FP8 scale  (Level 2, fine-grained)
  - Per-tensor FP32 scale                (Level 1, global normalization)

Reference: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── E2M1 FP4 lookup tables (positive values and their round-to-nearest boundaries) ──
# Positive representable values: 0.0 0.5 1.0 1.5 2.0 3.0 4.0 6.0
_FP4_POS_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# Midpoints between consecutive values → bucketize boundaries
_FP4_BOUNDARIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

NVFP4_BLOCK_SIZE = 16   # elements per micro-block
NVFP4_MAX        = 6.0  # max |value| representable in E2M1


# ─────────────────────────────────────────────────────────────────────────────
# Core quantization primitives (fully vectorized, no Python loops)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _round_to_fp4(x: torch.Tensor) -> torch.Tensor:
    """Vectorized round-to-nearest E2M1 FP4.  Input must be pre-scaled to [-6, 6]."""
    vals   = _FP4_POS_VALUES.to(device=x.device, dtype=x.dtype)
    bounds = _FP4_BOUNDARIES.to(device=x.device, dtype=x.dtype)
    sign   = x.sign()
    abs_x  = x.abs().clamp(max=NVFP4_MAX)
    # bucketize returns the first boundary index that is > abs_x → maps to nearest FP4
    return vals[torch.bucketize(abs_x, bounds)] * sign


@torch.no_grad()
def _round_to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round to E4M3 FP8 precision via cast (requires PyTorch >= 2.1)."""
    try:
        return x.to(torch.float8_e4m3fn).to(x.dtype)
    except (AttributeError, RuntimeError):
        return x  # graceful fallback: no-op when float8 is unavailable


# ─────────────────────────────────────────────────────────────────────────────
# Weight quantization
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def quantize_weight_nvfp4(w: torch.Tensor) -> torch.Tensor:
    """
    NVFP4 two-level weight pseudo-quantization.

    w : (out_features, in_features)
    Blocks of 16 are formed along the in_features dimension.
    """
    orig_dtype = w.dtype
    out_f, in_f = w.shape
    w = w.float()

    # ── Level 1: per-tensor FP32 normalization ───────────────────────────────
    tensor_scale = w.abs().max().clamp(min=1e-10)
    w_norm = w / tensor_scale                          # values roughly in [-1, 1]

    # ── Pad in_features to a multiple of NVFP4_BLOCK_SIZE ────────────────────
    pad = (NVFP4_BLOCK_SIZE - in_f % NVFP4_BLOCK_SIZE) % NVFP4_BLOCK_SIZE
    if pad:
        w_norm = F.pad(w_norm, (0, pad))               # (out_f, in_f + pad)
    # (out_f, num_blocks, 16)
    w_blocks = w_norm.reshape(out_f, -1, NVFP4_BLOCK_SIZE)

    # ── Level 2: per-block E4M3 FP8 scale ────────────────────────────────────
    block_amax    = w_blocks.abs().amax(dim=-1, keepdim=True)   # (out_f, nb, 1)
    # clamp to e4m3 min-subnormal so scale survives the FP8 cast
    block_scale   = (block_amax / NVFP4_MAX).clamp(min=2e-3)
    block_scale_e4m3 = _round_to_e4m3(block_scale)

    # ── Quantize (FP4) → dequantize ──────────────────────────────────────────
    w_q  = _round_to_fp4(w_blocks / block_scale_e4m3)
    w_dq = (w_q * block_scale_e4m3).reshape(out_f, -1)  # (out_f, in_f + pad)

    # ── Strip padding, restore global scale and original dtype ───────────────
    if pad:
        w_dq = w_dq[:, :in_f]
    return (w_dq * tensor_scale).to(orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Activation quantization
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def quantize_activation_nvfp4(x: torch.Tensor) -> torch.Tensor:
    """
    NVFP4 two-level activation pseudo-quantization.

    x : (..., hidden_dim)
    Blocks of 16 are formed along the hidden_dim dimension.
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    hidden_dim = orig_shape[-1]
    x = x.float()

    # ── Level 1: per-tensor FP32 normalization ───────────────────────────────
    tensor_scale = x.abs().max().clamp(min=1e-10)
    x_norm = x / tensor_scale

    # ── Flatten to 2D and pad ─────────────────────────────────────────────────
    x_2d  = x_norm.reshape(-1, hidden_dim)          # (batch_tokens, hidden_dim)
    batch = x_2d.shape[0]
    pad   = (NVFP4_BLOCK_SIZE - hidden_dim % NVFP4_BLOCK_SIZE) % NVFP4_BLOCK_SIZE
    if pad:
        x_2d = F.pad(x_2d, (0, pad))
    # (batch_tokens, num_blocks, 16)
    x_blocks = x_2d.reshape(batch, -1, NVFP4_BLOCK_SIZE)

    # ── Level 2: per-block E4M3 FP8 scale ────────────────────────────────────
    block_amax       = x_blocks.abs().amax(dim=-1, keepdim=True)
    block_scale      = (block_amax / NVFP4_MAX).clamp(min=2e-3)
    block_scale_e4m3 = _round_to_e4m3(block_scale)

    # ── Quantize (FP4) → dequantize ──────────────────────────────────────────
    x_q  = _round_to_fp4(x_blocks / block_scale_e4m3)
    x_dq = (x_q * block_scale_e4m3).reshape(batch, -1)

    # ── Strip padding, restore global scale, shape, and dtype ────────────────
    if pad:
        x_dq = x_dq[:, :hidden_dim]
    return (x_dq.reshape(orig_shape) * tensor_scale).to(orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Drop-in replacement module
# ─────────────────────────────────────────────────────────────────────────────

class NVFP4Linear(nn.Module):
    """Linear layer with NVFP4 weight pseudo-quantization and optional activation quantization."""

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, act_quant: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.act_quant    = act_quant

        self.register_buffer(
            "weight",
            torch.zeros(out_features, in_features, dtype=torch.float16, requires_grad=False),
        )
        self.register_buffer(
            "bias",
            torch.zeros(1, out_features, dtype=torch.float16, requires_grad=False)
            if bias else None,
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_quant:
            x = quantize_activation_nvfp4(x)
        return F.linear(x, self.weight, self.bias)

    @staticmethod
    def from_float(module: nn.Linear, act_quant: bool = True) -> "NVFP4Linear":
        assert isinstance(module, nn.Linear)
        new_module = NVFP4Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            act_quant=act_quant,
        )
        new_module.weight = quantize_weight_nvfp4(module.weight)
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self) -> str:
        return (
            f"NVFP4Linear({self.in_features}, {self.out_features}, "
            f"bias={self.bias is not None}, act_quant={self.act_quant})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model-level entry point (mirrors fake_quant.quantize_vla_like)
# ─────────────────────────────────────────────────────────────────────────────

def quantize_vla_like_nvfp4(model: nn.Module, act_quant: bool = True) -> nn.Module:
    """
    Replace GemmaMLP / GemmaAttention linear projections with NVFP4Linear.

    Mirrors the structure of fake_quant.quantize_vla_like().
    Called from vlaquant_demo.py as:
        model_nvfp4 = quantize_vla_like_nvfp4(policy._model, act_quant=True)
    """
    from transformers.models.gemma.modeling_gemma import GemmaAttention, GemmaMLP

    for _, m in model.paligemma_with_expert.paligemma.model.language_model.named_modules():
        if isinstance(m, GemmaMLP):
            m.gate_proj = NVFP4Linear.from_float(m.gate_proj, act_quant=act_quant)
            m.up_proj   = NVFP4Linear.from_float(m.up_proj,   act_quant=act_quant)
            m.down_proj = NVFP4Linear.from_float(m.down_proj, act_quant=act_quant)
        elif isinstance(m, GemmaAttention):
            m.q_proj = NVFP4Linear.from_float(m.q_proj, act_quant=act_quant)
            m.k_proj = NVFP4Linear.from_float(m.k_proj, act_quant=act_quant)
            m.v_proj = NVFP4Linear.from_float(m.v_proj, act_quant=act_quant)
            m.o_proj = NVFP4Linear.from_float(m.o_proj, act_quant=act_quant)

    return model
