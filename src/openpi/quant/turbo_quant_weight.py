"""
TurboQuant-based weight quantization for VLM linear layers.

Analog of fake_quant.py but using TurboQuant's rotation + Lloyd-Max quantization
instead of per-channel absmax.  This implements a "fake quantization" approach
(stores reconstructed fp16 weights) so it can be compared directly with VLaQuant.

All TurboQuant primitives (rotation matrix generation, Lloyd-Max solver, codebook)
are inlined here — no imports from the turboquant-pytorch package are used.

Design differences vs W8A8Linear (fake_quant.py):
  - Weights:     TurboQuant rotation + Lloyd-Max (per-row, MSE-optimal, variable bits)
                 rather than per-channel absmax INT8
  - Activations: configurable (`per_token` / `per_tensor` absmax, or
                 `turbo_per_token` for TurboQuant activation fake-quant)
  - Bit-width:   configurable (default 4-bit; 8-bit gives near-lossless quality)

Algorithm for weight quantization (one row = one vector):
  1. Normalize each weight row to the unit sphere; store L2 norms.
  2. Apply a random orthogonal rotation Pi so coordinates become ~iid N(0, 1/d).
  3. Find the nearest Lloyd-Max centroid for every coordinate.
  4. Dequantize: map indices back to centroid values.
  5. Undo rotation and rescale by saved norms → fake-quantized weight.

Both the rotation matrices and the Lloyd-Max codebooks are cached globally so
they are only computed once per (dimension, bits) pair across all layers.
"""

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import integrate


# ─────────────────────────────────────────────────────────────────────────────
# TurboQuant primitives (inlined from turboquant-pytorch/turboquant.py)
# ─────────────────────────────────────────────────────────────────────────────

def generate_rotation_matrix(
    d: int, seed: Optional[int] = None, device: str = "cpu"
) -> torch.Tensor:
    """
    Generate a Haar-distributed random orthogonal rotation matrix via QR
    decomposition of a Gaussian matrix.  Used in TurboQuant to decorrelate
    vector coordinates before per-coordinate Lloyd-Max quantization.
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    # Fix sign ambiguity in QR so that det(Q) = +1
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Lloyd-Max solver (inlined from turboquant-pytorch/lloyd_max.py)
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_approx_pdf(x: float, d: int) -> float:
    """Gaussian approximation N(0, 1/d) — accurate for d >= 64."""
    sigma2 = 1.0 / d
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))


def _beta_pdf(x: float, d: int) -> float:
    """Exact PDF of one coordinate after rotating a d-dim unit vector."""
    if abs(x) >= 1.0:
        return 0.0
    coeff = math.gamma(d / 2) / (math.sqrt(math.pi) * math.gamma((d - 1) / 2))
    return coeff * (1 - x * x) ** ((d - 3) / 2)


def solve_lloyd_max(
    d: int,
    bits: int,
    use_exact: bool = False,
    max_iter: int = 200,
    tol: float = 1e-10,
):
    """
    Solve the Lloyd-Max optimal scalar quantizer for the coordinate distribution
    that arises after rotating a d-dimensional unit vector by a random orthogonal
    matrix.

    Returns:
        centroids  : torch.Tensor of shape (2^bits,)
        boundaries : torch.Tensor of shape (2^bits - 1,)
    """
    n_levels = 2 ** bits
    pdf = (_beta_pdf if use_exact else _gaussian_approx_pdf)
    sigma = 1.0 / math.sqrt(d)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            numerator, _ = integrate.quad(lambda x, _d=d: x * pdf(x, _d), a, b)
            denominator, _ = integrate.quad(lambda x, _d=d: pdf(x, _d), a, b)
            new_centroids.append(
                numerator / denominator if denominator > 1e-15 else centroids[i]
            )
        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


class LloydMaxCodebook:
    """Precomputed Lloyd-Max codebook for a given vector dimension and bit-width."""

    def __init__(self, d: int, bits: int, use_exact: bool = False):
        self.d = d
        self.bits = bits
        self.n_levels = 2 ** bits
        self.centroids, self.boundaries = solve_lloyd_max(d, bits, use_exact)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Map values to nearest centroid indices via bucketize (no large intermediates)."""
        return torch.bucketize(x, self.boundaries.to(x.device))

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Map indices back to centroid values."""
        return self.centroids.to(indices.device)[indices]

    def __repr__(self):
        return f"LloydMaxCodebook(d={self.d}, bits={self.bits}, levels={self.n_levels})"


# ─────────────────────────────────────────────────────────────────────────────
# Global caches — rotation matrices and codebooks are expensive to recompute
# ─────────────────────────────────────────────────────────────────────────────

_codebook_cache: dict = {}   # (d, bits) -> LloydMaxCodebook
_rotation_cache: dict = {}   # (d, seed) -> orthogonal matrix (CPU tensor)


def _get_codebook(d: int, bits: int) -> LloydMaxCodebook:
    """Return (optionally cached) LloydMaxCodebook for dimension d, bits wide."""
    key = (d, bits)
    if key not in _codebook_cache:
        print(f"    [TurboQuant] Building Lloyd-Max codebook  d={d}, bits={bits} …")
        _codebook_cache[key] = LloydMaxCodebook(d, bits)
    return _codebook_cache[key]


def _get_rotation(d: int, seed: int, device) -> torch.Tensor:
    """Return (optionally cached) random orthogonal rotation matrix for dimension d."""
    key = (d, seed)
    if key not in _rotation_cache:
        _rotation_cache[key] = generate_rotation_matrix(d, seed=seed)  # kept on CPU
    return _rotation_cache[key].to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Core quantization functions
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def turboquant_weight_per_row(
    w: torch.Tensor,
    bits: int = 4,
    seed: int = 42,
) -> torch.Tensor:
    """
    Fake-quantize a weight matrix with TurboQuant rotation + Lloyd-Max.

    Each row (= one output neuron's weights) is treated as a vector:
      1. Normalize to unit sphere.
      2. Apply random rotation Pi  →  coordinates become ~N(0, 1/in).
      3. Assign every coordinate to its nearest Lloyd-Max centroid.
      4. Dequantize, un-rotate, rescale by saved norm.

    Args:
        w:    Weight tensor of shape (out_features, in_features).
        bits: Bit-width used for each coordinate (e.g. 4 → 16 levels).
        seed: Seed for the random rotation matrix.

    Returns:
        w_hat: Fake-quantized weight tensor with the same dtype and shape as w.
    """
    dtype = w.dtype
    device = w.device
    in_features = w.shape[1]

    w_f32 = w.float()

    # 1. Normalize rows to unit sphere
    row_norms = w_f32.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (out, 1)
    w_norm = w_f32 / row_norms                                      # (out, in)

    # 2. Rotate: coordinates become approximately iid N(0, 1/in_features)
    Pi = _get_rotation(in_features, seed=seed, device=device)       # (in, in)
    w_rotated = w_norm @ Pi.T                                        # (out, in)

    # 3. Lloyd-Max quantization via bucketize — O(out*in*log(n_levels)), no large intermediates
    #    torch.bucketize(x, boundaries) assigns each value to its partition bucket,
    #    which equals the index of the nearest centroid (boundaries are midpoints).
    codebook = _get_codebook(in_features, bits)
    boundaries = codebook.boundaries.to(device)   # (n_levels-1,) sorted
    centroids = codebook.centroids.to(device)     # (n_levels,)
    indices = torch.bucketize(w_rotated, boundaries)   # (out, in)
    w_hat_rotated = centroids[indices]                 # (out, in)

    # 4. Undo rotation and restore original scale
    w_hat = (w_hat_rotated @ Pi) * row_norms                         # (out, in)

    return w_hat.to(dtype)


@torch.no_grad()
def quantize_activation_per_token_turbo(
    t: torch.Tensor,
    bits: int = 4,
    seed: int = 1234,
) -> torch.Tensor:
    """
    Per-token TurboQuant activation quantization.

    Treat each token vector (last dimension) as one TurboQuant vector:
      normalize -> rotate -> Lloyd-Max bucketize/dequantize -> unrotate -> rescale.
    Returns a fake-quantized tensor with the same shape/dtype as input.
    """
    t_shape = t.shape
    d = t_shape[-1]

    t_2d = t.reshape(-1, d)
    t_f32 = t_2d.float()

    # 1) Normalize each token vector to unit sphere
    token_norms = t_f32.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    t_norm = t_f32 / token_norms

    # 2) Rotate
    Pi = _get_rotation(d, seed=seed, device=t.device)
    t_rot = t_norm @ Pi.T

    # 3) Lloyd-Max quantize/dequantize with bucketize
    codebook = _get_codebook(d, bits)
    boundaries = codebook.boundaries.to(t.device)
    centroids = codebook.centroids.to(t.device)
    indices = torch.bucketize(t_rot, boundaries)
    t_hat_rot = centroids[indices]

    # 4) Undo rotation and rescale
    t_hat = (t_hat_rot @ Pi) * token_norms
    return t_hat.reshape(t_shape).to(t.dtype)


@torch.no_grad()
def quantize_activation_per_token_absmax(
    t: torch.Tensor,
    n_bits: int = 8,
) -> torch.Tensor:
    """
    Per-token absmax activation quantization.

    Identical to the function in fake_quant.py so that activation quantization
    is kept constant when comparing VLaQuant vs TurboQuant weight quantization.
    """
    t_shape = t.shape
    t_2d = t.view(-1, t_shape[-1])
    scales = t_2d.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t_2d.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(
    t: torch.Tensor,
    n_bits: int = 8,
) -> torch.Tensor:
    """Per-tensor absmax activation quantization (alternative to per-token)."""
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales = scales.clamp(min=1e-5) / q_max
    t.div_(scales).round_().mul_(scales)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# TurboQuantLinear — drop-in replacement for nn.Linear
# ─────────────────────────────────────────────────────────────────────────────

class TurboQuantLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear using TurboQuant weight quantization.

    Weight quantization : TurboQuant rotation + Lloyd-Max (per-row, MSE-optimal)
    Activation quantization : configurable absmax or TurboQuant per-token fake-quant

    Analog of W8A8Linear in fake_quant.py, but replaces absmax with the
    theoretically optimal rotation-based quantizer from TurboQuant.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        act_quant: str = "per_token",
        quantize_output: bool = False,
        bits: int = 4,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        self.register_buffer(
            "weight",
            torch.zeros(out_features, in_features, dtype=torch.float16, requires_grad=False),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(1, out_features, dtype=torch.float16, requires_grad=False),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=self.bits)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=self.bits)
        elif act_quant == "turbo_per_token":
            self.act_quant_name = f"turbo_per_token_{self.bits}bit"
            self.act_quant = partial(
                quantize_activation_per_token_turbo,
                bits=self.bits,
                seed=1234,
            )
        else:
            raise ValueError(f"Invalid act_quant: {act_quant!r}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

        self.weight_quant_name = f"turboquant_per_row_{bits}bit"

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_x = self.act_quant(x)
        y = F.linear(q_x, self.weight, self.bias)
        return self.output_quant(y)

    @staticmethod
    def from_float(
        module: nn.Linear,
        bits: int = 4,
        act_quant: str = "turbo_per_token",
        quantize_output: bool = False,
        weight_seed: int = 42,
    ) -> "TurboQuantLinear":
        """Convert an nn.Linear to TurboQuantLinear with quantized weights."""
        assert isinstance(module, nn.Linear)
        new_module = TurboQuantLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            bits=bits,
        )
        new_module.weight = turboquant_weight_per_row(
            module.weight, bits=bits, seed=weight_seed
        )
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self) -> str:
        return (
            f"TurboQuantLinear({self.in_features}, {self.out_features}, "
            f"bias={self.bias is not None}, "
            f"weight_quant={self.weight_quant_name}, "
            f"act_quant={self.act_quant_name}, "
            f"output_quant={self.output_quant_name})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model-level entry point (analog of quantize_vla_like in fake_quant.py)
# ─────────────────────────────────────────────────────────────────────────────

def quantize_vla_like_turbo(
    model,
    bits: int = 4,
    act_quant: str = "turbo_per_token",
    quantize_bmm_input: bool = False,
):
    """
    Replace GemmaMLP and GemmaAttention Linear layers with TurboQuantLinear.

    Drop-in replacement for quantize_vla_like() from fake_quant.py.
    Uses TurboQuant rotation + Lloyd-Max for weight quantization instead of
    per-channel absmax.  Activation quantization scheme is kept identical.

    Args:
        model:              pi0.5 policy model.
        bits:               Bit-width for weight quantization (default 4).
                            Use 8 for near-lossless; 4 for ~2x compression vs INT8.
        act_quant:          Activation quantization scheme ("per_token" / "per_tensor").
        quantize_bmm_input: If True, also quantize QKV projection outputs
                            (simulates quantization of BMM inputs).

    Returns:
        model with targeted Linear layers replaced by TurboQuantLinear.
    """
    from transformers.models.gemma.modeling_gemma import GemmaAttention, GemmaMLP

    seed_counter = 0

    for name, m in model.paligemma_with_expert.paligemma.model.language_model.named_modules():
        if isinstance(m, GemmaMLP):
            print(f"  [TurboQuant] Quantizing MLP: {name}")
            m.gate_proj = TurboQuantLinear.from_float(
                m.gate_proj, bits=bits, act_quant=act_quant,
                weight_seed=seed_counter,
            )
            seed_counter += 1
            m.up_proj = TurboQuantLinear.from_float(
                m.up_proj, bits=bits, act_quant=act_quant,
                weight_seed=seed_counter,
            )
            seed_counter += 1
            m.down_proj = TurboQuantLinear.from_float(
                m.down_proj, bits=bits, act_quant=act_quant,
                weight_seed=seed_counter,
            )
            seed_counter += 1

        elif isinstance(m, GemmaAttention):
            print(f"  [TurboQuant] Quantizing Attention: {name}")
            m.q_proj = TurboQuantLinear.from_float(
                m.q_proj, bits=bits, act_quant=act_quant,
                quantize_output=quantize_bmm_input, weight_seed=seed_counter,
            )
            seed_counter += 1
            m.k_proj = TurboQuantLinear.from_float(
                m.k_proj, bits=bits, act_quant=act_quant,
                quantize_output=quantize_bmm_input, weight_seed=seed_counter,
            )
            seed_counter += 1
            m.v_proj = TurboQuantLinear.from_float(
                m.v_proj, bits=bits, act_quant=act_quant,
                quantize_output=quantize_bmm_input, weight_seed=seed_counter,
            )
            seed_counter += 1
            m.o_proj = TurboQuantLinear.from_float(
                m.o_proj, bits=bits, act_quant=act_quant,
                weight_seed=seed_counter,
            )
            seed_counter += 1

    return model
