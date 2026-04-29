import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from openpi.quant.fake_quant import quantize_vla_like
from openpi.quant.smooth_vla import smooth_lm
from openpi.quant.turbo_quant_weight import quantize_vla_like_turbo
from openpi.quant.nvfp4_quant import quantize_vla_like_nvfp4
import tqdm

from openpi.policies import policy_config
from openpi.policies.libero_policy import make_libero_example
from openpi.training import config as _config

import argparse


def create_synthetic_example(config_name):
    """Create a synthetic example based on the config type."""
    print("  - Using synthetic example (random data)")

    if "libero" in config_name.lower():
        example = make_libero_example()
        print("  - Type: LIBERO")
    else:
        print(f"  - Warning: Unknown config type '{config_name}', defaulting to LIBERO")
        example = make_libero_example()

    print(f"  - Prompt: {example.get('prompt', 'N/A')}")
    return example

def quant_pi05_policy(policy, smooth=False, weight_quant="per_channel", act_quant="per_token", n_bits=8, group_size=128):
    model_pi05_fp16 = policy._model
    # print(model_pi05_fp16)

    if smooth:
        act_scales = torch.load("/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_lijinhao/pi0.5_deploy/openpi_quant/vlaquant/random_act_scale_512.pt")
        smooth_lm(model_pi05_fp16, act_scales, 0.5)
        print(f'Smooth Complete.')
    
    model_pi05_w8a8 = quantize_vla_like(model_pi05_fp16, weight_quant=weight_quant, act_quant=act_quant, n_bits=n_bits, group_size=group_size)
    # print(model_pi05_w8a8)
    policy._model = model_pi05_w8a8
    print(f'W{n_bits}A{n_bits} W{weight_quant} A{act_quant} Group{group_size} Quant Complete.')
    return policy

def quant_pi05_policy_turbo(policy, n_bits=8):
    """
    Quantize the pi0.5 policy model using TurboQuant.

    Drop-in replacement for quant_pi05_policy() in vlaquant_demo.py.
    Does NOT require pre-computed activation scales (unlike SmoothQuant).

    Args:
        policy : loaded pi0.5 policy object.
        bits   : quantization bit-width (default 4; 8 for near-lossless).

    Returns:
        policy with GemmaMLP / GemmaAttention linear layers replaced by
        TurboQuantLinear (fake-quantized, weights stored in fp16).
    """
    model = policy._model
    print(f"  Applying TurboQuant {n_bits}-bit quantization …")
    policy._model = quantize_vla_like_turbo(model, bits=n_bits, act_quant="turbo_per_token")
    print(f'W{n_bits}A{n_bits} TurboQuant Complete.')
    return policy

def quant_pi05_policy_nvfp4(policy, n_bits=4):
    """
    Quantize the pi0.5 policy model using NVFP4.

    Drop-in replacement for quant_pi05_policy() in vlaquant_demo.py.
    Does NOT require pre-computed activation scales (unlike SmoothQuant).

    Args:
        policy : loaded pi0.5 policy object.
        bits   : quantization bit-width (default 4; 8 for near-lossless).

    Returns:
        policy with GemmaMLP / GemmaAttention linear layers replaced by
        TurboQuantLinear (fake-quantized, weights stored in fp16).
    """
    model = policy._model
    print(f"  Applying NVFP4 quantization …")
    policy._model = quantize_vla_like_nvfp4(model)
    print(f'NVFP4 Quant Complete.')
    return policy
