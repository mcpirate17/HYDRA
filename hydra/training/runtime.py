"""
Runtime setup utilities for HYDRA training.
Applies torch backend knobs, Dynamo/Inductor settings, and kernel patches.
"""
from __future__ import annotations

import os
import sys
import signal
from pathlib import Path

import torch
import torch._inductor.config as inductor_config


def configure_runtime() -> dict:
    """Apply global runtime settings and kernel patches.

    Returns a status dict for logging (e.g., Liger/Flash availability).
    """
    status = {}

    # Optional Windows console handler (keep Ctrl+C behavior intact)
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleCtrlHandler(None, True)

    # Enable HF_TRANSFER for faster downloads
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # CUDA/cuDNN optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Fix torch.compile recompilation storm for polymorphic layers
    torch._dynamo.config.allow_unspec_int_on_nn_module = True
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.recompile_limit = 32

    # Inductor defaults
    inductor_config.triton.cudagraphs = False

    # Kernel/backends
    from hydra.kernels.liger_integration import (
        LIGER_AVAILABLE,
        patch_hydra_with_liger,
    )
    from hydra.layers.common import (
        FLASH_ATTN_AVAILABLE,
        set_attention_backend,
    )

    if LIGER_AVAILABLE:
        patch_hydra_with_liger()
        status["liger"] = "enabled"
    else:
        status["liger"] = "unavailable"

    if FLASH_ATTN_AVAILABLE:
        set_attention_backend("flash")
        status["flash_attn"] = "enabled"
    else:
        status["flash_attn"] = "unavailable"

    return status
