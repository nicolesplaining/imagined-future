"""Import-only compatibility shim for LingBot's unused FlashAttention branch.

The released LIBERO server explicitly instantiates ``attn_mode='torch'`` but
imports ``flash_attn_func`` unconditionally at module import time.  Deadline
experiments put this directory last in the project PYTHONPATH so the official
PyTorch SDPA path can run while the optional CUDA extension is unavailable.
Calling this shim is an error, which makes accidental use fail loudly.
"""


def flash_attn_func(*args, **kwargs):
    raise RuntimeError(
        "FlashAttention compatibility shim was called; the LingBot LIBERO "
        "experiment must use the released attn_mode='torch' path"
    )
