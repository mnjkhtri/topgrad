import torch
import triton
import triton.language as tl

from ._common import _BLOCK_SIZE, _numel, _require_contiguous


@triton.jit
def _relu_forward_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    y = tl.maximum(x, 0)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _relu_backward_kernel(
    x_ptr, grad_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    grad = tl.load(grad_ptr + offs, mask=mask, other=0)
    # Mask gradients where activation is zero.
    out = tl.where(x > 0, grad, 0)
    tl.store(out_ptr + offs, out, mask=mask)


def _relu_forward(x):
    _require_contiguous(x)
    n_elements = _numel(x.shape)
    out = torch.empty_like(x)
    if n_elements == 0:
        return out
    blocks = triton.cdiv(n_elements, _BLOCK_SIZE)
    _relu_forward_kernel[(blocks,)](x, out, n_elements, BLOCK_SIZE=_BLOCK_SIZE)
    return out


def _relu_backward(x, grad):
    _require_contiguous(x)
    _require_contiguous(grad)
    n_elements = _numel(x.shape)
    out = torch.empty_like(grad)
    if n_elements == 0:
        return out
    blocks = triton.cdiv(n_elements, _BLOCK_SIZE)
    _relu_backward_kernel[(blocks,)](x, grad, out, n_elements, BLOCK_SIZE=_BLOCK_SIZE)
    return out


__all__ = [
    "_relu_forward",
    "_relu_backward",
    "_relu_forward_kernel",
    "_relu_backward_kernel",
]
