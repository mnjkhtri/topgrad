import torch
import triton
import triton.language as tl

from ._common import (
    _LOGSOFTMAX_BLOCK_N,
    _elementwise_mul,
    _require_contiguous,
    _reshape_to_2d,
    _row_sum_kernel,
    _rowwise_add,
)


@triton.jit
def _row_max_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_xm,
    stride_xn,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    x_row = x_ptr + row * stride_xm

    # Compute max per row for numerical stability.
    max_val = tl.full((BLOCK_N,), float("-inf"), dtype=tl.float32)
    for n in range(0, n_cols, BLOCK_N):
        n_idx = n + offs
        mask = n_idx < n_cols
        x = tl.load(x_row + n_idx * stride_xn, mask=mask, other=float("-inf"))
        max_val = tl.maximum(max_val, x)

    max_val = tl.max(max_val, axis=0)
    tl.store(out_ptr + row, max_val)


@triton.jit
def _row_sumexp_kernel(
    x_ptr,
    max_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_xm,
    stride_xn,
    stride_max,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    x_row = x_ptr + row * stride_xm
    max_val = tl.load(max_ptr + row * stride_max)

    # Sum exp(x - max) to avoid overflow.
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n in range(0, n_cols, BLOCK_N):
        n_idx = n + offs
        mask = n_idx < n_cols
        x = tl.load(x_row + n_idx * stride_xn, mask=mask, other=float("-inf"))
        acc += tl.exp(x - max_val)

    sum_val = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, sum_val)


@triton.jit
def _logsoftmax_kernel(
    x_ptr,
    max_ptr,
    sum_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    stride_max,
    stride_sum,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    x_row = x_ptr + row * stride_xm
    out_row = out_ptr + row * stride_outm
    max_val = tl.load(max_ptr + row * stride_max)
    sum_val = tl.load(sum_ptr + row * stride_sum)
    log_sum = tl.log(sum_val) + max_val

    for n in range(0, n_cols, BLOCK_N):
        n_idx = n + offs
        mask = n_idx < n_cols
        x = tl.load(x_row + n_idx * stride_xn, mask=mask, other=float("-inf"))
        y = x - log_sum
        tl.store(out_row + n_idx * stride_outn, y, mask=mask)


@triton.jit
def _softmax_kernel(
    x_ptr,
    max_ptr,
    sum_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    stride_max,
    stride_sum,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    x_row = x_ptr + row * stride_xm
    out_row = out_ptr + row * stride_outm
    max_val = tl.load(max_ptr + row * stride_max)
    sum_val = tl.load(sum_ptr + row * stride_sum)

    for n in range(0, n_cols, BLOCK_N):
        n_idx = n + offs
        mask = n_idx < n_cols
        x = tl.load(x_row + n_idx * stride_xn, mask=mask, other=float("-inf"))
        y = tl.exp(x - max_val) / sum_val
        tl.store(out_row + n_idx * stride_outn, y, mask=mask)


@triton.jit
def _logsoftmax_backward_kernel(
    grad_ptr,
    y_ptr,
    sum_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_gm,
    stride_gn,
    stride_ym,
    stride_yn,
    stride_outm,
    stride_outn,
    stride_sum,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    grad_row = grad_ptr + row * stride_gm
    y_row = y_ptr + row * stride_ym
    out_row = out_ptr + row * stride_outm
    sum_grad = tl.load(sum_ptr + row * stride_sum)

    for n in range(0, n_cols, BLOCK_N):
        n_idx = n + offs
        mask = n_idx < n_cols
        grad = tl.load(grad_row + n_idx * stride_gn, mask=mask, other=0.0)
        y = tl.load(y_row + n_idx * stride_yn, mask=mask, other=float("-inf"))
        softmax = tl.exp(y)
        out = grad - softmax * sum_grad
        tl.store(out_row + n_idx * stride_outn, out, mask=mask)


def _logsoftmax_forward(x):
    _require_contiguous(x)
    if x.numel() == 0:
        return torch.empty_like(x)

    x2d, n_rows, n_cols = _reshape_to_2d(x)
    if n_cols == 0:
        return torch.empty_like(x)

    row_max = torch.empty((n_rows,), device=x.device, dtype=x.dtype)
    row_sum = torch.empty((n_rows,), device=x.device, dtype=x.dtype)
    out2d = torch.empty_like(x2d)
    grid = (n_rows,)

    _row_max_kernel[grid](
        x2d,
        row_max,
        n_rows,
        n_cols,
        x2d.stride(0),
        x2d.stride(1),
        BLOCK_N=_LOGSOFTMAX_BLOCK_N,
    )
    _row_sumexp_kernel[grid](
        x2d,
        row_max,
        row_sum,
        n_rows,
        n_cols,
        x2d.stride(0),
        x2d.stride(1),
        row_max.stride(0),
        BLOCK_N=_LOGSOFTMAX_BLOCK_N,
    )
    _logsoftmax_kernel[grid](
        x2d,
        row_max,
        row_sum,
        out2d,
        n_rows,
        n_cols,
        x2d.stride(0),
        x2d.stride(1),
        out2d.stride(0),
        out2d.stride(1),
        row_max.stride(0),
        row_sum.stride(0),
        BLOCK_N=_LOGSOFTMAX_BLOCK_N,
    )
    return out2d.reshape(x.shape)


def _softmax_forward(x):
    _require_contiguous(x)
    if x.numel() == 0:
        return torch.empty_like(x)

    x2d, n_rows, n_cols = _reshape_to_2d(x)
    if n_cols == 0:
        return torch.empty_like(x)

    row_max = torch.empty((n_rows,), device=x.device, dtype=x.dtype)
    row_sum = torch.empty((n_rows,), device=x.device, dtype=x.dtype)
    out2d = torch.empty_like(x2d)
    grid = (n_rows,)

    _row_max_kernel[grid](
        x2d,
        row_max,
        n_rows,
        n_cols,
        x2d.stride(0),
        x2d.stride(1),
        BLOCK_N=_LOGSOFTMAX_BLOCK_N,
    )
    _row_sumexp_kernel[grid](
        x2d,
        row_max,
        row_sum,
        n_rows,
        n_cols,
        x2d.stride(0),
        x2d.stride(1),
        row_max.stride(0),
        BLOCK_N=_LOGSOFTMAX_BLOCK_N,
    )
    _softmax_kernel[grid](
        x2d,
        row_max,
        row_sum,
        out2d,
        n_rows,
        n_cols,
        x2d.stride(0),
        x2d.stride(1),
        out2d.stride(0),
        out2d.stride(1),
        row_max.stride(0),
        row_sum.stride(0),
        BLOCK_N=_LOGSOFTMAX_BLOCK_N,
    )
    return out2d.reshape(x.shape)


def _logsoftmax_backward(grad, y):
    _require_contiguous(grad)
    _require_contiguous(y)
    if grad.numel() == 0:
        return torch.empty_like(grad)

    grad2d, n_rows, n_cols = _reshape_to_2d(grad)
    y2d = y.reshape((n_rows, n_cols))
    if n_cols == 0:
        return torch.empty_like(grad)

    row_sum = torch.empty((n_rows,), device=grad.device, dtype=grad.dtype)
    out2d = torch.empty_like(grad2d)
    grid = (n_rows,)

    _row_sum_kernel[grid](
        grad2d,
        row_sum,
        n_rows,
        n_cols,
        grad2d.stride(0),
        grad2d.stride(1),
        BLOCK_N=_LOGSOFTMAX_BLOCK_N,
    )
    _logsoftmax_backward_kernel[grid](
        grad2d,
        y2d,
        row_sum,
        out2d,
        n_rows,
        n_cols,
        grad2d.stride(0),
        grad2d.stride(1),
        y2d.stride(0),
        y2d.stride(1),
        out2d.stride(0),
        out2d.stride(1),
        row_sum.stride(0),
        BLOCK_N=_LOGSOFTMAX_BLOCK_N,
    )
    return out2d.reshape(grad.shape)


def _softmax_backward(grad, y):
    _require_contiguous(grad)
    _require_contiguous(y)
    if grad.numel() == 0:
        return torch.empty_like(grad)

    grad2d, n_rows, n_cols = _reshape_to_2d(grad)
    y2d = y.reshape((n_rows, n_cols))
    if n_cols == 0:
        return torch.empty_like(grad)

    grad_y = _elementwise_mul(grad2d, y2d)
    row_sum = torch.empty((n_rows,), device=grad.device, dtype=grad.dtype)
    grid = (n_rows,)

    _row_sum_kernel[grid](
        grad_y,
        row_sum,
        n_rows,
        n_cols,
        grad_y.stride(0),
        grad_y.stride(1),
        BLOCK_N=_LOGSOFTMAX_BLOCK_N,
    )
    grad_minus = _rowwise_add(grad2d, _elementwise_mul(row_sum, -1.0))
    out2d = _elementwise_mul(grad_minus, y2d)
    return out2d.reshape(grad.shape)


__all__ = [
    "_row_max_kernel",
    "_row_sumexp_kernel",
    "_logsoftmax_kernel",
    "_softmax_kernel",
    "_logsoftmax_backward_kernel",
    "_logsoftmax_forward",
    "_softmax_forward",
    "_logsoftmax_backward",
    "_softmax_backward",
]
