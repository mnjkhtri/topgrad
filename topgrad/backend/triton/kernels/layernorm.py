import torch
import triton
import triton.language as tl

from ._common import (
    _BIAS_BLOCK_M,
    _BIAS_BLOCK_N,
    _bias_add,
    _col_sum,
    _colwise_mul,
    _elementwise_add,
    _elementwise_mul,
    _require_2d,
    _require_contiguous,
    _row_sum,
    _rowwise_add,
    _rowwise_mul,
    _rsqrt,
)


@triton.jit
def _rowwise_norm_kernel(
    x_ptr,
    mean_ptr,
    inv_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < n_rows
    mask_n = offs_n < n_cols
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + offs_m, mask=mask_m, other=0.0)
    inv = tl.load(inv_ptr + offs_m, mask=mask_m, other=0.0)
    out = (x - mean[:, None]) * inv[:, None]
    out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def _rowwise_norm_affine_kernel(
    x_ptr,
    mean_ptr,
    inv_ptr,
    gamma_ptr,
    beta_ptr,
    xhat_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_xm,
    stride_xn,
    stride_xhatm,
    stride_xhatn,
    stride_outm,
    stride_outn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < n_rows
    mask_n = offs_n < n_cols
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + offs_m, mask=mask_m, other=0.0)
    inv = tl.load(inv_ptr + offs_m, mask=mask_m, other=0.0)
    x_hat = (x - mean[:, None]) * inv[:, None]
    gamma = tl.load(gamma_ptr + offs_n, mask=mask_n, other=0.0)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0)
    out = x_hat * gamma[None, :] + beta[None, :]

    xhat_ptrs = (
        xhat_ptr + offs_m[:, None] * stride_xhatm + offs_n[None, :] * stride_xhatn
    )
    out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    tl.store(xhat_ptrs, x_hat, mask=mask)
    tl.store(out_ptrs, out, mask=mask)


def _rowwise_norm(x, mean, inv_std):
    _require_contiguous(x)
    _require_contiguous(mean)
    _require_contiguous(inv_std)
    _require_2d(x, "rowwise_norm")
    if mean.dim() != 1 or inv_std.dim() != 1:
        raise ValueError("rowwise norm expects 1D vectors")
    if x.shape[0] != mean.shape[0] or x.shape[0] != inv_std.shape[0]:
        raise ValueError("rowwise norm shape mismatch")
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    if n_rows == 0 or n_cols == 0:
        return out
    grid = (
        triton.cdiv(n_rows, _BIAS_BLOCK_M),
        triton.cdiv(n_cols, _BIAS_BLOCK_N),
    )
    _rowwise_norm_kernel[grid](
        x,
        mean,
        inv_std,
        out,
        n_rows,
        n_cols,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=_BIAS_BLOCK_M,
        BLOCK_N=_BIAS_BLOCK_N,
    )
    return out


def _rowwise_norm_affine(x, mean, inv_std, gamma, beta):
    _require_contiguous(x)
    _require_contiguous(mean)
    _require_contiguous(inv_std)
    _require_contiguous(gamma)
    _require_contiguous(beta)
    _require_2d(x, "rowwise_norm_affine")
    if mean.dim() != 1 or inv_std.dim() != 1 or gamma.dim() != 1 or beta.dim() != 1:
        raise ValueError("rowwise norm affine expects 1D vectors")
    if (
        x.shape[0] != mean.shape[0]
        or x.shape[0] != inv_std.shape[0]
        or x.shape[1] != gamma.shape[0]
        or x.shape[1] != beta.shape[0]
    ):
        raise ValueError("rowwise norm affine shape mismatch")
    n_rows, n_cols = x.shape
    x_hat = torch.empty_like(x)
    out = torch.empty_like(x)
    if n_rows == 0 or n_cols == 0:
        return x_hat, out
    grid = (
        triton.cdiv(n_rows, _BIAS_BLOCK_M),
        triton.cdiv(n_cols, _BIAS_BLOCK_N),
    )
    _rowwise_norm_affine_kernel[grid](
        x,
        mean,
        inv_std,
        gamma,
        beta,
        x_hat,
        out,
        n_rows,
        n_cols,
        x.stride(0),
        x.stride(1),
        x_hat.stride(0),
        x_hat.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=_BIAS_BLOCK_M,
        BLOCK_N=_BIAS_BLOCK_N,
    )
    return x_hat, out


def _layernorm_fused(x, gamma, beta, eps):
    _require_contiguous(x)
    _require_contiguous(gamma)
    _require_contiguous(beta)
    _require_2d(x, "layernorm")
    if gamma.dim() != 1 or beta.dim() != 1:
        raise ValueError("layernorm expects 1D gamma/beta")
    if x.shape[1] != gamma.shape[0] or x.shape[1] != beta.shape[0]:
        raise ValueError("layernorm gamma/beta shape mismatch")
    n_rows, n_cols = x.shape
    if n_cols == 0:
        x_hat = torch.empty_like(x)
        inv_std = torch.empty((n_rows,), device=x.device, dtype=x.dtype)
        y = torch.empty_like(x)
        return x_hat, inv_std, y
    scale = 1.0 / float(n_cols)
    mean = _elementwise_mul(_row_sum(x), scale)
    x_sq = _elementwise_mul(x, x)
    mean_sq = _elementwise_mul(_row_sum(x_sq), scale)
    mean_sq = _elementwise_add(
        mean_sq, _elementwise_mul(_elementwise_mul(mean, mean), -1.0)
    )
    inv_std = _rsqrt(mean_sq, eps)
    x_centered = _rowwise_add(x, _elementwise_mul(mean, -1.0))
    x_hat = _rowwise_mul(x_centered, inv_std)
    y = _colwise_mul(x_hat, gamma)
    y = _bias_add(y, beta)
    return x_hat, inv_std, y


def _layernorm_backward(grad, x_hat, inv_std, gamma):
    _require_contiguous(grad)
    _require_contiguous(x_hat)
    _require_contiguous(inv_std)
    _require_contiguous(gamma)
    _require_2d(grad, "layernorm_backward")
    if grad.shape != x_hat.shape:
        raise ValueError("layernorm backward expects matching grad/x_hat")
    if inv_std.dim() != 1:
        raise ValueError("layernorm backward expects 1D inv_std")
    if gamma.dim() != 1:
        raise ValueError("layernorm backward expects 1D gamma")
    if grad.shape[0] != inv_std.shape[0] or grad.shape[1] != gamma.shape[0]:
        raise ValueError("layernorm backward shape mismatch")
    n_rows, n_cols = grad.shape
    if n_cols == 0:
        dx = torch.empty_like(grad)
        dgamma = torch.empty_like(gamma)
        dbeta = torch.empty_like(gamma)
        return dx, dgamma, dbeta
    scale = 1.0 / float(n_cols)
    dgamma = _col_sum(_elementwise_mul(grad, x_hat))
    dbeta = _col_sum(grad)
    g_hat = _colwise_mul(grad, gamma)
    m1 = _elementwise_mul(_row_sum(g_hat), scale)
    m2 = _elementwise_mul(_row_sum(_elementwise_mul(g_hat, x_hat)), scale)
    x_hat_m2 = _rowwise_mul(x_hat, m2)
    dx = _rowwise_add(g_hat, _elementwise_mul(m1, -1.0))
    dx = _elementwise_add(dx, _elementwise_mul(x_hat_m2, -1.0))
    dx = _rowwise_mul(dx, inv_std)
    return dx, dgamma, dbeta


__all__ = [
    "_layernorm_backward",
    "_layernorm_fused",
    "_rowwise_norm",
    "_rowwise_norm_affine",
    "_rowwise_norm_kernel",
    "_rowwise_norm_affine_kernel",
]
