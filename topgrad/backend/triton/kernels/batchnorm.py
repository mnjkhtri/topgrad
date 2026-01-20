import torch
import triton
import triton.language as tl

from ._common import (
    _BIAS_BLOCK_M,
    _BIAS_BLOCK_N,
    _COL_SUM_BLOCK_M,
    _require_2d,
    _require_contiguous,
)


@triton.jit
def _colwise_norm_kernel(
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
    mean = tl.load(mean_ptr + offs_n, mask=mask_n, other=0.0)
    inv = tl.load(inv_ptr + offs_n, mask=mask_n, other=0.0)
    out = (x - mean[None, :]) * inv[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def _colwise_norm_affine_kernel(
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
    mean = tl.load(mean_ptr + offs_n, mask=mask_n, other=0.0)
    inv = tl.load(inv_ptr + offs_n, mask=mask_n, other=0.0)
    x_hat = (x - mean[None, :]) * inv[None, :]
    gamma = tl.load(gamma_ptr + offs_n, mask=mask_n, other=0.0)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0)
    out = x_hat * gamma[None, :] + beta[None, :]

    xhat_ptrs = (
        xhat_ptr + offs_m[:, None] * stride_xhatm + offs_n[None, :] * stride_xhatn
    )
    out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    tl.store(xhat_ptrs, x_hat, mask=mask)
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def _colwise_norm_affine_fused_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    var_ptr,
    inv_ptr,
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
    eps,
    BLOCK_M: tl.constexpr,
):
    col = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    x_ptrs = x_ptr + offs_m * stride_xm + col * stride_xn

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for m in range(0, n_rows, BLOCK_M):
        m_idx = m + offs_m
        mask = m_idx < n_rows
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x_f = x.to(tl.float32)
        acc += x_f
        acc2 += x_f * x_f
        x_ptrs += BLOCK_M * stride_xm

    gamma = tl.load(gamma_ptr + col, mask=col < n_cols, other=0.0)
    beta = tl.load(beta_ptr + col, mask=col < n_cols, other=0.0)
    sum_val = tl.sum(acc, axis=0)
    sum_sq = tl.sum(acc2, axis=0)
    mean = sum_val / n_rows
    var = sum_sq / n_rows - mean * mean
    inv = tl.rsqrt(var + eps)
    tl.store(mean_ptr + col, mean.to(gamma.dtype))
    tl.store(var_ptr + col, var.to(gamma.dtype))
    tl.store(inv_ptr + col, inv.to(gamma.dtype))

    x_ptrs = x_ptr + offs_m * stride_xm + col * stride_xn
    xhat_ptrs = xhat_ptr + offs_m * stride_xhatm + col * stride_xhatn
    out_ptrs = out_ptr + offs_m * stride_outm + col * stride_outn
    for m in range(0, n_rows, BLOCK_M):
        m_idx = m + offs_m
        mask = m_idx < n_rows
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x_f = x.to(tl.float32)
        x_hat = (x_f - mean) * inv
        out = x_hat * gamma + beta
        tl.store(xhat_ptrs, x_hat.to(x.dtype), mask=mask)
        tl.store(out_ptrs, out.to(x.dtype), mask=mask)
        x_ptrs += BLOCK_M * stride_xm
        xhat_ptrs += BLOCK_M * stride_xhatm
        out_ptrs += BLOCK_M * stride_outm


def _colwise_norm(x, mean, inv_std):
    _require_contiguous(x)
    _require_contiguous(mean)
    _require_contiguous(inv_std)
    _require_2d(x, "colwise_norm")
    if mean.dim() != 1 or inv_std.dim() != 1:
        raise ValueError("colwise norm expects 1D vectors")
    if x.shape[1] != mean.shape[0] or x.shape[1] != inv_std.shape[0]:
        raise ValueError("colwise norm shape mismatch")
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    if n_rows == 0 or n_cols == 0:
        return out
    grid = (
        triton.cdiv(n_rows, _BIAS_BLOCK_M),
        triton.cdiv(n_cols, _BIAS_BLOCK_N),
    )
    _colwise_norm_kernel[grid](
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


def _colwise_norm_affine(x, mean, inv_std, gamma, beta):
    _require_contiguous(x)
    _require_contiguous(mean)
    _require_contiguous(inv_std)
    _require_contiguous(gamma)
    _require_contiguous(beta)
    _require_2d(x, "colwise_norm_affine")
    if mean.dim() != 1 or inv_std.dim() != 1 or gamma.dim() != 1 or beta.dim() != 1:
        raise ValueError("colwise norm affine expects 1D vectors")
    if (
        x.shape[1] != mean.shape[0]
        or x.shape[1] != inv_std.shape[0]
        or x.shape[1] != gamma.shape[0]
        or x.shape[1] != beta.shape[0]
    ):
        raise ValueError("colwise norm affine shape mismatch")
    n_rows, n_cols = x.shape
    x_hat = torch.empty_like(x)
    out = torch.empty_like(x)
    if n_rows == 0 or n_cols == 0:
        return x_hat, out
    grid = (
        triton.cdiv(n_rows, _BIAS_BLOCK_M),
        triton.cdiv(n_cols, _BIAS_BLOCK_N),
    )
    _colwise_norm_affine_kernel[grid](
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


def _colwise_norm_affine_fused(x, gamma, beta, eps):
    _require_contiguous(x)
    _require_contiguous(gamma)
    _require_contiguous(beta)
    _require_2d(x, "colwise_norm_affine_fused")
    if gamma.dim() != 1 or beta.dim() != 1:
        raise ValueError("colwise norm affine expects 1D vectors")
    if x.shape[1] != gamma.shape[0] or x.shape[1] != beta.shape[0]:
        raise ValueError("colwise norm affine shape mismatch")
    n_rows, n_cols = x.shape
    mean = torch.empty((n_cols,), device=x.device, dtype=x.dtype)
    var = torch.empty((n_cols,), device=x.device, dtype=x.dtype)
    inv_std = torch.empty((n_cols,), device=x.device, dtype=x.dtype)
    x_hat = torch.empty_like(x)
    out = torch.empty_like(x)
    if n_rows == 0 or n_cols == 0:
        return mean, var, inv_std, x_hat, out
    _colwise_norm_affine_fused_kernel[(n_cols,)](
        x,
        gamma,
        beta,
        mean,
        var,
        inv_std,
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
        eps,
        BLOCK_M=_COL_SUM_BLOCK_M,
    )
    return mean, var, inv_std, x_hat, out


__all__ = [
    "_colwise_norm",
    "_colwise_norm_affine",
    "_colwise_norm_kernel",
    "_colwise_norm_affine_kernel",
    "_colwise_norm_affine_fused",
    "_colwise_norm_affine_fused_kernel",
]
