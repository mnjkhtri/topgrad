import torch
import triton
import triton.language as tl

from ._common import (
    _COL2IM_BLOCK_M,
    _COL2IM_BLOCK_N,
    _IM2COL_BLOCK_M,
    _IM2COL_BLOCK_N,
    _fill_const,
    _require_contiguous,
    _supports_triton_atomics,
)


def _conv_output_hw(h, w, kh, kw, stride, padding):
    # Compute output height/width for a 2D convolution.
    sh, sw = stride
    ph, pw = padding
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1
    if h_out < 0:
        h_out = 0
    if w_out < 0:
        w_out = 0
    return h_out, w_out


@triton.jit
def _im2col_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    h,
    w,
    c,
    h_out,
    w_out,
    kh,
    kw,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    stride_xn,
    stride_xh,
    stride_xw,
    stride_xc,
    stride_outm,
    stride_outn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = offs_m[:, None]
    cols = offs_n[None, :]
    mask_m = offs_m < n_rows
    mask_n = offs_n < n_cols

    hw_out = h_out * w_out
    n_idx = rows // hw_out
    rem_hw = rows - n_idx * hw_out
    h_out_idx = rem_hw // w_out
    w_out_idx = rem_hw - h_out_idx * w_out

    kwc = kw * c
    kh_idx = cols // kwc
    rem_k = cols - kh_idx * kwc
    kw_idx = rem_k // c
    c_idx = rem_k - kw_idx * c

    h_in = h_out_idx * stride_h + kh_idx - pad_h
    w_in = w_out_idx * stride_w + kw_idx - pad_w

    mask_hw = (h_in >= 0) & (h_in < h) & (w_in >= 0) & (w_in < w)
    mask = mask_m[:, None] & mask_n[None, :] & mask_hw

    x_ptrs = (
        x_ptr
        + n_idx * stride_xn
        + h_in * stride_xh
        + w_in * stride_xw
        + c_idx * stride_xc
    )
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    out_ptrs = out_ptr + rows * stride_outm + cols * stride_outn
    tl.store(out_ptrs, x, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _col2im_kernel(
    col_ptr,
    dx_ptr,
    n_rows,
    n_cols,
    h,
    w,
    c,
    h_out,
    w_out,
    kh,
    kw,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    stride_colm,
    stride_coln,
    stride_xn,
    stride_xh,
    stride_xw,
    stride_xc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = offs_m[:, None]
    cols = offs_n[None, :]
    mask_m = offs_m < n_rows
    mask_n = offs_n < n_cols
    mask_col = mask_m[:, None] & mask_n[None, :]

    hw_out = h_out * w_out
    n_idx = rows // hw_out
    rem_hw = rows - n_idx * hw_out
    h_out_idx = rem_hw // w_out
    w_out_idx = rem_hw - h_out_idx * w_out

    kwc = kw * c
    kh_idx = cols // kwc
    rem_k = cols - kh_idx * kwc
    kw_idx = rem_k // c
    c_idx = rem_k - kw_idx * c

    h_in = h_out_idx * stride_h + kh_idx - pad_h
    w_in = w_out_idx * stride_w + kw_idx - pad_w

    mask_hw = (h_in >= 0) & (h_in < h) & (w_in >= 0) & (w_in < w)
    mask = mask_col & mask_hw

    col_ptrs = col_ptr + rows * stride_colm + cols * stride_coln
    vals = tl.load(col_ptrs, mask=mask_col, other=0.0)
    dx_ptrs = (
        dx_ptr
        + n_idx * stride_xn
        + h_in * stride_xh
        + w_in * stride_xw
        + c_idx * stride_xc
    )
    tl.atomic_add(dx_ptrs, vals, mask=mask)


def _im2col(x, kh, kw, stride, padding):
    _require_contiguous(x)
    n, h, w, c = x.shape
    sh, sw = stride
    ph, pw = padding
    h_out, w_out = _conv_output_hw(h, w, kh, kw, stride, padding)
    n_rows = n * h_out * w_out
    n_cols = kh * kw * c
    out = torch.empty((n_rows, n_cols), device=x.device, dtype=x.dtype)
    cache = (x.shape, (kh, kw, c), (h_out, w_out), stride, padding)
    if n_rows == 0 or n_cols == 0:
        return out, cache

    grid = (
        triton.cdiv(n_rows, _IM2COL_BLOCK_M),
        triton.cdiv(n_cols, _IM2COL_BLOCK_N),
    )
    _im2col_kernel[grid](
        x,
        out,
        n_rows,
        n_cols,
        h,
        w,
        c,
        h_out,
        w_out,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        out.stride(0),
        out.stride(1),
        BLOCK_M=_IM2COL_BLOCK_M,
        BLOCK_N=_IM2COL_BLOCK_N,
        num_warps=4,
    )
    return out, cache


def _col2im_fallback(col, cache):
    _require_contiguous(col)
    (n, h, w, c), (kh, kw, _), (h_out, w_out), stride, padding = cache
    sh, sw = stride
    ph, pw = padding
    n_rows = n * h_out * w_out
    n_cols = kh * kw * c
    if n_rows == 0 or n_cols == 0:
        return torch.zeros((n, h, w, c), device=col.device, dtype=col.dtype)

    col_view = col.reshape(n, h_out, w_out, kh, kw, c)
    dx_p = torch.zeros(
        (n, h + 2 * ph, w + 2 * pw, c), device=col.device, dtype=col.dtype
    )
    for i in range(kh):
        i_max = i + sh * h_out
        for j in range(kw):
            j_max = j + sw * w_out
            dx_p[:, i:i_max:sh, j:j_max:sw, :].add_(col_view[:, :, :, i, j, :])

    if ph == 0 and pw == 0:
        return dx_p
    return dx_p[:, ph : ph + h, pw : pw + w, :].contiguous()


def _col2im(col, cache):
    _require_contiguous(col)
    (n, h, w, c), (kh, kw, _), (h_out, w_out), stride, padding = cache
    sh, sw = stride
    ph, pw = padding
    n_rows = n * h_out * w_out
    n_cols = kh * kw * c
    if n_rows == 0 or n_cols == 0:
        return torch.zeros((n, h, w, c), device=col.device, dtype=col.dtype)
    if not _supports_triton_atomics(col.device):
        return _col2im_fallback(col, cache)

    out = _fill_const((n, h, w, c), col.dtype, col.device, 0.0)
    grid = (
        triton.cdiv(n_rows, _COL2IM_BLOCK_M),
        triton.cdiv(n_cols, _COL2IM_BLOCK_N),
    )
    _col2im_kernel[grid](
        col,
        out,
        n_rows,
        n_cols,
        h,
        w,
        c,
        h_out,
        w_out,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        col.stride(0),
        col.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_M=_COL2IM_BLOCK_M,
        BLOCK_N=_COL2IM_BLOCK_N,
        num_warps=4,
    )
    return out


__all__ = [
    "_im2col_kernel",
    "_col2im_kernel",
    "_im2col",
    "_col2im_fallback",
    "_col2im",
]
