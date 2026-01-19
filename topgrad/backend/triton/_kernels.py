import torch
import triton
import triton.language as tl

_BLOCK_SIZE = 1024
_MATMUL_BLOCK_M = 128
_MATMUL_BLOCK_N = 128
_MATMUL_BLOCK_K = 32
_MATMUL_GROUP_M = 8
_LOGSOFTMAX_BLOCK_N = 1024
_BIAS_BLOCK_M = 128
_BIAS_BLOCK_N = 128
_COL_SUM_BLOCK_M = 128
_IM2COL_BLOCK_M = 64
_IM2COL_BLOCK_N = 64
_COL2IM_BLOCK_M = 64
_COL2IM_BLOCK_N = 64


@triton.jit
def _reduce_sum_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    acc = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, acc)


@triton.jit
def _fill_from_scalar_kernel(
    scalar_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    scalar = tl.load(scalar_ptr) * scale
    tl.store(out_ptr + offs, scalar, mask=mask)


@triton.jit
def _fill_const_kernel(out_ptr, n_elements, value, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(out_ptr + offs, value, mask=mask)


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    y = tl.load(y_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def _mul_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    y = tl.load(y_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, x * y, mask=mask)


@triton.jit
def _mul_scalar_kernel(x_ptr, out_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, x * scalar, mask=mask)


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
    out = tl.where(x > 0, grad, 0)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    group_m = tl.minimum(GROUP_M, grid_m)

    group_size = group_m * grid_n
    group_id = pid // group_size
    first_pid_m = group_id * group_m
    pid_in_group = pid % group_size
    pid_m = first_pid_m + (pid_in_group % group_m)
    pid_n = pid_in_group // group_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)


@triton.jit
def _batched_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    group_m = tl.minimum(GROUP_M, grid_m)

    group_size = group_m * grid_n
    group_id = pid // group_size
    first_pid_m = group_id * group_m
    pid_in_group = pid % group_size
    pid_m = first_pid_m + (pid_in_group % group_m)
    pid_n = pid_in_group // group_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_batch_ptr = a_ptr + pid_b * stride_ab
    b_batch_ptr = b_ptr + pid_b * stride_bb
    c_batch_ptr = c_ptr + pid_b * stride_cb

    a_ptrs = a_batch_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_batch_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_batch_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)


@triton.jit
def _bias_add_kernel(
    out_ptr,
    bias_ptr,
    n_rows,
    n_cols,
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

    out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0)
    out = tl.load(out_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0)
    out += bias[None, :]
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _colwise_mul_kernel(
    x_ptr,
    vec_ptr,
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
    vec = tl.load(vec_ptr + offs_n, mask=mask_n, other=0.0)
    out = x * vec[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def _rowwise_mul_kernel(
    x_ptr,
    vec_ptr,
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
    vec = tl.load(vec_ptr + offs_m, mask=mask_m, other=0.0)
    out = x * vec[:, None]
    out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def _rowwise_add_kernel(
    x_ptr,
    vec_ptr,
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
    vec = tl.load(vec_ptr + offs_m, mask=mask_m, other=0.0)
    out = x + vec[:, None]
    out_ptrs = out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
    tl.store(out_ptrs, out, mask=mask)


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
def _col_sum_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
):
    col = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    x_ptrs = x_ptr + offs_m * stride_xm + col * stride_xn

    for m in range(0, n_rows, BLOCK_M):
        m_idx = m + offs_m
        mask = m_idx < n_rows
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        acc += x
        x_ptrs += BLOCK_M * stride_xm

    sum_val = tl.sum(acc, axis=0)
    tl.store(out_ptr + col, sum_val)


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
def _row_sum_kernel(
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

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n in range(0, n_cols, BLOCK_N):
        n_idx = n + offs
        mask = n_idx < n_cols
        x = tl.load(x_row + n_idx * stride_xn, mask=mask, other=0.0)
        acc += x

    sum_val = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, sum_val)


@triton.jit
def _rsqrt_kernel(x_ptr, out_ptr, n_elements, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = 1.0 / tl.sqrt(x + eps)
    tl.store(out_ptr + offs, out, mask=mask)


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

def _numel(shape):
    if isinstance(shape, int):
        return int(shape)
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _require_contiguous(x):
    if not x.is_contiguous():
        raise ValueError("Triton backend expects contiguous tensors.")


def _require_2d(x, name):
    if x.dim() != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(x.shape)}")


def _conv_output_hw(h, w, kh, kw, stride, padding):
    sh, sw = stride
    ph, pw = padding
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1
    if h_out < 0:
        h_out = 0
    if w_out < 0:
        w_out = 0
    return h_out, w_out


def _supports_triton_atomics(device):
    # Triton atomics use acquire/release semantics that require sm70+.
    if device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 7


def _triton_sum(x):
    _require_contiguous(x)
    n_elements = _numel(x.shape)
    if n_elements == 0:
        out = torch.empty((1,), device=x.device, dtype=x.dtype)
        _fill_const_kernel[(1,)](out, 1, 0.0, BLOCK_SIZE=1)
        return out

    current = x
    while True:
        n_elements = _numel(current.shape)
        blocks = triton.cdiv(n_elements, _BLOCK_SIZE)
        out = torch.empty((blocks,), device=current.device, dtype=current.dtype)
        _reduce_sum_kernel[(blocks,)](
            current,
            out,
            n_elements,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
        if blocks == 1:
            return out
        current = out


def _fill_from_scalar(scalar, shape, dtype, device, scale):
    n_elements = _numel(shape)
    out = torch.empty(shape, device=device, dtype=dtype)
    if n_elements == 0:
        return out
    blocks = triton.cdiv(n_elements, _BLOCK_SIZE)
    _fill_from_scalar_kernel[(blocks,)](
        scalar, out, n_elements, scale, BLOCK_SIZE=_BLOCK_SIZE
    )
    return out


def _fill_const(shape, dtype, device, value):
    n_elements = _numel(shape)
    out = torch.empty(shape, device=device, dtype=dtype)
    if n_elements == 0:
        return out
    blocks = triton.cdiv(n_elements, _BLOCK_SIZE)
    _fill_const_kernel[(blocks,)](
        out, n_elements, value, BLOCK_SIZE=_BLOCK_SIZE
    )
    return out


def _elementwise_add(x, y):
    _require_contiguous(x)
    _require_contiguous(y)
    if x.shape != y.shape:
        raise ValueError("elementwise add expects matching shapes")
    n_elements = _numel(x.shape)
    out = torch.empty_like(x)
    if n_elements == 0:
        return out
    blocks = triton.cdiv(n_elements, _BLOCK_SIZE)
    _add_kernel[(blocks,)](x, y, out, n_elements, BLOCK_SIZE=_BLOCK_SIZE)
    return out


def _elementwise_mul(x, y):
    _require_contiguous(x)
    if isinstance(y, torch.Tensor):
        _require_contiguous(y)
        if x.shape != y.shape:
            raise ValueError("elementwise mul expects matching shapes")
        n_elements = _numel(x.shape)
        out = torch.empty_like(x)
        if n_elements == 0:
            return out
        blocks = triton.cdiv(n_elements, _BLOCK_SIZE)
        _mul_kernel[(blocks,)](
            x, y, out, n_elements, BLOCK_SIZE=_BLOCK_SIZE
        )
        return out

    scalar = float(y)
    n_elements = _numel(x.shape)
    out = torch.empty_like(x)
    if n_elements == 0:
        return out
    blocks = triton.cdiv(n_elements, _BLOCK_SIZE)
    _mul_scalar_kernel[(blocks,)](
        x, out, n_elements, scalar, BLOCK_SIZE=_BLOCK_SIZE
    )
    return out


def _relu_forward(x):
    _require_contiguous(x)
    n_elements = _numel(x.shape)
    out = torch.empty_like(x)
    if n_elements == 0:
        return out
    blocks = triton.cdiv(n_elements, _BLOCK_SIZE)
    _relu_forward_kernel[(blocks,)](
        x, out, n_elements, BLOCK_SIZE=_BLOCK_SIZE
    )
    return out


def _relu_backward(x, grad):
    _require_contiguous(x)
    _require_contiguous(grad)
    n_elements = _numel(x.shape)
    out = torch.empty_like(grad)
    if n_elements == 0:
        return out
    blocks = triton.cdiv(n_elements, _BLOCK_SIZE)
    _relu_backward_kernel[(blocks,)](
        x, grad, out, n_elements, BLOCK_SIZE=_BLOCK_SIZE
    )
    return out


def _matmul(a, b):
    _require_2d(a, "matmul lhs")
    _require_2d(b, "matmul rhs")
    if a.shape[1] != b.shape[0]:
        raise ValueError("matmul shape mismatch")
    if a.device != b.device:
        raise ValueError("matmul expects tensors on same device")

    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (
        triton.cdiv(m, _MATMUL_BLOCK_M)
        * triton.cdiv(n, _MATMUL_BLOCK_N),
    )

    _matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=_MATMUL_BLOCK_M,
        BLOCK_N=_MATMUL_BLOCK_N,
        BLOCK_K=_MATMUL_BLOCK_K,
        GROUP_M=_MATMUL_GROUP_M,
        num_warps=4,
        num_stages=2,
    )
    return c


def _batched_matmul(a, b):
    if a.dim() != 3 or b.dim() != 3:
        raise ValueError("batched matmul expects 3D inputs")
    if a.shape[0] != b.shape[0]:
        raise ValueError("batched matmul batch size mismatch")
    if a.shape[2] != b.shape[1]:
        raise ValueError("batched matmul shape mismatch")
    if a.device != b.device:
        raise ValueError("batched matmul expects tensors on same device")

    batch, m, k = a.shape
    _, _, n = b.shape
    out = torch.empty((batch, m, n), device=a.device, dtype=a.dtype)
    if batch == 0 or m == 0 or n == 0:
        return out
    grid = (
        triton.cdiv(m, _MATMUL_BLOCK_M)
        * triton.cdiv(n, _MATMUL_BLOCK_N),
        batch,
    )
    _batched_matmul_kernel[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_M=_MATMUL_BLOCK_M,
        BLOCK_N=_MATMUL_BLOCK_N,
        BLOCK_K=_MATMUL_BLOCK_K,
        GROUP_M=_MATMUL_GROUP_M,
        num_warps=4,
        num_stages=2,
    )
    return out


def _bias_add(out, bias):
    _require_contiguous(out)
    _require_contiguous(bias)
    if bias.dim() != 1:
        raise ValueError("bias must be 1D")
    if out.shape[-1] != bias.shape[0]:
        raise ValueError("bias shape mismatch")

    n_rows, n_cols = out.shape
    grid = (
        triton.cdiv(n_rows, _BIAS_BLOCK_M),
        triton.cdiv(n_cols, _BIAS_BLOCK_N),
    )
    _bias_add_kernel[grid](
        out,
        bias,
        n_rows,
        n_cols,
        out.stride(0),
        out.stride(1),
        BLOCK_M=_BIAS_BLOCK_M,
        BLOCK_N=_BIAS_BLOCK_N,
    )
    return out


def _colwise_mul(x, vec):
    _require_contiguous(x)
    _require_contiguous(vec)
    _require_2d(x, "colwise_mul")
    if vec.dim() != 1:
        raise ValueError("colwise mul expects 1D vector")
    if x.shape[1] != vec.shape[0]:
        raise ValueError("colwise mul shape mismatch")
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    if n_rows == 0 or n_cols == 0:
        return out
    grid = (
        triton.cdiv(n_rows, _BIAS_BLOCK_M),
        triton.cdiv(n_cols, _BIAS_BLOCK_N),
    )
    _colwise_mul_kernel[grid](
        x,
        vec,
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


def _rowwise_mul(x, vec):
    _require_contiguous(x)
    _require_contiguous(vec)
    _require_2d(x, "rowwise_mul")
    if vec.dim() != 1:
        raise ValueError("rowwise mul expects 1D vector")
    if x.shape[0] != vec.shape[0]:
        raise ValueError("rowwise mul shape mismatch")
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    if n_rows == 0 or n_cols == 0:
        return out
    grid = (
        triton.cdiv(n_rows, _BIAS_BLOCK_M),
        triton.cdiv(n_cols, _BIAS_BLOCK_N),
    )
    _rowwise_mul_kernel[grid](
        x,
        vec,
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


def _rowwise_add(x, vec):
    _require_contiguous(x)
    _require_contiguous(vec)
    _require_2d(x, "rowwise_add")
    if vec.dim() != 1:
        raise ValueError("rowwise add expects 1D vector")
    if x.shape[0] != vec.shape[0]:
        raise ValueError("rowwise add shape mismatch")
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    if n_rows == 0 or n_cols == 0:
        return out
    grid = (
        triton.cdiv(n_rows, _BIAS_BLOCK_M),
        triton.cdiv(n_cols, _BIAS_BLOCK_N),
    )
    _rowwise_add_kernel[grid](
        x,
        vec,
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


def _colwise_norm(x, mean, inv_std):
    _require_contiguous(x)
    _require_contiguous(mean)
    _require_contiguous(inv_std)
    _require_2d(x, "colwise_norm")
    if mean.dim() != 1 or inv_std.dim() != 1:
        raise ValueError("colwise norm expects 1D mean and inv_std")
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


def _rowwise_norm(x, mean, inv_std):
    _require_contiguous(x)
    _require_contiguous(mean)
    _require_contiguous(inv_std)
    _require_2d(x, "rowwise_norm")
    if mean.dim() != 1 or inv_std.dim() != 1:
        raise ValueError("rowwise norm expects 1D mean and inv_std")
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


def _col_sum(x):
    _require_contiguous(x)
    _require_2d(x, "col_sum")
    n_rows, n_cols = x.shape
    out = torch.empty((n_cols,), device=x.device, dtype=x.dtype)
    if n_rows == 0:
        return out
    _col_sum_kernel[(n_cols,)](
        x,
        out,
        n_rows,
        n_cols,
        x.stride(0),
        x.stride(1),
        BLOCK_M=_COL_SUM_BLOCK_M,
    )
    return out


def _row_sum(x):
    _require_contiguous(x)
    _require_2d(x, "row_sum")
    n_rows, n_cols = x.shape
    out = torch.empty((n_rows,), device=x.device, dtype=x.dtype)
    if n_cols == 0:
        return out
    _row_sum_kernel[(n_rows,)](
        x,
        out,
        n_rows,
        n_cols,
        x.stride(0),
        x.stride(1),
        BLOCK_N=_LOGSOFTMAX_BLOCK_N,
    )
    return out


def _rsqrt(x, eps):
    _require_contiguous(x)
    n_elements = _numel(x.shape)
    out = torch.empty_like(x)
    if n_elements == 0:
        return out
    blocks = triton.cdiv(n_elements, _BLOCK_SIZE)
    _rsqrt_kernel[(blocks,)](
        x, out, n_elements, eps, BLOCK_SIZE=_BLOCK_SIZE
    )
    return out


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


def _reshape_to_2d(x):
    n_cols = x.shape[-1]
    n_rows = _numel(x.shape[:-1])
    return x.reshape((n_rows, n_cols)), n_rows, n_cols


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
    "_numel",
    "_require_contiguous",
    "_triton_sum",
    "_fill_from_scalar",
    "_fill_const",
    "_elementwise_add",
    "_elementwise_mul",
    "_relu_forward",
    "_relu_backward",
    "_matmul",
    "_batched_matmul",
    "_bias_add",
    "_colwise_mul",
    "_rowwise_mul",
    "_rowwise_add",
    "_colwise_norm",
    "_rowwise_norm",
    "_col_sum",
    "_row_sum",
    "_rsqrt",
    "_im2col",
    "_col2im",
    "_reshape_to_2d",
    "_logsoftmax_forward",
    "_softmax_forward",
    "_logsoftmax_backward",
    "_softmax_backward",
]
