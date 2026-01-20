import torch
import triton
import triton.language as tl

_BLOCK_SIZE = 1024
_MATMUL_BLOCK_M = 128
_MATMUL_BLOCK_N = 128
_MATMUL_BLOCK_K = 32
_MATMUL_GROUP_M = 8
_LOGSOFTMAX_BLOCK_N = 1024
_NORM_BLOCK_N = 1024
_BIAS_BLOCK_M = 128
_BIAS_BLOCK_N = 128
_COL_SUM_BLOCK_M = 128
_IM2COL_BLOCK_M = 64
_IM2COL_BLOCK_N = 64
_COL2IM_BLOCK_M = 64
_COL2IM_BLOCK_N = 64


def _numel(shape):
    # Compute total element count for an int or iterable shape.
    if isinstance(shape, int):
        return int(shape)
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _require_contiguous(x):
    # Triton kernels assume contiguous layouts for simple pointer arithmetic.
    if not x.is_contiguous():
        raise ValueError("Triton backend expects contiguous tensors.")


def _require_2d(x, name):
    # Guard for kernels that operate on 2D matrices.
    if x.dim() != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(x.shape)}")


def _supports_triton_atomics(device):
    # Triton atomics use acquire/release semantics that require sm70+.
    if device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 7


def _reshape_to_2d(x):
    # Collapse all but the last dimension into a 2D matrix view.
    n_cols = x.shape[-1]
    n_rows = _numel(x.shape[:-1])
    return x.reshape((n_rows, n_cols)), n_rows, n_cols


@triton.jit
def _fill_from_scalar_kernel(
    scalar_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr
):
    # Fill a flat buffer with a scaled scalar value.
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    scalar = tl.load(scalar_ptr) * scale
    tl.store(out_ptr + offs, scalar, mask=mask)


@triton.jit
def _fill_const_kernel(out_ptr, n_elements, value, BLOCK_SIZE: tl.constexpr):
    # Fill a flat buffer with a constant value.
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(out_ptr + offs, value, mask=mask)


@triton.jit
def _reduce_sum_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Reduce each block to a single sum value (one output per program).
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    acc = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, acc)


def _triton_sum(x):
    # Sum all elements using iterative block reductions.
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
        # Reduce in chunks until a single block remains.
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
    # Materialize a tensor filled from a single scalar (with optional scale).
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
    # Materialize a tensor filled with a constant value.
    n_elements = _numel(shape)
    out = torch.empty(shape, device=device, dtype=dtype)
    if n_elements == 0:
        return out
    blocks = triton.cdiv(n_elements, _BLOCK_SIZE)
    _fill_const_kernel[(blocks,)](out, n_elements, value, BLOCK_SIZE=_BLOCK_SIZE)
    return out


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
    # Add a 1D bias to each row of a 2D output tile.
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
    # Multiply each column by a vector (broadcast along rows).
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
    # Multiply each row by a vector (broadcast along columns).
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
    # Add a per-row vector to each row.
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


def _bias_add(out, bias):
    # In-place row-wise bias add for 2D tensors.
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
    # Return x * vec where vec broadcasts over columns.
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
    # Return x * vec where vec broadcasts over rows.
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
    # Return x + vec where vec broadcasts over rows.
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


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Elementwise add over a flat buffer.
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    y = tl.load(y_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def _mul_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Elementwise multiply over a flat buffer.
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    y = tl.load(y_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, x * y, mask=mask)


@triton.jit
def _mul_scalar_kernel(x_ptr, out_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr):
    # Multiply a flat buffer by a scalar.
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, x * scalar, mask=mask)


def _elementwise_add(x, y):
    # Elementwise add for identically-shaped tensors.
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
    # Elementwise multiply supporting tensor or scalar RHS.
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
        _mul_kernel[(blocks,)](x, y, out, n_elements, BLOCK_SIZE=_BLOCK_SIZE)
        return out

    scalar = float(y)
    n_elements = _numel(x.shape)
    out = torch.empty_like(x)
    if n_elements == 0:
        return out
    blocks = triton.cdiv(n_elements, _BLOCK_SIZE)
    _mul_scalar_kernel[(blocks,)](x, out, n_elements, scalar, BLOCK_SIZE=_BLOCK_SIZE)
    return out


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
    # Sum each column of a 2D matrix.
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
def _row_sum_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_xm,
    stride_xn,
    BLOCK_N: tl.constexpr,
):
    # Sum each row of a 2D matrix.
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


def _col_sum(x):
    # Column-wise sum (returns length-n_cols vector).
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
    # Row-wise sum (returns length-n_rows vector).
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
    # Blocked GEMM for 2D matrices (C = A @ B).
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

    # Accumulate in fp32 for better numerical stability.
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
    # Blocked GEMM for batched 3D tensors.
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

    # Accumulate per-batch tile in fp32.
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


def _matmul(a, b):
    # Convenience wrapper for 2D matmul kernel launch.
    _require_2d(a, "matmul lhs")
    _require_2d(b, "matmul rhs")
    if a.shape[1] != b.shape[0]:
        raise ValueError("matmul shape mismatch")
    if a.device != b.device:
        raise ValueError("matmul expects tensors on same device")

    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(m, _MATMUL_BLOCK_M) * triton.cdiv(n, _MATMUL_BLOCK_N),)

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
    # Convenience wrapper for 3D batched matmul kernel launch.
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
        triton.cdiv(m, _MATMUL_BLOCK_M) * triton.cdiv(n, _MATMUL_BLOCK_N),
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


@triton.jit
def _rsqrt_kernel(x_ptr, out_ptr, n_elements, eps, BLOCK_SIZE: tl.constexpr):
    # Compute reciprocal sqrt elementwise with epsilon.
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    out = 1.0 / tl.sqrt(x + eps)
    tl.store(out_ptr + offs, out, mask=mask)


def _rsqrt(x, eps):
    # Reciprocal sqrt over a tensor.
    _require_contiguous(x)
    n_elements = _numel(x.shape)
    out = torch.empty_like(x)
    if n_elements == 0:
        return out
    blocks = triton.cdiv(n_elements, _BLOCK_SIZE)
    _rsqrt_kernel[(blocks,)](x, out, n_elements, eps, BLOCK_SIZE=_BLOCK_SIZE)
    return out


__all__ = [
    "_BLOCK_SIZE",
    "_MATMUL_BLOCK_M",
    "_MATMUL_BLOCK_N",
    "_MATMUL_BLOCK_K",
    "_MATMUL_GROUP_M",
    "_LOGSOFTMAX_BLOCK_N",
    "_BIAS_BLOCK_M",
    "_BIAS_BLOCK_N",
    "_COL_SUM_BLOCK_M",
    "_IM2COL_BLOCK_M",
    "_IM2COL_BLOCK_N",
    "_COL2IM_BLOCK_M",
    "_COL2IM_BLOCK_N",
    "_numel",
    "_require_contiguous",
    "_require_2d",
    "_supports_triton_atomics",
    "_reshape_to_2d",
    "_elementwise_add",
    "_elementwise_mul",
    "_add_kernel",
    "_mul_kernel",
    "_mul_scalar_kernel",
    "_col_sum",
    "_row_sum",
    "_col_sum_kernel",
    "_row_sum_kernel",
    "_matmul",
    "_batched_matmul",
    "_matmul_kernel",
    "_batched_matmul_kernel",
    "_fill_from_scalar",
    "_fill_const",
    "_fill_from_scalar_kernel",
    "_fill_const_kernel",
    "_bias_add",
    "_colwise_mul",
    "_rowwise_mul",
    "_rowwise_add",
    "_bias_add_kernel",
    "_colwise_mul_kernel",
    "_rowwise_mul_kernel",
    "_rowwise_add_kernel",
    "_rsqrt",
    "_rsqrt_kernel",
]
