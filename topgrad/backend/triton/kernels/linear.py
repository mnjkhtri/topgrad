import torch
import triton
import triton.language as tl

from ._common import (
    _MATMUL_BLOCK_K,
    _MATMUL_BLOCK_M,
    _MATMUL_BLOCK_N,
    _MATMUL_GROUP_M,
    _require_2d,
)


@triton.jit
def _linear_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias,
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

    bias = tl.load(bias_ptr + offs_n * stride_bias, mask=offs_n < N, other=0.0)
    accumulator += bias[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)


def _linear(a, b, bias):
    _require_2d(a, "matmul lhs")
    _require_2d(b, "matmul rhs")
    if bias.dim() != 1:
        raise ValueError("bias must be 1D")
    if a.shape[1] != b.shape[0]:
        raise ValueError("matmul shape mismatch")
    if b.shape[1] != bias.shape[0]:
        raise ValueError("bias shape mismatch")
    if a.device != b.device or a.device != bias.device:
        raise ValueError("matmul expects tensors on same device")

    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if m == 0 or n == 0:
        return c
    grid = (triton.cdiv(m, _MATMUL_BLOCK_M) * triton.cdiv(n, _MATMUL_BLOCK_N),)

    _linear_kernel[grid](
        a,
        b,
        bias,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        bias.stride(0),
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


__all__ = [
    "_linear_kernel",
    "_linear",
]
