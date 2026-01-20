import math
from types import SimpleNamespace

import numpy as np
import torch

from ..op import Op
from .data import TritonTensor
from .kernels._common import (
    _batched_matmul,
    _bias_add,
    _col_sum,
    _colwise_mul,
    _elementwise_add,
    _elementwise_mul,
    _fill_const,
    _fill_from_scalar,
    _matmul,
    _numel,
    _require_contiguous,
    _reshape_to_2d,
    _row_sum,
    _rowwise_add,
    _rowwise_mul,
    _rsqrt,
    _triton_sum,
)
from .kernels.batchnorm import _colwise_norm_affine, _colwise_norm_affine_fused
from .kernels.conv import _col2im, _im2col
from .kernels.layernorm import _layernorm_backward, _layernorm_fused
from .kernels.linear import _linear
from .kernels.relu import _relu_backward, _relu_forward
from .kernels.softmax import (
    _logsoftmax_backward,
    _logsoftmax_forward,
    _softmax_backward,
    _softmax_forward,
)


class TritonBackend:
    def __init__(self, device="cuda"):
        self.device = device
        self._ops = SimpleNamespace(
            Sum=Sum,
            Mean=Mean,
            Mul=Mul,
            Add=Add,
            ReLU=ReLU,
            LogSoftmax=LogSoftmax,
            Linear=Linear,
            Reshape=Reshape,
            Conv=Conv,
            BatchNorm=BatchNorm,
            LayerNorm=LayerNorm,
            Attention=Attention,
        )

    @property
    def ops(self):
        return self._ops

    def wrap(self, x):
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
            if self.device is not None:
                t = t.to(self.device)
            if not t.is_contiguous():
                t = t.contiguous()
            return TritonTensor(t)
        return x

    def unwrap(self, x: TritonTensor):
        if isinstance(x, TritonTensor):
            disable = getattr(
                torch._C, "DisableTorchDispatch", torch._C._DisableTorchDispatch
            )
            # Bypass torch dispatch to get a base Tensor for numpy conversion.
            with disable():
                raw = x.detach()
                if raw.is_cuda:
                    raw = raw.cpu()
                return raw.numpy()
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def add(self, x, y):
        # Used for optimizer math; keep out of autograd graph.
        return _elementwise_add(x, y)

    def mul(self, x, y):
        # Used for optimizer math; keep out of autograd graph.
        return _elementwise_mul(x, y)


class Sum(Op):
    """scalar reduce for losses or metrics"""

    def forward(self, x):
        raw_x = x
        numel = _numel(raw_x.shape)
        self.save_for_backward(raw_x.shape, raw_x.dtype, numel)
        return _triton_sum(raw_x)

    def backward(self, grad):
        raw_grad = grad
        x_shape, x_dtype, numel = self._intermediate
        return [
            _fill_from_scalar(raw_grad, x_shape, x_dtype, raw_grad.device, scale=1.0)
        ]


class Mean(Op):
    """scalar mean reduce"""

    def forward(self, x):
        raw_x = x
        numel = _numel(raw_x.shape)
        self.save_for_backward(raw_x.shape, raw_x.dtype, numel)
        summed = _triton_sum(raw_x)
        scale = 1.0 / float(numel) if numel else 0.0
        return _fill_from_scalar(summed, (1,), raw_x.dtype, raw_x.device, scale=scale)

    def backward(self, grad):
        raw_grad = grad
        x_shape, x_dtype, numel = self._intermediate
        scale = 1.0 / float(numel) if numel else 0.0
        return [
            _fill_from_scalar(raw_grad, x_shape, x_dtype, raw_grad.device, scale=scale)
        ]


class ReLU(Op):
    """relu nonlinearity"""

    def forward(self, x):
        raw_x = x
        self.save_for_backward(raw_x)
        return _relu_forward(raw_x)

    def backward(self, grad):
        raw_grad = grad
        (raw_x,) = self._intermediate
        return [_relu_backward(raw_x, raw_grad)]


class LogSoftmax(Op):
    """log softmax over last dim for classifiers"""

    def forward(self, x):
        raw_x = x
        y = _logsoftmax_forward(raw_x)
        self.save_for_backward(y)
        return y

    def backward(self, grad):
        raw_grad = grad
        (y,) = self._intermediate
        return [_logsoftmax_backward(raw_grad, y)]


class Add(Op):
    """addition for skip paths"""

    def forward(self, x, y):
        raw_x = x
        raw_y = y
        return _elementwise_add(raw_x, raw_y)

    def backward(self, grad):
        raw_grad = grad
        return [raw_grad, raw_grad]


class Mul(Op):
    """elementwise multiply; use for masking or loss scaling"""

    def forward(self, x, y):
        raw_x = x
        raw_y = y
        self.save_for_backward(raw_x, raw_y)
        return _elementwise_mul(raw_x, raw_y)

    def backward(self, grad):
        raw_grad = grad
        raw_x, raw_y = self._intermediate
        if isinstance(raw_y, torch.Tensor):
            return [
                _elementwise_mul(raw_grad, raw_y),
                _elementwise_mul(raw_grad, raw_x),
            ]

        return [_elementwise_mul(raw_grad, raw_y)]


class Linear(Op):
    """dense layer primitive"""

    def forward(self, x, w, b):
        """x (batch, in_features); w (in_features, out_features); b (out_features,)"""
        raw_x = x
        raw_w = w
        raw_b = b
        self.save_for_backward(raw_x, raw_w)
        return _linear(raw_x, raw_w, raw_b)

    def backward(self, grad):
        raw_grad = grad
        raw_x, raw_w = self._intermediate
        grad_x = _matmul(raw_grad, raw_w.transpose(0, 1))
        grad_w = _matmul(raw_x.transpose(0, 1), raw_grad)
        grad_b = _col_sum(raw_grad)
        return [
            grad_x,
            grad_w,
            grad_b,
        ]


class Reshape(Op):
    """reshape for flatten or view changes"""

    def forward(self, x, ns):
        raw_x = x
        self.save_for_backward(raw_x.shape, ns)
        return raw_x.reshape(ns)

    def backward(self, grad):
        raw_grad = grad
        os, _ = self._intermediate
        return [raw_grad.reshape(os)]


class Conv(Op):
    """nhwc conv with tuple stride/padding via Triton kernels"""

    def forward(self, x, w, b, stride, padding):
        raw_x = x
        raw_w = w
        raw_b = b
        _require_contiguous(raw_x)
        _require_contiguous(raw_w)
        _require_contiguous(raw_b)
        if raw_x.dim() != 4:
            raise ValueError("conv expects 4D nhwc input")
        if raw_w.dim() != 4:
            raise ValueError("conv expects 4D (kh, kw, cin, cout) weights")
        if raw_b.dim() != 1:
            raise ValueError("conv expects 1D bias (cout,)")

        kh, kw, cin, cout = raw_w.shape
        if raw_x.shape[3] != cin:
            raise ValueError("conv input channels do not match weights")
        if raw_b.shape[0] != cout:
            raise ValueError("conv bias shape mismatch")

        x_col, im2col_cache = _im2col(raw_x, kh, kw, stride, padding)
        h_out, w_out = im2col_cache[2]
        n = raw_x.shape[0]
        if x_col.numel() == 0:
            out = torch.empty(
                (n, h_out, w_out, cout), device=raw_x.device, dtype=raw_x.dtype
            )
            self.save_for_backward(x_col, raw_w, im2col_cache)
            return out

        w_col = raw_w.reshape(kh * kw * cin, cout)
        out_cols = _matmul(x_col, w_col)
        out_cols = _bias_add(out_cols, raw_b)
        out = out_cols.reshape(n, h_out, w_out, cout)
        self.save_for_backward(x_col, raw_w, im2col_cache)
        return out

    def backward(self, grad):
        raw_grad = grad
        _require_contiguous(raw_grad)
        x_col, raw_w, im2col_cache = self._intermediate
        (n, h, w, c), (kh, kw, cin), (h_out, w_out), _, _ = im2col_cache
        cout = raw_w.shape[3]
        grad_col = raw_grad.reshape(n * h_out * w_out, cout)

        if grad_col.numel() == 0:
            dx = _fill_const((n, h, w, c), raw_grad.dtype, raw_grad.device, 0.0)
            dw = _fill_const(raw_w.shape, raw_w.dtype, raw_w.device, 0.0)
            db = _fill_const((cout,), raw_grad.dtype, raw_grad.device, 0.0)
            return [
                dx,
                dw,
                db,
            ]

        dw_col = _matmul(x_col.transpose(0, 1), grad_col)
        dw = dw_col.reshape(kh, kw, cin, cout)
        db = _col_sum(grad_col)

        w_col = raw_w.reshape(kh * kw * cin, cout)
        dcols = _matmul(grad_col, w_col.transpose(0, 1))
        dx = _col2im(dcols, im2col_cache)
        return [
            dx,
            dw,
            db,
        ]


class BatchNorm(Op):
    """batch norm with running stats for conv/mlp via Triton kernels"""

    def forward(self, x, gamma, beta, state, eps=1e-5, momentum=0.1, mode="T"):
        assert mode in {"T", "E"}, "BatchNorm model can only be either train or eval."
        raw_x = x
        raw_gamma = gamma
        raw_beta = beta
        _require_contiguous(raw_x)
        _require_contiguous(raw_gamma)
        _require_contiguous(raw_beta)
        if raw_gamma.dim() != 1 or raw_beta.dim() != 1:
            raise ValueError("batchnorm expects 1D gamma/beta")
        if (
            raw_x.shape[-1] != raw_gamma.shape[0]
            or raw_x.shape[-1] != raw_beta.shape[0]
        ):
            raise ValueError("batchnorm gamma/beta shape mismatch")

        x2d, n_rows, _ = _reshape_to_2d(raw_x)

        if mode == "T":
            mean, var, inv_std, x_hat, y2d = _colwise_norm_affine_fused(
                x2d, raw_gamma, raw_beta, eps
            )
            if state is not None:
                state.running_mean = _elementwise_add(
                    _elementwise_mul(state.running_mean, 1.0 - momentum),
                    _elementwise_mul(mean, momentum),
                )
                state.running_var = _elementwise_add(
                    _elementwise_mul(state.running_var, 1.0 - momentum),
                    _elementwise_mul(var, momentum),
                )
        elif mode == "E":
            if state is None:
                raise ValueError("BatchNorm state is required for eval mode.")
            mean = state.running_mean
            var = state.running_var

        if mode == "E":
            inv_std = _rsqrt(var, eps)
            x_hat, y2d = _colwise_norm_affine(x2d, mean, inv_std, raw_gamma, raw_beta)
        y = y2d.reshape(raw_x.shape)
        self.save_for_backward(x_hat, inv_std, raw_gamma, raw_x.shape, mode)
        return y

    def backward(self, grad):
        raw_grad = grad
        _require_contiguous(raw_grad)
        x_hat, inv_std, gamma, os, mode = self._intermediate
        grad2d, n_rows, _ = _reshape_to_2d(raw_grad)
        x_hat2d = x_hat.reshape(grad2d.shape)
        inv_rows = 1.0 / float(n_rows) if n_rows else 0.0

        dbeta = _col_sum(grad2d)
        dgamma = _col_sum(_elementwise_mul(grad2d, x_hat2d))

        g_hat = _colwise_mul(grad2d, gamma)
        if mode == "T":
            m1 = _elementwise_mul(_col_sum(g_hat), inv_rows)
            m2 = _elementwise_mul(_col_sum(_elementwise_mul(g_hat, x_hat2d)), inv_rows)
            glocal = _colwise_mul(g_hat, inv_std)
            mean_correction = _elementwise_mul(inv_std, m1)
            neg_mean = _elementwise_mul(mean_correction, -1.0)
            dx2d = _bias_add(glocal, neg_mean)
            var_scale = _elementwise_mul(inv_std, m2)
            var_correction = _colwise_mul(x_hat2d, var_scale)
            dx2d = _elementwise_add(dx2d, _elementwise_mul(var_correction, -1.0))
        else:
            dx2d = _colwise_mul(g_hat, inv_std)

        return [
            dx2d.reshape(os),
            dgamma,
            dbeta,
        ]


class LayerNorm(Op):
    """layer norm across feature dim via Triton kernels"""

    def forward(self, x, gamma, beta, eps=1e-5):
        raw_x = x
        raw_gamma = gamma
        raw_beta = beta
        _require_contiguous(raw_x)
        _require_contiguous(raw_gamma)
        _require_contiguous(raw_beta)
        if raw_gamma.dim() != 1 or raw_beta.dim() != 1:
            raise ValueError("layernorm expects 1D gamma/beta")
        if (
            raw_x.shape[-1] != raw_gamma.shape[0]
            or raw_x.shape[-1] != raw_beta.shape[0]
        ):
            raise ValueError("layernorm gamma/beta shape mismatch")

        x2d, _, n_cols = _reshape_to_2d(raw_x)
        x_hat, inv_std, y2d = _layernorm_fused(x2d, raw_gamma, raw_beta, eps)
        y = y2d.reshape(raw_x.shape)
        self.save_for_backward(x_hat, inv_std, raw_gamma, raw_x.shape)
        return y

    def backward(self, grad):
        raw_grad = grad
        _require_contiguous(raw_grad)
        x_hat, inv_std, gamma, os = self._intermediate
        grad2d, n_rows, n_cols = _reshape_to_2d(raw_grad)
        x_hat2d = x_hat.reshape(grad2d.shape)
        dx2d, dgamma, dbeta = _layernorm_backward(
            grad2d, x_hat2d, inv_std, gamma
        )

        return [
            dx2d.reshape(os),
            dgamma,
            dbeta,
        ]


class Attention(Op):
    """attention block using Triton kernels"""

    def forward(self, x, wq, wk, wv):
        raw_x = x
        raw_wq = wq
        raw_wk = wk
        raw_wv = wv
        _require_contiguous(raw_x)
        _require_contiguous(raw_wq)
        _require_contiguous(raw_wk)
        _require_contiguous(raw_wv)
        if raw_x.dim() != 3:
            raise ValueError("attention expects 3D input (batch, tokens, dim)")
        if raw_wq.dim() != 2 or raw_wk.dim() != 2 or raw_wv.dim() != 2:
            raise ValueError("attention expects 2D weight matrices")

        batch, tokens, dim = raw_x.shape
        if raw_wq.shape != (dim, dim):
            raise ValueError("attention wq shape mismatch")
        if raw_wk.shape != (dim, dim):
            raise ValueError("attention wk shape mismatch")
        if raw_wv.shape != (dim, dim):
            raise ValueError("attention wv shape mismatch")

        x2d = raw_x.reshape((-1, dim))
        q = _matmul(x2d, raw_wq).reshape((batch, tokens, dim))
        k = _matmul(x2d, raw_wk).reshape((batch, tokens, dim))
        v = _matmul(x2d, raw_wv).reshape((batch, tokens, dim))

        scale = 1.0 / math.sqrt(dim) if dim else 1.0
        scores = _batched_matmul(q, k.transpose(1, 2))
        scores = _elementwise_mul(scores, scale)
        attn = _softmax_forward(scores)
        y = _batched_matmul(attn, v)
        self.save_for_backward(raw_x, raw_wq, raw_wk, raw_wv, q, k, v, attn)
        return y

    def backward(self, grad):
        raw_grad = grad
        _require_contiguous(raw_grad)
        raw_x, raw_wq, raw_wk, raw_wv, q, k, v, attn = self._intermediate
        _, _, dim = raw_x.shape

        dv = _batched_matmul(attn.transpose(1, 2), raw_grad)
        dw_attn = _batched_matmul(raw_grad, v.transpose(1, 2))
        dw_scores = _softmax_backward(dw_attn, attn)
        scale = 1.0 / math.sqrt(dim) if dim else 1.0
        dw_scores = _elementwise_mul(dw_scores, scale)

        dq = _batched_matmul(dw_scores, k)
        dk = _batched_matmul(dw_scores.transpose(1, 2), q)

        dq2d = dq.reshape((-1, dim))
        dk2d = dk.reshape((-1, dim))
        dv2d = dv.reshape((-1, dim))
        x2d = raw_x.reshape((-1, dim))

        dx2d = _elementwise_add(
            _matmul(dq2d, raw_wq.transpose(0, 1)),
            _matmul(dk2d, raw_wk.transpose(0, 1)),
        )
        dx2d = _elementwise_add(dx2d, _matmul(dv2d, raw_wv.transpose(0, 1)))
        dx = dx2d.reshape(raw_x.shape)

        dwq = _matmul(x2d.transpose(0, 1), dq2d)
        dwk = _matmul(x2d.transpose(0, 1), dk2d)
        dwv = _matmul(x2d.transpose(0, 1), dv2d)
        return [
            dx,
            dwq,
            dwk,
            dwv,
        ]


__all__ = [
    "TritonBackend",
    "TritonTensor",
]
