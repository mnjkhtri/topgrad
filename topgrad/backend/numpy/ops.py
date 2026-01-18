from types import SimpleNamespace

import numpy as np

from ..op import Op


class NumpyBackend:
    def __init__(self):
        # Define preferred types for this backend
        self.float = np.float32
        self.int = np.int32
        self._ops = SimpleNamespace(
            Sum=Sum,
            Mean=Mean,
            ReLU=ReLU,
            LogSoftmax=LogSoftmax,
            Mul=Mul,
            Add=Add,
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
        return x

    def unwrap(self, x):
        return x

    def add(self, x, y):
        # Used for optimizer math; keep out of autograd graph.
        return np.add(x, y)

    def mul(self, x, y):
        # Used for optimizer math; keep out of autograd graph.
        return np.multiply(x, y)


class Sum(Op):
    """scalar reduce for losses or metrics"""

    def forward(self, x):
        raw_x = x
        self.save_for_backward(raw_x.shape, raw_x.dtype)
        return np.sum(raw_x)

    def backward(self, grad):
        raw_grad = grad
        x_shape, x_dtype = self._intermediate
        return [np.full(x_shape, raw_grad, dtype=x_dtype)]


class Mean(Op):
    """scalar mean reduce"""

    def forward(self, x):
        raw_x = x
        self.save_for_backward(raw_x.shape, raw_x.dtype)
        return np.mean(raw_x)

    def backward(self, grad):
        raw_grad = grad
        x_shape, x_dtype = self._intermediate
        numel = 1
        for dim in x_shape:
            numel *= int(dim)
        return [np.full(x_shape, raw_grad / numel, dtype=x_dtype)]


class Mul(Op):
    """elementwise multiply; use for masking or loss scaling"""

    def forward(self, x, y):
        raw_x = x
        raw_y = y
        self.save_for_backward(raw_x, raw_y)
        return raw_x * raw_y

    def backward(self, grad):
        raw_grad = grad
        raw_x, raw_y = self._intermediate
        return [
            raw_grad * raw_y,
            raw_grad * raw_x,
        ]


class ReLU(Op):
    """relu nonlinearity"""

    def forward(self, x):
        raw_x = x
        self.save_for_backward(raw_x)
        return np.maximum(raw_x, 0.0)

    def backward(self, grad):
        raw_grad = grad
        (raw_x,) = self._intermediate
        grad_x = raw_grad.copy()
        grad_x[raw_x < 0] = 0
        return [grad_x]


class LogSoftmax(Op):
    """log softmax over last dim for classifiers"""

    def forward(self, x):
        raw_x = x
        axis = -1
        max_x = np.max(raw_x, axis=axis, keepdims=True)
        log_sum_exp = max_x + np.log(
            np.sum(np.exp(raw_x - max_x), axis=axis, keepdims=True)
        )
        y = raw_x - log_sum_exp
        self.save_for_backward(y)
        return y

    def backward(self, grad):
        raw_grad = grad
        axis = -1
        (y,) = self._intermediate
        softmax_y = np.exp(y)
        sum_grad = np.sum(raw_grad, axis=axis, keepdims=True)
        grad_x = raw_grad - softmax_y * sum_grad
        return [grad_x]


class Linear(Op):
    """dense layer primitive"""

    def forward(self, x, w, b):
        """x (..., in_features); w (in_features, out_features); b (out_features,)"""
        raw_x = x
        raw_w = w
        raw_b = b
        self.save_for_backward(raw_x, raw_w)
        return raw_x @ raw_w + raw_b

    def backward(self, grad):
        raw_grad = grad
        raw_x, raw_w = self._intermediate
        in_features, out_features = raw_w.shape[0], raw_w.shape[1]
        grad_x = raw_grad @ raw_w.T
        grad_w = raw_x.reshape(-1, in_features).T @ raw_grad.reshape(-1, out_features)
        grad_b = raw_grad.reshape(-1, raw_grad.shape[-1]).sum(axis=0)
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


class Add(Op):
    """addition for skip paths"""

    def forward(self, x, y):
        raw_x = x
        raw_y = y
        return raw_x + raw_y

    def backward(self, grad):
        raw_grad = grad
        return [raw_grad, raw_grad]


class Conv(Op):
    """nhwc conv with tuple stride/padding"""

    @staticmethod
    def _im2col(x, kh, kw, stride, pad):
        """im2col for nhwc -> (n*h_out*w_out, kh*kw*c)"""
        n, h, w, c = x.shape
        sh, sw = stride
        ph, pw = pad

        h_out = (h + 2 * ph - kh) // sh + 1
        w_out = (w + 2 * pw - kw) // sw + 1

        x_p = np.pad(x, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode="constant")

        x_col = np.zeros((n, h_out, w_out, kh, kw, c), dtype=x.dtype)
        for i in range(kh):
            i_max = i + sh * h_out
            for j in range(kw):
                j_max = j + sw * w_out
                x_col[:, :, :, i, j, :] = x_p[:, i:i_max:sh, j:j_max:sw, :]

        x_col = x_col.reshape(n * h_out * w_out, kh * kw * c)
        cache = (x.shape, (kh, kw, c), (h_out, w_out), stride, pad)
        return x_col, cache

    @staticmethod
    def _col2im(col, cache):
        """reverse im2col to rebuild dx"""
        (n, h, w, c), (kh, kw, _), (h_out, w_out), stride, pad = cache
        sh, sw = stride
        ph, pw = pad

        col = col.reshape(n, h_out, w_out, kh, kw, c)

        dx_p = np.zeros((n, h + 2 * ph, w + 2 * pw, c), dtype=col.dtype)
        for i in range(kh):
            i_max = i + sh * h_out
            for j in range(kw):
                j_max = j + sw * w_out
                dx_p[:, i:i_max:sh, j:j_max:sw, :] += col[:, :, :, i, j, :]

        if ph == 0 and pw == 0:
            return dx_p
        return dx_p[:, ph : ph + h, pw : pw + w, :]

    def forward(self, x, w, b, stride, padding):
        """x (n, h, w, cin); w (kh, kw, cin, cout); b (cout,)"""
        raw_x = x
        raw_w = w
        raw_b = b
        kh, kw, cin, cout = raw_w.shape
        x_col, im2col_cache = self._im2col(raw_x, kh, kw, stride, padding)
        w_col = raw_w.reshape(kh * kw * cin, cout)
        out_cols = x_col @ w_col
        out_cols += raw_b
        n, h, w, _ = raw_x.shape
        ph, pw = padding
        sh, sw = stride
        h_out = (h + 2 * ph - kh) // sh + 1
        w_out = (w + 2 * pw - kw) // sw + 1
        out = out_cols.reshape(n, h_out, w_out, cout)

        self.save_for_backward(x_col, raw_w, im2col_cache)
        return out

    def backward(self, grad):
        raw_grad = grad
        x_col, raw_w, im2col_cache = self._intermediate
        kh, kw, cin, cout = raw_w.shape

        n, h_out, w_out, _ = raw_grad.shape
        grad_col = raw_grad.reshape(n * h_out * w_out, cout)

        dw_col = x_col.T @ grad_col
        dw = dw_col.reshape(kh, kw, cin, cout)

        db = grad_col.sum(axis=0)

        w_col = raw_w.reshape(kh * kw * cin, cout)
        dcols = grad_col @ w_col.T
        dx = self._col2im(dcols, im2col_cache)

        return [
            dx,
            dw,
            db,
        ]


class BatchNorm(Op):
    """batch norm with running stats for conv/mlp"""

    def forward(self, x, gamma, beta, state, eps=1e-5, momentum=0.1, mode="T"):
        assert mode in {"T", "E"}, "BatchNorm model can only be either train or eval."
        raw_x = x
        raw_gamma = gamma
        raw_beta = beta
        x_flat = raw_x.reshape(-1, raw_x.shape[-1])

        if mode == "T":
            mean = x_flat.mean(axis=0)
            var = x_flat.var(axis=0)
            running_mean = (1 - momentum) * state.running_mean + momentum * mean
            running_var = (1 - momentum) * state.running_var + momentum * var
            state.running_mean = running_mean
            state.running_var = running_var
        elif mode == "E":
            mean = state.running_mean
            var = state.running_var
        else:
            raise ValueError(f"unknown mode {mode}")

        inv_std = 1.0 / np.sqrt(var + eps)
        x_hat_flat = (x_flat - mean) * inv_std
        y_flat = raw_gamma * x_hat_flat + raw_beta
        y = y_flat.reshape(raw_x.shape)
        self.save_for_backward(x_hat_flat, inv_std, raw_gamma, raw_x.shape, mode)
        return y

    def backward(self, grad):
        raw_grad = grad
        x_hat_flat, inv_std, raw_gamma, os, mode = self._intermediate
        grad_flat = raw_grad.reshape(-1, os[-1])
        g_hat = raw_gamma * grad_flat
        m1 = g_hat.mean(axis=0, keepdims=True)
        m2 = (g_hat * x_hat_flat).mean(axis=0, keepdims=True)
        glocal = inv_std * g_hat
        mean_correction = inv_std * m1 if mode == "T" else 0.0
        var_correction = inv_std * m2 * x_hat_flat if mode == "T" else 0.0
        dx_flat = glocal - mean_correction - var_correction
        return [
            dx_flat.reshape(os),
            (grad_flat * x_hat_flat).sum(axis=0),
            grad_flat.sum(axis=0),
        ]


class LayerNorm(Op):
    """layer norm across feature dim"""

    def forward(self, x, gamma, beta, eps=1e-5):
        raw_x = x
        raw_gamma = gamma
        raw_beta = beta
        mean = raw_x.mean(axis=-1, keepdims=True)
        var = raw_x.var(axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + eps)
        x_hat = (raw_x - mean) * inv_std
        y = raw_gamma * x_hat + raw_beta
        self.save_for_backward(x_hat, inv_std, raw_gamma, raw_x.shape)
        return y

    def backward(self, grad):
        raw_grad = grad
        x_hat, inv_std, raw_gamma, os = self._intermediate
        grad_flat = raw_grad.reshape(-1, os[-1])
        x_hat_flat = x_hat.reshape(-1, os[-1])
        inv_std_flat = inv_std.reshape(-1, 1)
        dgamma = (grad_flat * x_hat_flat).sum(axis=0)
        dbeta = grad_flat.sum(axis=0)
        g_hat = raw_gamma * grad_flat
        m1 = g_hat.mean(axis=-1, keepdims=True)
        m2 = (g_hat * x_hat_flat).mean(axis=-1, keepdims=True)
        dx_flat = inv_std_flat * (g_hat - m1 - x_hat_flat * m2)
        return [
            dx_flat.reshape(os),
            dgamma,
            dbeta,
        ]


class Attention(Op):
    """attention block using shared nhwc-style math"""

    def forward(self, x, wq, wk, wv):
        raw_x = x
        raw_wq = wq
        raw_wk = wk
        raw_wv = wv
        q = raw_x @ raw_wq
        k = raw_x @ raw_wk
        v = raw_x @ raw_wv

        k_t = np.transpose(k, (0, 2, 1))
        w_scores = (q @ k_t) / np.sqrt(q.shape[-1])
        w_exp_scores = np.exp(w_scores - np.max(w_scores, axis=-1, keepdims=True))
        w_attn = w_exp_scores / np.sum(w_exp_scores, axis=-1, keepdims=True)
        y = w_attn @ v
        self.save_for_backward(raw_x, raw_wq, raw_wk, raw_wv, q, k, v, w_attn)
        return y

    def backward(self, grad):
        raw_grad = grad
        raw_x, raw_wq, raw_wk, raw_wv, q, k, v, w_attn = self._intermediate
        _, _, d = raw_x.shape

        w_attn_t = np.transpose(w_attn, (0, 2, 1))
        dv = w_attn_t @ raw_grad

        v_t = np.transpose(v, (0, 2, 1))
        dw_attn = raw_grad @ v_t

        dw_scores = (
            (dw_attn - np.sum(dw_attn * w_attn, axis=-1, keepdims=True))
            * w_attn
            / np.sqrt(d)
        )

        dq = dw_scores @ k
        dk = np.transpose(dw_scores, (0, 2, 1)) @ q

        dx = dq @ raw_wq.T + dk @ raw_wk.T + dv @ raw_wv.T

        x2 = raw_x.reshape(-1, d)
        dwq = x2.T @ dq.reshape(-1, d)
        dwk = x2.T @ dk.reshape(-1, d)
        dwv = x2.T @ dv.reshape(-1, d)

        return [
            dx,
            dwq,
            dwk,
            dwv,
        ]
