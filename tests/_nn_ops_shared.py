import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")


class BackendHelper:
    def __init__(self, name):
        self.name = name

    def randn(self, shape):
        if self.name == "triton":
            return torch.randn(shape, device="cuda", dtype=torch.float32).contiguous()
        return np.random.randn(*shape).astype(np.float32)

    def make_pair(self, shape):
        if self.name == "triton":
            data = self.randn(shape)
            return data.clone(), data.clone().detach().requires_grad_(True)
        data = self.randn(shape)
        return data.copy(), torch.tensor(data.copy(), requires_grad=True)

    def make_state(self, num_features):
        if self.name == "triton":
            running_mean = torch.zeros((num_features,), device="cuda", dtype=torch.float32)
            running_var = torch.ones((num_features,), device="cuda", dtype=torch.float32)
        else:
            running_mean = np.zeros((num_features,), dtype=np.float32)
            running_var = np.ones((num_features,), dtype=np.float32)

        class _State:
            def __init__(self, rm, rv):
                self.running_mean = rm
                self.running_var = rv

        return _State(running_mean, running_var)

    def tolerances(self):
        if self.name == "triton":
            return 1e-3, 1e-3
        return 1e-4, 1e-4


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        if type(x) is not torch.Tensor:
            disable = getattr(torch._C, "DisableTorchDispatch", None)
            if disable is None:
                disable = torch._C._DisableTorchDispatch
            with disable():
                x = x.as_subclass(torch.Tensor)
        return x.detach().cpu().numpy()
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def run_relu_cases(helper, make_tensor):
    for shape in [(100,), (32, 32), (8, 16, 8)]:
        data, x_th = helper.make_pair(shape)
        x_t = make_tensor(data)
        y_t = x_t.relu()
        y_th = torch.nn.functional.relu(x_th)
        rtol, atol = helper.tolerances()
        np.testing.assert_allclose(to_numpy(y_t.data), to_numpy(y_th), rtol=rtol, atol=atol)
        y_t.sum().backward()
        y_th.sum().backward()
        np.testing.assert_allclose(to_numpy(x_t.grad), to_numpy(x_th.grad), rtol=rtol, atol=atol)


def run_logsoftmax_cases(helper, make_tensor):
    for shape in [(32, 32), (8, 16, 8), (4, 5, 5, 8)]:
        data, x_th = helper.make_pair(shape)
        x_t = make_tensor(data)
        y_t = x_t.logsoftmax()
        y_th = torch.nn.functional.log_softmax(x_th, dim=-1)
        rtol, atol = helper.tolerances()
        np.testing.assert_allclose(to_numpy(y_t.data), to_numpy(y_th), rtol=rtol, atol=atol)
        y_t.sum().backward()
        y_th.sum().backward()
        np.testing.assert_allclose(to_numpy(x_t.grad), to_numpy(x_th.grad), rtol=rtol, atol=atol)


def run_linear_cases(helper, make_tensor):
    cases = [
        ((16, 32), (32, 64)),
        ((128, 784), (784, 10)),
    ]
    for x_shape, (in_features, out_features) in cases:
        x_data = helper.randn(x_shape)
        w_data = helper.randn((in_features, out_features))
        b_data = helper.randn((out_features,))

        x_t = make_tensor(x_data)
        w_t = make_tensor(w_data)
        b_t = make_tensor(b_data)

        if helper.name == "triton":
            x_th = x_data.clone().detach().requires_grad_(True)
            w_th = w_data.clone().detach().requires_grad_(True)
            b_th = b_data.clone().detach().requires_grad_(True)
        else:
            x_th = torch.tensor(x_data.copy(), requires_grad=True)
            w_th = torch.tensor(w_data.copy(), requires_grad=True)
            b_th = torch.tensor(b_data.copy(), requires_grad=True)

        y_t = x_t.linear(w_t, b_t)
        y_th = torch.matmul(x_th, w_th) + b_th

        rtol, atol = helper.tolerances()
        np.testing.assert_allclose(to_numpy(y_t.data), to_numpy(y_th), rtol=rtol, atol=atol)

        y_t.sum().backward()
        y_th.sum().backward()

        np.testing.assert_allclose(to_numpy(x_t.grad), to_numpy(x_th.grad), rtol=rtol, atol=atol)
        np.testing.assert_allclose(to_numpy(w_t.grad), to_numpy(w_th.grad), rtol=rtol, atol=atol)
        np.testing.assert_allclose(to_numpy(b_t.grad), to_numpy(b_th.grad), rtol=rtol, atol=atol)


def run_conv_cases(helper, make_tensor):
    cases = [
        ((2, 8, 8, 3, 4), (1, 1), (1, 1), (0, 0)),
        ((1, 5, 5, 2, 3), (3, 3), (1, 1), (1, 1)),
        ((3, 16, 16, 5, 6), (3, 3), (2, 2), (0, 0)),
    ]
    for x_shape, k_shape, stride, padding in cases:
        n, h, w, cin, cout = x_shape
        kw, kh = k_shape
        sh, sw = stride
        ph, pw = padding

        x_data = helper.randn((n, h, w, cin))
        w_data = helper.randn((kh, kw, cin, cout))
        b_data = helper.randn((cout,))

        x_t = make_tensor(x_data)
        w_t = make_tensor(w_data)
        b_t = make_tensor(b_data)

        if helper.name == "triton":
            x_th = x_data.permute(0, 3, 1, 2).contiguous().requires_grad_(True)
            w_th = w_data.permute(3, 2, 0, 1).contiguous().requires_grad_(True)
            b_th = b_data.clone().detach().requires_grad_(True)
        else:
            x_th = torch.tensor(x_data.transpose(0, 3, 1, 2).copy(), requires_grad=True)
            w_th = torch.tensor(w_data.transpose(3, 2, 0, 1).copy(), requires_grad=True)
            b_th = torch.tensor(b_data.copy(), requires_grad=True)

        y_t = x_t.conv(w_t, b_t, stride=(sh, sw), padding=(ph, pw))
        y_th = torch.nn.functional.conv2d(
            x_th, w_th, b_th, stride=(sh, sw), padding=(ph, pw)
        )

        y_ref = y_th.detach().permute(0, 2, 3, 1).contiguous()
        rtol, atol = helper.tolerances()
        np.testing.assert_allclose(to_numpy(y_t.data), to_numpy(y_ref), rtol=rtol, atol=atol)

        y_t.sum().backward()
        y_th.sum().backward()

        x_grad_ref = x_th.grad.detach().permute(0, 2, 3, 1).contiguous()
        np.testing.assert_allclose(to_numpy(x_t.grad), to_numpy(x_grad_ref), rtol=rtol, atol=atol)

        w_grad_ref = w_th.grad.detach().permute(2, 3, 1, 0).contiguous()
        np.testing.assert_allclose(to_numpy(w_t.grad), to_numpy(w_grad_ref), rtol=rtol, atol=atol)

        b_grad_ref = b_th.grad.detach()
        np.testing.assert_allclose(to_numpy(b_t.grad), to_numpy(b_grad_ref), rtol=rtol, atol=atol)


def run_batchnorm_cases(helper, make_tensor):
    eps = 1e-5
    momentum = 0.1
    cases = [((128, 784), "T"), ((8, 10, 16), "E")]
    for x_shape, mode in cases:
        num_features = x_shape[-1]

        x_data = helper.randn(x_shape)
        gamma_data = helper.randn((num_features,))
        beta_data = helper.randn((num_features,))

        x_t = make_tensor(x_data)
        gamma_t = make_tensor(gamma_data)
        beta_t = make_tensor(beta_data)

        if helper.name == "triton":
            x_th = x_data.clone().detach().requires_grad_(True)
            gamma_th = gamma_data.clone().detach().requires_grad_(True)
            beta_th = beta_data.clone().detach().requires_grad_(True)
            running_mean_th = torch.zeros((num_features,), device="cuda", dtype=torch.float32)
            running_var_th = torch.ones((num_features,), device="cuda", dtype=torch.float32)
        else:
            x_th = torch.tensor(x_data.copy(), requires_grad=True)
            gamma_th = torch.tensor(gamma_data.copy(), requires_grad=True)
            beta_th = torch.tensor(beta_data.copy(), requires_grad=True)
            running_mean_th = torch.zeros((num_features,), dtype=torch.float32)
            running_var_th = torch.ones((num_features,), dtype=torch.float32)

        state = helper.make_state(num_features)

        y_t = x_t.batchnorm(gamma_t, beta_t, state, eps=eps, momentum=momentum, mode=mode)

        y_th = torch.movedim(
            torch.nn.functional.batch_norm(
                torch.movedim(x_th, -1, 1),
                running_mean=running_mean_th,
                running_var=running_var_th,
                weight=gamma_th,
                bias=beta_th,
                training=True if mode == "T" else False,
                momentum=momentum,
                eps=eps,
            ),
            1,
            -1,
        ).contiguous()

        rtol, atol = helper.tolerances()
        np.testing.assert_allclose(to_numpy(y_t.data), to_numpy(y_th), rtol=rtol, atol=atol)

        y_t.sum().backward()
        y_th.sum().backward()

        np.testing.assert_allclose(to_numpy(x_t.grad), to_numpy(x_th.grad), rtol=rtol, atol=atol)
        np.testing.assert_allclose(to_numpy(gamma_t.grad), to_numpy(gamma_th.grad), rtol=rtol, atol=atol)
        np.testing.assert_allclose(to_numpy(beta_t.grad), to_numpy(beta_th.grad), rtol=rtol, atol=atol)


def run_layernorm_cases(helper, make_tensor):
    eps = 1e-5
    cases = [(128, 784), (8, 10, 16)]
    for x_shape in cases:
        num_features = x_shape[-1]

        x_data = helper.randn(x_shape)
        gamma_data = helper.randn((num_features,))
        beta_data = helper.randn((num_features,))

        x_t = make_tensor(x_data)
        gamma_t = make_tensor(gamma_data)
        beta_t = make_tensor(beta_data)

        if helper.name == "triton":
            x_th = x_data.clone().detach().requires_grad_(True)
            gamma_th = gamma_data.clone().detach().requires_grad_(True)
            beta_th = beta_data.clone().detach().requires_grad_(True)
        else:
            x_th = torch.tensor(x_data.copy(), requires_grad=True)
            gamma_th = torch.tensor(gamma_data.copy(), requires_grad=True)
            beta_th = torch.tensor(beta_data.copy(), requires_grad=True)

        y_t = x_t.layernorm(gamma_t, beta_t, eps=eps)
        y_th = torch.nn.functional.layer_norm(
            x_th, normalized_shape=(num_features,), weight=gamma_th, bias=beta_th, eps=eps
        )

        rtol, atol = helper.tolerances()
        np.testing.assert_allclose(to_numpy(y_t.data), to_numpy(y_th), rtol=rtol, atol=atol)

        y_t.sum().backward()
        y_th.sum().backward()

        np.testing.assert_allclose(to_numpy(x_t.grad), to_numpy(x_th.grad), rtol=rtol, atol=atol)
        np.testing.assert_allclose(to_numpy(gamma_t.grad), to_numpy(gamma_th.grad), rtol=rtol, atol=atol)
        np.testing.assert_allclose(to_numpy(beta_t.grad), to_numpy(beta_th.grad), rtol=rtol, atol=atol)


def run_attention_cases(helper, make_tensor):
    cases = [(2, 4, 8), (1, 2, 16), (3, 5, 4)]
    for batch, tokens, dim in cases:
        x_data = helper.randn((batch, tokens, dim))
        wq_data = helper.randn((dim, dim))
        wk_data = helper.randn((dim, dim))
        wv_data = helper.randn((dim, dim))

        x_t = make_tensor(x_data)
        wq_t = make_tensor(wq_data)
        wk_t = make_tensor(wk_data)
        wv_t = make_tensor(wv_data)

        if helper.name == "triton":
            x_th = x_data.clone().detach().requires_grad_(True)
            wq_th = wq_data.clone().detach().requires_grad_(True)
            wk_th = wk_data.clone().detach().requires_grad_(True)
            wv_th = wv_data.clone().detach().requires_grad_(True)
        else:
            x_th = torch.tensor(x_data.copy(), requires_grad=True)
            wq_th = torch.tensor(wq_data.copy(), requires_grad=True)
            wk_th = torch.tensor(wk_data.copy(), requires_grad=True)
            wv_th = torch.tensor(wv_data.copy(), requires_grad=True)

        q = x_th @ wq_th
        k = x_th @ wk_th
        v = x_th @ wv_th
        scores = (q @ k.transpose(1, 2)) / math.sqrt(dim)
        attn = torch.softmax(scores, dim=-1)
        y_th = attn @ v

        y_t = x_t.attention(wq_t, wk_t, wv_t)

        rtol, atol = helper.tolerances()
        np.testing.assert_allclose(to_numpy(y_t.data), to_numpy(y_th), rtol=rtol, atol=atol)

        y_t.sum().backward()
        y_th.sum().backward()

        np.testing.assert_allclose(to_numpy(x_t.grad), to_numpy(x_th.grad), rtol=rtol, atol=atol)
        np.testing.assert_allclose(to_numpy(wq_t.grad), to_numpy(wq_th.grad), rtol=rtol, atol=atol)
        np.testing.assert_allclose(to_numpy(wk_t.grad), to_numpy(wk_th.grad), rtol=rtol, atol=atol)
        np.testing.assert_allclose(to_numpy(wv_t.grad), to_numpy(wv_th.grad), rtol=rtol, atol=atol)
