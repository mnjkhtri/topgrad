import numpy as np
import torch
from topgrad.tensor import Tensor
import pytest

@pytest.mark.parametrize("old_shape, new_shape", [
    ((100,),          (20, 5)),         # 1D -> 2D
    ((32, 32),        (1024,)),         # 2D -> 1D
    ((16, 64),        (16, 8, 8)),      # 2D -> 3D
    ((8, 16, 8),      (8, 128)),        # 3D -> 2D
    ((32, 8, 16, 8),  (32, 1024)),      # 4D -> 2D
])
def test_op_reshape(old_shape, new_shape):

    input_np = np.random.randn(*old_shape).astype(np.float32)

    input_t = Tensor(input_np.copy())
    input_torch = torch.tensor(input_np.copy(), requires_grad=True)

    result_t = input_t.reshape(new_shape)
    result_torch = input_torch.reshape(new_shape)

    np.testing.assert_allclose(
        result_t.data,
        result_torch.detach().numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Forward pass result for Reshape does not match PyTorch"
    )

    result_t.sum().backward()
    result_torch.sum().backward()

    np.testing.assert_allclose(
        input_t.grad,
        input_torch.grad.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Backward pass gradient for Reshape does not match PyTorch"
    )

@pytest.mark.parametrize("shape", [
    (100,),          # 1D Vector
    (32, 32),        # 2D Square Matrix
    (16, 64),        # 2D Rectangular Matrix
    (8, 16, 8),      # 3D Tensor
    (32, 8, 16, 8)   # 4D Tensor
])
def test_op_logsoftmax(shape):

    input_np = np.random.randn(*shape).astype(np.float32)

    input_t = Tensor(input_np.copy())
    input_torch = torch.tensor(input_np.copy(), requires_grad=True)

    result_t = input_t.logsoftmax()
    result_torch = torch.nn.functional.log_softmax(input_torch, dim=-1)

    np.testing.assert_allclose(
        result_t.data,
        result_torch.detach().numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Forward pass result for LogSoftmax does not match PyTorch"
    )

    result_t.sum().backward()
    result_torch.sum().backward()

    np.testing.assert_allclose(
        input_t.grad,
        input_torch.grad.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Backward pass gradient for LogSoftmax does not match PyTorch"
    )

@pytest.mark.parametrize("shape", [
    (100,),          # 1D Vector
    (32, 32),        # 2D Square Matrix
    (16, 64),        # 2D Rectangular Matrix
    (8, 16, 8),      # 3D Tensor
    (32, 8, 16, 8)   # 4D Tensor
])
def test_op_relu(shape):

    input_np = np.random.randn(*shape).astype(np.float32)

    input_t = Tensor(input_np.copy())
    input_torch = torch.tensor(input_np.copy(), requires_grad=True)

    result_t = input_t.relu()
    result_torch = torch.nn.functional.relu(input_torch)

    np.testing.assert_allclose(
        result_t.data,
        result_torch.detach().numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Forward pass result for ReLU does not match PyTorch"
    )

    result_t.sum().backward()
    result_torch.sum().backward()

    np.testing.assert_allclose(
        input_t.grad,
        input_torch.grad.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Backward pass gradient for ReLU does not match PyTorch"
    )

@pytest.mark.parametrize("x_shape, in_features, out_features", [
    # 1D (Batchless)
    ((32,), 32, 64),
    ((1,), 1, 1),
    # 2D (Standard Batched)
    ((16, 32), 32, 64),
    ((128, 784), 784, 10),
    # 3D (e.g., Batch, Sequence Length, Features)
    ((8, 10, 16), 16, 32),
    # 4D (e.g., Batch, Height, Width, Channels)
    ((4, 5, 5, 8), 8, 16),
])
def test_op_linear(x_shape, in_features, out_features):
    # Initialize random data for inputs, weights, and bias
    x_np = np.random.randn(*x_shape).astype(np.float32)
    w_np = np.random.randn(in_features, out_features).astype(np.float32)
    b_np = np.random.randn(out_features).astype(np.float32)

    # Create tensors for our framework and for PyTorch
    x_t = Tensor(x_np.copy())
    w_t = Tensor(w_np.copy())
    b_t = Tensor(b_np.copy())
    x_torch = torch.tensor(x_np.copy(), requires_grad=True)
    w_torch = torch.tensor(w_np.copy(), requires_grad=True)
    b_torch = torch.tensor(b_np.copy(), requires_grad=True)

    # Forward pass
    result_t = x_t.linear(w_t, b_t)
    result_torch = torch.matmul(x_torch, w_torch) + b_torch

    np.testing.assert_allclose(
        result_t.data,
        result_torch.detach().numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"Forward pass for shape {x_shape} does not match PyTorch"
    )

    # Backward pass
    result_t.sum().backward()
    result_torch.sum().backward()

    # Assert gradients for all inputs (x, w, b)
    np.testing.assert_allclose(
        x_t.grad,
        x_torch.grad.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"Backward pass gradient for 'x' (shape {x_shape}) does not match PyTorch"
    )
    np.testing.assert_allclose(
        w_t.grad,
        w_torch.grad.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"Backward pass gradient for 'w' (shape {x_shape}) does not match PyTorch"
    )
    np.testing.assert_allclose(
        b_t.grad,
        b_torch.grad.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"Backward pass gradient for 'b' (shape {x_shape}) does not match PyTorch"
    )

@pytest.mark.parametrize("N, H, W, Cin, Cout, KH, KW, sh, sw, ph, pw", [
    # 1) Simple 1x1 conv
    (2, 8, 8, 3, 4, 1, 1, 1, 1, 0, 0),
    # 2) Edge case
    (1, 5, 5, 2, 3, 3, 3, 1, 1, 1, 1),
    # 3) Same padding, stride 1
    (1, 7, 9, 8, 16, 3, 3, 1, 1, 1, 1),
    # 4) Valid padding
    (3, 16, 16, 5, 6, 3, 3, 2, 2, 0, 0),
    # 5) Rect kernel
    (2, 15, 11, 4, 7, 5, 2, 2, 1, 2, 0),
    # 6) Bigger channels
    (4, 10, 12, 8, 8, 3, 3, 1, 2, 1, 0),
])
def test_op_conv(N, H, W, Cin, Cout, KH, KW, sh, sw, ph, pw):
    x_np = np.random.randn(N, H, W, Cin).astype(np.float32)
    w_np = np.random.randn(KH, KW, Cin, Cout).astype(np.float32)
    b_np = np.random.randn(Cout).astype(np.float32)

    x_t = Tensor(x_np.copy())
    w_t = Tensor(w_np.copy())
    b_t = Tensor(b_np.copy())

    x_th = torch.tensor(x_np.transpose(0,3,1,2).copy(), requires_grad=True)  # N, Cin, H, W
    w_th = torch.tensor(w_np.transpose(3,2,0,1).copy(), requires_grad=True)  # Cout, Cin, KH, KW
    b_th = torch.tensor(b_np.copy(), requires_grad=True)

    y_t = x_t.conv(w_t, b_t, stride=(sh, sw), padding=(ph, pw))
    y_th = torch.nn.functional.conv2d(x_th, w_th, b_th, stride=(sh, sw), padding=(ph, pw))

    # bring torch output back to NHWC
    y_ref = y_th.detach().permute(0, 2, 3, 1).contiguous().numpy()

    # shape check
    H_out = (H + 2*ph - KH) // sh + 1
    W_out = (W + 2*pw - KW) // sw + 1
    assert y_t.data.shape == (N, H_out, W_out, Cout)

    np.testing.assert_allclose(
        y_t.data, y_ref, rtol=1e-4, atol=1e-4,
        err_msg=f"Forward mismatch for (N={N},H={H},W={W},Cin={Cin},Cout={Cout},K=({KH},{KW}),"
                f"stride=({sh},{sw}),pad=({ph},{pw}))"
    )

    y_t.sum().backward()
    y_th.sum().backward()

    x_grad_ref = x_th.grad.detach().permute(0, 2, 3, 1).contiguous().numpy()
    np.testing.assert_allclose(
        x_t.grad, x_grad_ref,
        rtol=1e-4,
        atol=1e-4,
        err_msg="dx mismatch"
    )

    w_grad_ref = w_th.grad.detach().permute(2, 3, 1, 0).contiguous().numpy()
    np.testing.assert_allclose(
        w_t.grad, w_grad_ref,
        rtol=1e-4,
        atol=1e-4,
        err_msg="dw mismatch"
    )

    b_grad_ref = b_th.grad.detach().numpy()
    np.testing.assert_allclose(
        b_t.grad, b_grad_ref,
        rtol=1e-4,
        atol=1e-4,
        err_msg="db mismatch"
    )