import numpy as np
import torch
from topgrad.tensor import Tensor
import pytest


@pytest.mark.parametrize("shape", [
    (100,),          # 1D Vector
    (32, 32),        # 2D Square Matrix
    (16, 64),        # 2D Rectangular Matrix
    (8, 16, 8),      # 3D Tensor
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
def test_op_linear_nd(x_shape, in_features, out_features):
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