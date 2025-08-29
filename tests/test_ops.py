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
def test_op_sum(shape):

    input_np = np.random.randn(*shape).astype(np.float32)

    input_t = Tensor(input_np.copy())
    input_torch = torch.tensor(input_np.copy(), requires_grad=True)

    result_t = input_t.sum()
    result_torch = input_torch.sum()

    np.testing.assert_allclose(
        result_t.data,
        result_torch.detach().numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Forward pass result for ReLU does not match PyTorch"
    )

    result_t.backward()
    result_torch.backward()

    np.testing.assert_allclose(
        input_t.grad,
        input_torch.grad.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Backward pass gradient for ReLU does not match PyTorch"
    )

@pytest.mark.parametrize("shape", [
    (100,),          # 1D Vector
    (32, 32),        # 2D Square Matrix
    (16, 64),        # 2D Rectangular Matrix
    (8, 16, 8),      # 3D Tensor
])
def test_op_neg(shape):

    # Create identical inputs
    input_np = np.random.randn(*shape).astype(np.float32)

    input_t = Tensor(input_np.copy())
    input_torch = torch.tensor(input_np.copy(), requires_grad=True)

    # Forward pass
    result_t = input_t.neg()
    result_torch = -input_torch

    np.testing.assert_allclose(
        result_t.data,
        result_torch.detach().numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Forward pass result for Neg does not match PyTorch"
    )

    # Backward pass
    # Use a scalar loss to make .backward() valid
    loss_t = result_t.sum()
    loss_torch = result_torch.sum()

    loss_t.backward()
    loss_torch.backward()

    np.testing.assert_allclose(
        input_t.grad,
        input_torch.grad.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Backward pass gradient for Neg does not match PyTorch"
    )

@pytest.mark.parametrize("shape", [
    (100,),          # 1D Vector
    (32, 32),        # 2D Square Matrix
    (16, 64),        # 2D Rectangular Matrix
    (8, 16, 8),      # 3D Tensor
])
def test_op_add(shape):

    input_a_np = np.random.randn(*shape).astype(np.float32)
    input_b_np = np.random.randn(*shape).astype(np.float32)

    input_a_t = Tensor(input_a_np.copy())
    input_b_t = Tensor(input_b_np.copy())
    input_a_torch = torch.tensor(input_a_np.copy(), requires_grad=True)
    input_b_torch = torch.tensor(input_b_np.copy(), requires_grad=True)

    result_t = input_a_t.add(input_b_t)
    result_torch = input_a_torch + input_b_torch

    np.testing.assert_allclose(
        result_t.data,
        result_torch.detach().numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Forward pass result for Add does not match PyTorch"
    )

    result_t.sum().backward()
    result_torch.sum().backward()

    np.testing.assert_allclose(
        input_a_t.grad,
        input_a_torch.grad.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Backward pass gradient for Add does not match PyTorch"
    )

    np.testing.assert_allclose(
        input_b_t.grad,
        input_b_torch.grad.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Backward pass gradient for Add does not match PyTorch"
    )

@pytest.mark.parametrize("shape", [
    (100,),          # 1D Vector
    (32, 32),        # 2D Square Matrix
    (16, 64),        # 2D Rectangular Matrix
    (8, 16, 8),      # 3D Tensor
])
def test_op_mul(shape):

    input_a_np = np.random.randn(*shape).astype(np.float32)
    input_b_np = np.random.randn(*shape).astype(np.float32)

    input_a_t = Tensor(input_a_np.copy())
    input_b_t = Tensor(input_b_np.copy())
    input_a_torch = torch.tensor(input_a_np.copy(), requires_grad=True)
    input_b_torch = torch.tensor(input_b_np.copy(), requires_grad=True)

    result_t = input_a_t.mul(input_b_t)
    result_torch = input_a_torch * input_b_torch

    np.testing.assert_allclose(
        result_t.data,
        result_torch.detach().numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Forward pass result for Mul does not match PyTorch"
    )

    result_t.sum().backward()
    result_torch.sum().backward()

    np.testing.assert_allclose(
        input_a_t.grad,
        input_a_torch.grad.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Backward pass gradient for Mul does not match PyTorch"
    )

    np.testing.assert_allclose(
        input_b_t.grad,
        input_b_torch.grad.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Backward pass gradient for Mul does not match PyTorch"
    )

@pytest.mark.parametrize("shape", [
    (100,),          # 1D Vector
    (32, 32),        # 2D Square Matrix
    (16, 64),        # 2D Rectangular Matrix
    (8, 16, 8),      # 3D Tensor
])
def test_op_mean(shape):

    input_np = np.random.randn(*shape).astype(np.float32)

    input_t = Tensor(input_np.copy())
    input_torch = torch.tensor(input_np.copy(), requires_grad=True)

    # Forward pass
    result_t = input_t.mean()
    result_torch = input_torch.mean()

    np.testing.assert_allclose(
        result_t.data,
        result_torch.detach().numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Forward pass result for Mean does not match PyTorch"
    )

    # Backward pass
    result_t.backward()
    result_torch.backward()

    np.testing.assert_allclose(
        input_t.grad,
        input_torch.grad.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="Backward pass gradient for Mean does not match PyTorch"
    )