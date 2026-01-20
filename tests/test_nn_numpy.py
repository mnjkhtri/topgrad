import pytest

pytest.importorskip("torch")

from topgrad.backend import get_backend, set_backend
from topgrad.tensor import Tensor

from ._nn_ops_shared import (
    BackendHelper,
    run_attention_cases,
    run_batchnorm_cases,
    run_conv_cases,
    run_layernorm_cases,
    run_linear_cases,
    run_logsoftmax_cases,
    run_relu_cases,
)


def _make_tensor(data):
    return Tensor(data.copy())


class _NumpyHelper(BackendHelper):
    def __init__(self):
        super().__init__("numpy")


def setup_module():
    global _PREV_BACKEND
    _PREV_BACKEND = get_backend()
    set_backend("numpy")


def teardown_module():
    set_backend(_PREV_BACKEND)


def test_relu_numpy():
    helper = _NumpyHelper()
    run_relu_cases(helper, _make_tensor)


def test_logsoftmax_numpy():
    helper = _NumpyHelper()
    run_logsoftmax_cases(helper, _make_tensor)


def test_linear_numpy():
    helper = _NumpyHelper()
    run_linear_cases(helper, _make_tensor)


def test_conv_numpy():
    helper = _NumpyHelper()
    run_conv_cases(helper, _make_tensor)


def test_batchnorm_numpy():
    helper = _NumpyHelper()
    run_batchnorm_cases(helper, _make_tensor)


def test_layernorm_numpy():
    helper = _NumpyHelper()
    run_layernorm_cases(helper, _make_tensor)


def test_attention_numpy():
    helper = _NumpyHelper()
    run_attention_cases(helper, _make_tensor)
