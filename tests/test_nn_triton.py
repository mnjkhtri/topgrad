import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available for Triton tests", allow_module_level=True)

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
    return Tensor(data.clone())


class _TritonHelper(BackendHelper):
    def __init__(self):
        super().__init__("triton")


def setup_module():
    global _PREV_BACKEND
    _PREV_BACKEND = get_backend()
    set_backend("triton")


def teardown_module():
    set_backend(_PREV_BACKEND)


def test_relu_triton():
    helper = _TritonHelper()
    run_relu_cases(helper, _make_tensor)


def test_logsoftmax_triton():
    helper = _TritonHelper()
    run_logsoftmax_cases(helper, _make_tensor)


def test_linear_triton():
    helper = _TritonHelper()
    run_linear_cases(helper, _make_tensor)


def test_conv_triton():
    helper = _TritonHelper()
    run_conv_cases(helper, _make_tensor)


def test_batchnorm_triton():
    helper = _TritonHelper()
    run_batchnorm_cases(helper, _make_tensor)


def test_layernorm_triton():
    helper = _TritonHelper()
    run_layernorm_cases(helper, _make_tensor)


def test_attention_triton():
    helper = _TritonHelper()
    run_attention_cases(helper, _make_tensor)
