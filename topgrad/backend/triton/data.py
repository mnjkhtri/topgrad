import torch
from torch.utils._pytree import tree_map


def _get_aten_op(name):
    return getattr(torch.ops.aten, name, None)


# Define a strict allowlist of metadata/view ops used by the Triton backend.
# Keep this minimal to force compute ops into Triton implementations.
ALLOWED_METADATA_OPS = {
    op
    for op in (
        _get_aten_op("reshape"),  # Tensor.reshape(...)
        _get_aten_op("view"),  # Tensor.view(...)
        _get_aten_op("transpose"),  # Tensor.transpose(...)
        _get_aten_op("detach"),  # Tensor.detach()
        _get_aten_op("size"),  # Tensor.size()
        _get_aten_op("stride"),  # Tensor.stride()
        _get_aten_op("numel"),  # Tensor.numel()
        _get_aten_op("dim"),  # Tensor.dim()
        _get_aten_op("is_contiguous"),  # Tensor.is_contiguous()
        _get_aten_op("contiguous"),  # Tensor.contiguous()
        _get_aten_op("empty_like"),  # torch.empty_like(...) in kernels
    )
    if op is not None
}


class TritonTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem):
        if isinstance(elem, TritonTensor):
            return elem
        return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """
        Intercepts all PyTorch operations called on this tensor.
        Raises an error if the operation is not in the ALLOWED_METADATA_OPS list.
        """
        if kwargs is None:
            kwargs = {}

        # Check if the requested operation (or its packet) is allowed
        packet = getattr(func, "overloadpacket", None)
        if func not in ALLOWED_METADATA_OPS and packet not in ALLOWED_METADATA_OPS:
            raise RuntimeError(
                f"Operation {func} is forbidden and must be implemented in Triton."
            )

        # Helpers to strip and re-apply the TritonTensor wrapper
        def unwrap(x):
            return x.as_subclass(torch.Tensor) if isinstance(x, TritonTensor) else x

        def wrap(x):
            return (
                TritonTensor(x)
                if isinstance(x, torch.Tensor) and not isinstance(x, TritonTensor)
                else x
            )

        # Run the allowed metadata op using the underlying PyTorch logic
        # We disable dispatch to prevent infinite recursion
        disable = getattr(
            torch._C, "DisableTorchDispatch", torch._C._DisableTorchDispatch
        )
        with disable():
            args = tree_map(unwrap, args)
            kwargs = tree_map(unwrap, kwargs)
            out = func(*args, **kwargs)

        return tree_map(wrap, out)
