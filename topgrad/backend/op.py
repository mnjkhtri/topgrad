class Op:
    def __init__(self, *tensors, backend):
        self.parents = tensors
        self.backend = backend
        self._intermediate = []  # keep intermediate numpy arrays

    def save_for_backward(self, *x):
        self._intermediate.extend(x)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, backend=None):
        from topgrad.backend import get_backend  # late import to avoid circular deps
        from topgrad.tensor import Tensor  # late import to avoid circular deps

        if backend is None:
            backend = get_backend()
        parents = tuple(a for a in args if isinstance(a, Tensor))
        op = cls(*parents, backend=backend)
        fwd_args = [a.data if isinstance(a, Tensor) else a for a in args]
        out = op.forward(*fwd_args)
        wrap = getattr(backend, "_wrap", None)
        if callable(wrap):
            out = wrap(out)
        ret = Tensor(out)
        ret._op = op
        return ret
