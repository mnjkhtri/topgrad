class Op:
    def __init__(self, *tensors):
        self.parents = tensors
        self._intermediate = []  # keep intermediate numpy arrays

    def save_for_backward(self, *x):
        self._intermediate.extend(x)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args):
        from topgrad.tensor import Tensor  # late import to avoid circular deps

        parents = tuple(a for a in args if isinstance(a, Tensor))
        op = cls(*parents)
        fwd_args = [a.data if isinstance(a, Tensor) else a for a in args]
        out = op.forward(*fwd_args)
        ret = Tensor(out)
        ret._op = op
        return ret
