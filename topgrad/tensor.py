import numpy as np


class Tensor:
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            print("Error constructing tensor with {data}")
            assert(False)
        self.data = data
        self.grad = None

        self._op = None 
        # It's a pointer to the Op object which is an operation that created this tensor.
        # The Op instance contains information needed to compute gradients for that step during the backward pass
        # Specifically it will contain:
        # - __intermediate:
        # - parents: A tuple containing reference to the input Tensor objects. This forms a compute graph
        # If a tensor was created manually (not as the result of an op), its _op is None

    @classmethod
    def zeros(cls, *shape): return cls(np.zeros(shape, dtype=np.float32))

    @classmethod
    def ones(cls, *shape): return cls(np.ones(shape, dtype=np.float32))

    @classmethod
    def randn(cls, *shape): return cls(np.random.randn(*shape).astype(np.float32))

    @classmethod
    def He(cls, *shape, dist = "normal"):
        assert len(shape) == 2, "He init is only for 2D tensors"
        fan_in, _ = shape
        if dist == "normal":
            std_dev = np.sqrt(2.0 / fan_in)
            return cls((np.random.randn(*shape) * std_dev).astype(np.float32))
        elif dist == "uniform":
            upper = np.sqrt(6.0 / fan_in)
            return cls(np.random.uniform(-upper, upper, size=shape).astype(np.float32))
        pass

    @classmethod
    def Xavier(cls, *shape, dist = "normal"):
        assert len(shape) == 2, "Xavier init is only for 2D tensors"
        fan_in, fan_out = shape
        if dist == "normal":
            std_dev = np.sqrt(2.0 / (fan_in + fan_out))
            return cls((np.random.randn(*shape) * std_dev).astype(np.float32))
        elif dist == "uniform":
            upper = np.sqrt(6.0 / (fan_in + fan_out))
            return cls(np.random.uniform(-upper, upper, size=shape).astype(np.float32))
        pass

    def __repr__(self):
        return f"Tensor {self.data} with grad {self.grad}"
    
    def backward(self, allow_fill=True):
        
        # Base condition: If the tensor has no op, it means it's an original input. So we cant propagate any further back from it. The recursion stops here.
        if self._op is None:
            return

        # Init gradient: The backward pass starts from the final tensor. The gradient of a variable with respect to itself is 1.
        # This section initializes the final tensor's gradient to an array of ones.
        if (self.grad is None) and allow_fill:
            assert self.data.size == 1
            self.grad = np.ones_like(self.data)

        assert(self.grad is not None) #if not leaf tensors, must have gradient

        grads = self._op.backward(self.grad)
        # It calls the backward method of the op that created this tensor. It passes the current tensor's incoming gradient (self.grad) to that function.
        # The op's job is to use this incoming gradient to compute the local gradients with respect to its own input.
        # The grads variable will contain the result of back calculation. It must be a TUPLE OF GRADIENTS
        # The op just calculates the gradients. Distribution happens here.

        for t, g in zip(self._op.parents, grads):
            assert g.shape == t.data.shape,f"grad shape must match tensor shape {g.shape}, {t.data.shape}"
            if t.grad is None:
               t.grad = g
            else:
               t.grad += g
            t.backward(False)

    # foundational ops:
    def sum(self): return Sum.apply(self)
    def add(self, x): return Add.apply(self, x)
    def mul(self, x): return Mul.apply(self, x)
    def dot(self, w): return Dot.apply(self, w)

    # nn ops:
    def relu(self): return ReLU.apply(self)
    def linear(self, w, b): return Linear.apply(self, w, b)
    def logsoftmax(self): return LogSoftmax.apply(self)



class Op:
    def __init__(self, *tensors):
        self.parents = tensors
        self._intermediate = [] # save intermediate values as ndarrays

    def save_for_backward(self, *x):
        self._intermediate.extend(x)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
        # Computes the output value for the Op and also responsible for saving any intermediate values needed for backward pass

    def backward(self, *args, **kwargs):
        raise NotImplementedError
        # Calculates the gradients for that specific operation by applying chain rule. Utilizes upstream gradient and intermediates
        # The blame for the parents for erring childs. Haha a chain rule joke

    @classmethod
    def apply(cls, *args):
        # Connects the Tneosr objects with the internal world of numpy calculations. Instantiates the class and does necessary soup.
        op = cls(*args)
        result = op.forward(*[t.data for t in args])
        ret = Tensor(result)
        ret._op = op
        return ret

# Test ops:

class Sum(Op):
    def forward(self, x):
        self.save_for_backward(x.shape)
        return np.array(x.sum())
    def backward(self, grad):
        x_shape, = self._intermediate
        return [grad * np.ones(x_shape)]

class Add(Op):
    def forward(self, x, y):
        return x + y
    def backward(self, grad):
        return [grad, grad]

class Mul(Op):
    def forward(self, x, y):
      self.save_for_backward(x, y)
      return x * y
    def backward(self, grad):
      x, y = self._intermediate
      return [grad * y, grad * x]
    
# NN ops:

class LogSoftmax(Op):
    # assume axis = -1
    def forward(self, x):
        axis = -1
        max_x = x.max(axis=axis, keepdims=True)
        log_sum_exp = max_x + np.log(np.exp(x - max_x).sum(axis=axis, keepdims=True))
        y = x - log_sum_exp
        self.save_for_backward(y)
        return y
    def backward(self, grad):
        axis = -1
        y, = self._intermediate
        softmax_y = np.exp(y)
        sum_grad = grad.sum(axis=axis, keepdims=True)
        grad_x = grad - softmax_y * sum_grad
        return [grad_x]

class ReLU(Op):
    def forward(self, x):
        self.save_for_backward(x)
        return np.maximum(x, 0)
    def backward(self, grad):
        x, = self._intermediate
        grad_copy = grad.copy(); grad_copy[x < 0] = 0; return [grad_copy]

class Linear(Op):
    def forward(self, x, w, b):
        self.save_for_backward(x, w)
        # Perform the linear transformation: y = xw + b
        return np.dot(x, w) + b
    def backward(self, grad):
        x, w = self._intermediate
        in_features = w.shape[0]
        out_features = w.shape[1]
        grad_x = grad.dot(w.T)
        x_reshaped = x.reshape(-1, in_features)
        grad_reshaped = grad.reshape(-1, out_features)
        grad_w = x_reshaped.T.dot(grad_reshaped)
        axes_to_sum = tuple(range(grad.ndim - 1))
        grad_b = np.sum(grad, axis=axes_to_sum) if axes_to_sum else grad
        return [grad_x, grad_w, grad_b]