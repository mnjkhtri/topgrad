import numpy as np

class Engine:
    training = True   # default

    @classmethod
    def train(cls):
        cls.training = True

    @classmethod
    def eval(cls):
        cls.training = False
        
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

    @property
    def shape(self): return self.data.shape

    @classmethod
    def zeros(cls, *shape): return cls(np.zeros(shape, dtype=np.float32))

    @classmethod
    def ones(cls, *shape): return cls(np.ones(shape, dtype=np.float32))

    @classmethod
    def randn(cls, *shape): return cls(np.random.randn(*shape).astype(np.float32))

    @classmethod
    def He(cls, *shape, dist = "normal"):
        fan_in, _ = (int(np.prod(shape[:-2])) * shape[-2], int(np.prod(shape[:-2])) * shape[-1]) if len(shape) > 1 else (shape[0], shape[0])
        if dist == "normal":
            std_dev = np.sqrt(2.0 / fan_in)
            return cls((np.random.randn(*shape) * std_dev).astype(np.float32))
        elif dist == "uniform":
            upper = np.sqrt(6.0 / fan_in)
            return cls(np.random.uniform(-upper, upper, size=shape).astype(np.float32))
        pass

    @classmethod
    def Xavier(cls, *shape, dist = "normal"):
        fan_in, fan_out = (int(np.prod(shape[:-2])) * shape[-2], int(np.prod(shape[:-2])) * shape[-1]) if len(shape) > 1 else (shape[0], shape[0])
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

    # nn ops:
    def reshape(self, ns): return Reshape.apply(self, ns)
    def logsoftmax(self): return LogSoftmax.apply(self)
    def relu(self): return ReLU.apply(self)
    def linear(self, w, b): return Linear.apply(self, w, b)
    def conv(self, w, b, stride, padding): return Conv.apply(self, w, b, stride, padding)



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
        # Only tensors are graph parents
        parents = tuple(a for a in args if isinstance(a, Tensor))
        op = cls(*parents)
        # Forward sees raw numpy arrays for tensors, raw values otherwise
        fwd_args = [a.data if isinstance(a, Tensor) else a for a in args]
        out = op.forward(*fwd_args)
        ret = Tensor(out)
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
class Reshape(Op):
    def forward(self, x, ns):
        self.save_for_backward(x.shape, ns)
        return x.reshape(ns)
    def backward(self, grad):
        os, _ = self._intermediate
        return [grad.reshape(os)]

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
        """
        x: (..., in_features)
        w: (in_features, out_features)
        b: (out_features,)
        """
        self.save_for_backward(x, w)
        return np.dot(x, w) + b
    def backward(self, grad):
        """
        grad: (..., out_features)
        """
        x, w = self._intermediate
        in_features, out_features = w.shape[0], w.shape[1]
        grad_x = grad.dot(w.T)
        grad_w = x.reshape(-1, in_features).T.dot(grad.reshape(-1, out_features))
        grad_b = grad.reshape(-1, grad.shape[-1]).sum(axis=0)
        return [grad_x, grad_w, grad_b]

class Conv(Op):
    """
    note: To downsample by stride S while keeping output ≈ input / S, use padding = (kernel_size - stride) / 2 per side. Thanks.
    """
    @staticmethod
    def _im2col(x, KH, KW, stride, pad):
        """
        x: (N, H, W, C)
        returns:
          x_col: (N*H_out*W_out, KH*KW*C)
          cache: shapes needed for backward
        """
        N, H, W, C = x.shape
        sh, sw = stride
        ph, pw = pad

        H_out = (H + 2*ph - KH) // sh + 1
        W_out = (W + 2*pw - KW) // sw + 1

        # zero-pad on H and W
        x_p = np.pad(x, ((0,0), (ph,ph), (pw,pw), (0,0)), mode='constant')

        x_col = np.empty((N, H_out, W_out, KH, KW, C), dtype=x.dtype)
        for i in range(KH):
            i_max = i + sh*H_out
            for j in range(KW):
                j_max = j + sw*W_out
                x_col[:, :, :, i, j, :] = x_p[:, i:i_max:sh, j:j_max:sw, :]

        x_col = x_col.reshape(N*H_out*W_out, KH*KW*C)
        cache = (x.shape, (KH, KW, C), (H_out, W_out), stride, pad)
        return x_col, cache
    
    @staticmethod
    def _col2im(col, cache):
        """
        Reverse of im2col to reconstruct gradient wrt input.
        col: (N*H_out*W_out, KH*KW*C)
        """
        (N, H, W, C), (KH, KW, _), (H_out, W_out), stride, pad = cache
        sh, sw = stride
        ph, pw = pad

        col = col.reshape(N, H_out, W_out, KH, KW, C)

        dx_p = np.zeros((N, H + 2*ph, W + 2*pw, C), dtype=col.dtype)
        for i in range(KH):
            i_max = i + sh*H_out
            for j in range(KW):
                j_max = j + sw*W_out
                dx_p[:, i:i_max:sh, j:j_max:sw, :] += col[:, :, :, i, j, :]

        # crop padding
        if ph == 0 and pw == 0:
            return dx_p
        return dx_p[:, ph:ph+H, pw:pw+W, :]

    def forward(self, x, w, b, stride, padding):
        """
        x: (N, H, W, Cin)
        w: (KH, KW, Cin, Cout)
        b: (Cout,)
        """
        KH, KW, Cin, Cout = w.shape
        x_col, im2col_cache = self._im2col(x, KH, KW, stride, padding)
        W_col = w.reshape(KH*KW*Cin, Cout)           # (K, Cout)
        out_cols = x_col @ W_col                     # (N*H_out*W_out, Cout)
        out_cols += b                                # broadcast add bias
        # restore to NHWC
        N, H, W, _ = x.shape
        ph, pw = padding
        sh, sw = stride
        H_out = (H + 2*ph - KH)//sh + 1
        W_out = (W + 2*pw - KW)//sw + 1
        out = out_cols.reshape(N, H_out, W_out, Cout)

        # save for backward
        self.save_for_backward(x_col, w, im2col_cache)
        return out
    
    def backward(self, grad):
        """
        grad: (N, H_out, W_out, Cout)
        """
        (x_col, w, im2col_cache) = self._intermediate
        KH, KW, Cin, Cout = w.shape

        # flatten grad over spatial positions
        N, H_out, W_out, Cout_check = grad.shape
        assert Cout_check == Cout
        grad_col = grad.reshape(N*H_out*W_out, Cout)    # (NHW, Cout)

        # dW = x_col^T @ grad_col
        dw_col = x_col.T @ grad_col                      # (K, Cout)
        dw = dw_col.reshape(KH, KW, Cin, Cout)

        # dB
        db = grad_col.sum(axis=0)                        # (Cout,)

        # dX via col2im: (NHW, K) = grad_col @ W_col^T
        W_col = w.reshape(KH*KW*Cin, Cout)               # (K, Cout)
        dcols = grad_col @ W_col.T                       # (NHW, K)
        dx = self._col2im(dcols, im2col_cache)           # (N, H, W, Cin)

        return [dx, dw, db]
    