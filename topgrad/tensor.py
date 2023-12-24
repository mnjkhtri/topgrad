from functools import partialmethod
import numpy as np

class Tensor:
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            print("Error constructing tensor with {data}")
            assert(False)
        self.data = data
        self.grad = None

        self._ctx = None #The subclass Function object from which it originated; contains parents and saved_tensors; also forward and backward

    def __repr__(self):
        return f"Tensor {self.data} with grad {self.grad}"
    
    def backward(self, allow_fill=True):
        #Checks if there is a context associated with the current tensor, if not, it means this means tensor is not part of a computation that requires backprop
        if self._ctx is None:
            return

        #If conditions are met, it assumes that the tensor is a scalar and fills its graident with ones 
        if self.grad is None and allow_fill:
            assert self.data.size == 1
            self.grad = np.ones_like(self.data)

        assert(self.grad is not None) #if not leaf tensors, must have gradient

        grads = self._ctx.backward(self._ctx, self.grad)
        if len(self._ctx.parents) == 1:
            grads = [grads]

        for t, g in zip(self._ctx.parents, grads):
            if g.shape != t.data.shape:
                print(f"grad shape must match tensor shape in {self._ctx}, {g.shape}, {t.data.shape}")
                assert(False)
            t.grad = g
            t.backward(False)

    def mean(self):
      div = Tensor(np.array([1/self.data.size]))
      return self.sum().mul(div)

# An instantiation of the Function is the context
class Function:
    def __init__(self, *tensors):
        self.parents = tensors #tensor
        self.saved_tensors = [] #ndarrays

    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)

    @staticmethod
    def apply(self, fxn, *x): #attach to Tensor object with appropriate fxn object
        x = (self,)+x
        ctx = fxn(*x)
        ret = Tensor(fxn.forward(ctx, *[t.data for t in x]))
        ret._ctx = ctx
        return ret

#attach apply method of Function to the Tensor class under the given name:
def register(name, fxn):
    setattr(Tensor, name, partialmethod(fxn.apply, fxn))

""" 
tensor.<name>(*args) actually calls with tuple of (self, args)

so in forward() except whatever arguments as ndarrays
"""

#Unaries:

class Sum(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.array([input.sum()]) #not as a scalar;

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return grad_output * np.ones_like(input)
register('sum', Sum)

class ReLU(Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return np.maximum(input, 0)

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = grad_output.copy()
    grad_input[input < 0] = 0
    return grad_input
register('relu', ReLU)

class LogSoftmax(Function):
  @staticmethod
  def forward(ctx, input):
    def logsumexp(x):
      c = x.max(axis=1)
      return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))
    output = input - logsumexp(input).reshape((-1, 1))
    ctx.save_for_backward(output)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    output, = ctx.saved_tensors
    return grad_output - np.exp(output)*grad_output.sum(axis=1).reshape((-1, 1))
register('logsoftmax', LogSoftmax)

#Binaries:

class Add(Function):
  @staticmethod
  def forward(ctx, x, y):
    return x + y
    
  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, grad_output
register('add', Add)

class Mul(Function):
  @staticmethod
  def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x*y

  @staticmethod
  def backward(ctx, grad_output):
    x,y = ctx.saved_tensors
    return y*grad_output, x*grad_output
register('mul', Mul)

class Dot(Function):
  @staticmethod
  def forward(ctx, input, weight):
    ctx.save_for_backward(input, weight)
    return np.dot(input, weight)

  @staticmethod
  def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    grad_input = grad_output.dot(weight.T)
    grad_weight = input.T.dot(grad_output)
    return grad_input, grad_weight
register('dot', Dot)