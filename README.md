# Topgrad

## Scope

Minimal pet autodiff framework for learning and tinkering. Single-file core (`tensor.py`) with a tiny set of NN ops, plus a few MNIST examples to prove it works. Not a full framework; just enough to explore ideas and compare against PyTorch.

The best part about having your own automatic differentiation framework is that you can name it. The entire library is just one file `tensor.py`, wrapping a backend array as a Tensor.

Design goal: keep the whole system small and readable so you can trace a forward pass, inspect saved tensors, and understand how backward is constructed. The core tensor engine is intentionally minimal, while examples and tests show what already works end-to-end (basic MLPs, conv nets, and attention on MNIST).

NN ops implemented:
- Sum
- Mean
- ReLU
- LogSoftmax
- Mul
- Add
- Linear: simple matmul in numpy
- Reshape
- Conv: GEMM via im2col
- BatchNorm: implemented (the backward is the tricky part)
- LayerNorm: simpler than BN, no running state
- Attention: surprisingly straightforward and effective

Unit tests compare against PyTorch at tol = 1e-4.

The optimizers are in `optim.py`. Currently supporting:
- SGD

## Backends

Supported:
- `numpy`
- `triton`

Select a backend when running examples:
```
python3 -m examples.train <mlp/conv/deepconv/attention> --backend <numpy/triton>
```

## Examples

1. simple MLP MNIST: `python3 -m examples.train mlp` (`examples/models/mlp.py`)
    - simple baseline
    - multilayer
    - nonlinearity
    - He init
2. conv MNIST: `python3 -m examples.train conv` (`examples/models/conv.py`)
    - strided convolution
    - flatten
3. deep conv MNIST with normalization: `python3 -m examples.train deepconv` (`examples/models/deepconv.py`)
    - resnet style
    - stemming, spatial (identity), strided convolutions
    - original: relu(batchnorm(conv(x)) + x)
    - newer: conv(batchnorm(relu(x))) + x, but you can't just swap this in without rewriting the architecture
4. attention MNIST: `python3 -m examples.train attention` (`examples/models/attention.py`)
    - transformer style
    - original: layernorm(x + attention(x))
    - new: x + attention(layernorm(x))
-----

## Ops derivations

High-level compute and memory costs (order-of-magnitude) for the forward+backward of each op. These are rough scaling laws intended for intuition and comparison, not exact FLOP counts. Memory refers to extra saved tensors needed for backward (excluding parameter storage), and costs are per batch unless noted.

| Op | Compute (fwd + bwd) | Extra memory |
| --- | --- | --- |
| Sum | `O(N)` | `O(1)` (shape) |
| Mean | `O(N)` | `O(1)` (shape) |
| ReLU | `O(N)` | `O(N)` (save mask or `x`) |
| LogSoftmax | `O(N)` | `O(N)` (save `y`) |
| Add | `O(N)` | `O(1)` |
| Mul | `O(N)` | `O(N)` (save `x,y`) |
| Reshape | `O(1)` | `O(1)` (shapes) |
| Linear | `O(B D M)` | `O(B D + D M)` (save `x, W`) |
| Conv (NHWC) | `O(B H W C_in C_out k^2)` | `O(B H W C_in k^2)` (im2col) |
| BatchNorm | `O(N)` | `O(N)` (save `x̂`, stats) |
| LayerNorm | `O(N)` | `O(N)` (save `x̂`, stats) |
| Attention (single-head) | `O(T^2 D)` | `O(T^2 + T D)` (save `A`, `QKV`) |

Notation:

| Symbol | Meaning |
| --- | --- |
| `N` | total elements |
| `B` | batch size |
| `D` | feature size |
| `M` | output size |
| `H/W` | spatial dims |
| `C_in/C_out` | channels |
| `k` | kernel size |
| `T` | sequence length |
| `A` | attention weights |
| `QKV` | query/key/value |

## Architecture

This framework is deliberately small and explicit, so you can trace every forward op and see exactly how gradients are produced. The core is a single Tensor class plus a tiny Op protocol, with backends swapping out the array/kernels underneath.

### Tensor object and graph nodes

There is no `requires_grad` flag. Any Tensor produced by an op becomes part of the graph, and gradients are computed for every parent when `backward()` is called.

Each Tensor holds:
- `data`: the backend array (numpy ndarray or a Triton-backed tensor)
- `grad`: accumulated gradient array (None until backward)
- `_op`: the Op instance that produced this Tensor (None for leaf Tensors)

In PyTorch, tensors with and without grad share the same abstraction (detach just drops the grad fn). Here there is a single Tensor abstraction whose storage is defined by the active backend (numpy or Triton), so there is no grad switch (a win for clarity).


### Op protocol and saved intermediates

`Op.apply` (topgrad/backend/op.py) is the bridge between raw arrays and Tensor objects:
- It collects all Tensor arguments as `parents`.
- It calls `forward` with raw arrays (Tensor.data).
- It returns a new Tensor whose `_op` points to this Op instance.
- Each Op stores any backward-needed arrays in `_intermediate` via `save_for_backward`.

```
Op.apply(x, y, ...)
  parents = [x, y]
  out = op.forward(x.data, y.data, ...)
  return Tensor(out) with _op=op
```

This keeps the graph explicit and minimal: every edge is a Tensor, every node is an Op.

### Backprop execution model

`Tensor.backward()` is a depth-first recursive traversal:
- If called on a Tensor with no `_op`, it stops.
- It seeds the gradient with ones (same shape as `data`) when `grad` is None.
- It calls `_op.backward(grad)` to get grads for each parent.
- It accumulates into each parent.grad and recurses.

```
z.backward()
  dz = ones_like(z)
  [dy, dx] = op.backward(dz)
  y.grad += dy; y.backward(False)
  x.grad += dx; x.backward(False)
```

Educational note: there is no topological sort or visited-set. On graphs with shared parents, recursion can revisit nodes and re-propagate grads. This is intentional simplicity, not a full DAG engine.

### Backend layer

Backends provide:
- a set of Ops (`backend.ops.<OpName>`)
- light helpers for non-autograd math in optim (`backend.add`, `backend.mul`)

Two backends are wired in:
- **numpy**: pure numpy arrays, easy to read and debug.
- **triton**: uses Triton kernels and a custom wrapper that blocks non-allowed PyTorch ops so compute stays in Triton.

Backend selection is global via `set_backend("numpy" | "triton")`.

### Models and training loop

Models are plain Python classes:
- weights are Tensors initialized with `Tensor.He`, `Tensor.zeros`, or `Tensor.ones`
- `forward` builds a graph by chaining Tensor ops
- `parameters()` returns a list of Tensors for the optimizer

Training (`examples/train.py`) does:
```
x -> model.forward -> logits
logits -> logsoftmax -> loss (mul with one-hot labels, sum, negate)
loss.backward()
SGD.step() + zero_grad()
```
