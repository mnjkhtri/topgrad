# Topgrad

The best part about having your own automatic differentiation framework is that you can name it. The entire library if just one file `tensor.py` which wraps the numpy ndarray as Tensor and in the fale file nn ops are implmeneted with numpy as backend for both forward and backward.

NN ops implemented:
- Softmax
- ReLU
- Reshape
- Linear: simply matmul in np
- Conv: as gemm using im2col approach
- Batch Norm: wtf is this backward?
- Layer Norm: so much simple than BN, no state, no momentum
- Add:
- Attention: way better than expected, easy win

with unit tests written comparing to pytorch at tol = 1e-4.

The optimizers are in `optim.py`. Currently supporting:
- SGD


## Examples

run examples as:
```
python3 -m examples.<insert name here>
```

1. simple mlp mnist: `examples/01_mlp_mnist.py`
    - simple stuff
    - multi layer
    - cute non linearity
    - He init
2. conv mnist: `examples/02_conv_mnist.py`
    - strided convolution
    - flatten
3. deep conv mnist with normalization: `examples/03_deep_conv_mnist.py`
    - resnet style
    - stemming, spatial (identity), strided convolutions
    - original: relu(batchnorm(conv(x)) + x)
    - apprarently there is new: conv(batchnorm(relu(x))) + x, but cant just switch in old architecture, need to rewrite all
4. attention mnist:
    - transformer style
    - original: layernorm(x + attention(x))
    - new: x + attention(layernorm(x))
5. vit minst:
    - todo
-----

The playground folder is where I play with concepts in PyTorch before porting to topgrad.