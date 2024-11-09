# TOPGRAD

The best part about having your own automatic differentiation framework is that you can name it.

### Runs MNIST:

```bash
PYTHONPATH="./" python3 test/mnist.py
```

-----

The playground folder is where I implement models on PyTorch before porting to topgrad so cd into it (:

### How about ResNet in Torch? (runs RN101 v1.5)

```bash
python3 resnet.py <insert url here>
```

### Well I got Transformers as well

Trains GPT in addition dataset
```bash
python3 topgpt/addition.py
```

Load pretrained OpenAI GPT2 (small and medium, minor changes for larger)
```bash
python3 topgpt/gpt2.py
```

### Inference on ViT:

```bash
python3 vit.py <insert url here>
```