# TOPGRAD

The best part about having your own automatic differentiation framework is that you can name it.

### Runs MNIST:

```Python
PYTHONPATH="./" python3 test/mnist.py
```

### How about ResNet in Torch? (runs RN101 v1.5)

```Python
python3 playground/resnet.py <insert url here>
```

### Well I got Transformers as well

Trains GPT in addition dataset
```Python
python3 playground/topgpt/addition.py
```

Load pretrained OpenAI GPT2 (small and medium, minor changes for larger)
```Python
python3 playground/topgpt/gpt2.py
```