import argparse

import numpy as np

from examples.models.attention import TopAttentionNet
from examples.models.conv import TopConvNet
from examples.models.deepconv import TopDeepConvNet
from examples.models.mlp import TopMLPNet
from examples.utils import MNIST
from topgrad.backend import get_backend, set_backend
from topgrad.optim import SGD
from topgrad.tensor import Tensor

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib is optional for examples
    plt = None

MODEL_SPECS = {
    "mlp": {
        "factory": TopMLPNet,
        "input_shape": (28 * 28,),
        "batch_size": 256,
        "lr": 0.0001,
    },
    "conv": {
        "factory": TopConvNet,
        "input_shape": (28, 28, 1),
        "batch_size": 32,
        "lr": 0.0001,
    },
    "deepconv": {
        "factory": TopDeepConvNet,
        "input_shape": (28, 28, 1),
        "batch_size": 8,
        "lr": 0.0001,
    },
    "attention": {
        "factory": TopAttentionNet,
        "input_shape": (28, 28),
        "batch_size": 256,
        "lr": 0.0001,
    },
}


def run_training(model, X, Y, X_valid, Y_valid, batch_size, lr):
    backend = get_backend()
    optim = SGD(model.parameters(), lr=lr)

    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []
    steps = []

    def to_scalar(x):
        return float(np.asarray(x).item())

    def plot_metrics():
        if plt is None:
            print("matplotlib is not installed; skipping plots.")
            return
        plt.figure(figsize=(12, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(steps, train_losses, label="train")
        plt.plot(steps, valid_losses, label="valid")
        plt.xlabel("steps")
        plt.ylabel("loss")
        plt.title("train/valid loss")
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(steps, train_accuracies, label="train")
        plt.plot(steps, valid_accuracies, label="valid")
        plt.xlabel("steps")
        plt.ylabel("accuracy")
        plt.title("train/valid accuracy")
        plt.legend()

        plt.show()

    def eval_batch(sample_size):
        assert sample_size < X_valid.shape[0]

        samp = np.random.randint(0, X_valid.shape[0], size=sample_size)
        x_batch = np.ascontiguousarray(X_valid[samp])
        y_batch = np.ascontiguousarray(Y_valid[samp])
        x, y_probs = Tensor(backend.wrap(x_batch)), Tensor(backend.wrap(y_batch))
        log_probs = model.forward(x).logsoftmax()
        loss = log_probs.mul(y_probs).sum().mul(-1.0)
        # no backward

        y_labels = np.argmax(backend.unwrap(y_probs.data), axis=1)
        y_pred_labels = np.argmax(backend.unwrap(log_probs.data), axis=1)

        accuracy = 100 * np.mean(y_labels == y_pred_labels)
        return to_scalar(backend.unwrap(loss.data)), accuracy

    i = 0
    try:
        print("Starting training... Press Ctrl+C to stop.")
        while True:
            samp = np.random.randint(0, X.shape[0], size=batch_size)
            x_batch = np.ascontiguousarray(X[samp])
            y_batch = np.ascontiguousarray(Y[samp])
            x, y_probs = Tensor(backend.wrap(x_batch)), Tensor(backend.wrap(y_batch))

            log_probs = model.forward(x).logsoftmax()
            loss = log_probs.mul(y_probs).sum().mul(-1.0)

            loss.backward()
            optim.step()
            optim.zero_grad()

            if i % 1 == 0:
                # TRAIN:
                y_labels = np.argmax(backend.unwrap(y_probs.data), axis=1)
                y_pred_labels = np.argmax(backend.unwrap(log_probs.data), axis=1)
                accuracy = 100 * np.mean(y_labels == y_pred_labels)

                # VALID:
                eval_loss, eval_accuracy = eval_batch(batch_size)

                print(
                    f"minbatch {i:5d} | loss (train/valid): {to_scalar(backend.unwrap(loss.data)):.4f}, {eval_loss:.4f} | accuracy(train/valid): {accuracy:.2f}, {eval_accuracy:.2f}"
                )

            steps.append(i)
            train_losses.append(to_scalar(backend.unwrap(loss.data)))
            valid_losses.append(eval_loss)
            train_accuracies.append(accuracy)
            valid_accuracies.append(eval_accuracy)

            i += 1

    except KeyboardInterrupt:
        print(
            "\n\nTraining interrupted by user. Thanks for training with us. Here is your plot:"
        )
        plot_metrics()


def run_mnist(model_factory, input_shape, batch_size, lr, backend="numpy"):
    set_backend(backend)
    X, Y = MNIST.load_data(input_shape=input_shape)
    X_valid, Y_valid = MNIST.load_data(train=False, input_shape=input_shape)
    model = model_factory()
    run_training(model, X, Y, X_valid, Y_valid, batch_size, lr)


def main():
    parser = argparse.ArgumentParser(description="Train a model on MNIST.")
    parser.add_argument("model", help="Model name (mlp/conv/deepconv/attention).")
    parser.add_argument(
        "--backend", default="numpy", help="Backend name (numpy/triton)."
    )
    args = parser.parse_args()

    name = args.model.strip().lower()
    spec = MODEL_SPECS.get(name)
    if spec is None:
        print(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_SPECS)}")
        return

    run_mnist(
        spec["factory"],
        spec["input_shape"],
        spec["batch_size"],
        lr=spec["lr"],
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
