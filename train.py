"""
train.py — load MNIST, train, evaluate, plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork, get_accuracy, one_hot_encode


def load_mnist():
    from sklearn.datasets import fetch_openml
    print("Downloading MNIST (this takes ~30 seconds the first time)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

    X = mnist.data / 255.0
    Y = mnist.target.astype(int)

    X_train, X_val = X[:60000].T, X[60000:].T
    Y_train, Y_val = Y[:60000],   Y[60000:]

    print(f"Training set:   {X_train.shape}  labels: {Y_train.shape}")
    print(f"Validation set: {X_val.shape}  labels: {Y_val.shape}")
    return X_train, Y_train, X_val, Y_val


def plot_training_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(len(history["train_loss"]))

    ax1.plot(epochs, history["train_loss"], label="Train loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   label="Val loss",   linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Loss over training"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train accuracy", linewidth=2)
    ax2.plot(epochs, history["val_acc"],   label="Val accuracy",   linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy over training"); ax2.legend()
    ax2.grid(alpha=0.3); ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()


def plot_sample_predictions(nn, X_val, Y_val, num_samples=16):
    indices  = np.random.choice(X_val.shape[1], num_samples, replace=False)
    X_sample = X_val[:, indices]
    Y_sample = Y_val[indices]
    preds    = nn.predict(X_sample)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_sample[:, i].reshape(28, 28), cmap="gray")
        color = "green" if preds[i] == Y_sample[i] else "red"
        ax.set_title(f"Pred: {preds[i]}  True: {Y_sample[i]}", color=color, fontsize=9)
        ax.axis("off")

    plt.suptitle("Sample predictions (green=correct, red=wrong)", y=1.01)
    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(nn, X_val, Y_val):
    preds = nn.predict(X_val)
    cm = np.zeros((10, 10), dtype=int)
    for true, pred in zip(Y_val, preds):
        cm[true, pred] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)

    for i in range(10):
        for j in range(10):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_xlabel("Predicted digit"); ax.set_ylabel("True digit")
    ax.set_title("Confusion matrix (validation set)")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    print("=" * 50)
    print("Loading MNIST...")
    print("=" * 50)
    X_train, Y_train, X_val, Y_val = load_mnist()

    print("\n" + "=" * 50)
    print("Training neural network...")
    print("=" * 50)
    nn = NeuralNetwork(layer_sizes=(784, 128, 64, 10), learning_rate=0.1)
    history = nn.train(X_train, Y_train, X_val, Y_val, epochs=300, batch_size=64)

    print("\n" + "=" * 50)
    print("Final results")
    print("=" * 50)
    print(f"Train accuracy: {get_accuracy(nn.forward(X_train), Y_train) * 100:.2f}%")
    print(f"Val accuracy:   {get_accuracy(nn.forward(X_val),   Y_val)   * 100:.2f}%")

    nn.save("model_weights.npz")

    print("\nGenerating plots...")
    plot_training_curves(history)
    plot_sample_predictions(nn, X_val, Y_val)
    plot_confusion_matrix(nn, X_val, Y_val)