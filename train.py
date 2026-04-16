"""
train.py
--------
Loads MNIST, preprocesses it, trains the neural network, and plots results.
Run this file: python train.py
"""

import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork, get_accuracy, one_hot_encode


# ─────────────────────────────────────────────
# Step 1: Load and preprocess MNIST
# ─────────────────────────────────────────────

def load_mnist():
    """
    Downloads MNIST via Keras (data only — no model code) and preprocesses it.

    Preprocessing steps:
      1. Flatten: 28×28 images → 784-length vectors
      2. Normalize: pixel values 0–255 → 0.0–1.0
         Neural networks train much better when inputs are small numbers.
         Large inputs lead to large activations, large gradients, and
         unstable training. Dividing by 255 is the simplest fix.
      3. Transpose: (m, 784) → (784, m)
         Our network expects each column to be one image. Keras gives us
         each row as one image, so we flip it.

    Returns X_train (784, 60000), Y_train (60000,),
            X_val   (784, 10000), Y_val   (10000,)
    """
    from sklearn.datasets import fetch_openml
    print("Downloading MNIST (this takes ~30 seconds the first time)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

    X = mnist.data / 255.0        # normalize 0–255 → 0.0–1.0
    Y = mnist.target.astype(int)  # labels come as strings, convert to int

    # Split into 60k train / 10k validation (same as before)
    X_train, X_val = X[:60000], X[60000:]
    Y_train, Y_val = Y[:60000], Y[60000:]

    # Transpose so each column = one image: (784, m)
    X_train = X_train.T
    X_val   = X_val.T

    print(f"Training set:   {X_train.shape}  labels: {Y_train.shape}")
    print(f"Validation set: {X_val.shape}  labels: {Y_val.shape}")
    return X_train, Y_train, X_val, Y_val


# ─────────────────────────────────────────────
# Step 2: Plot training curves
# ─────────────────────────────────────────────

def plot_training_curves(history):
    """
    Plots loss and accuracy over training epochs.

    Two things to look for:
    - Loss should decrease and flatten out (converging)
    - Train acc and val acc should stay close together
      (if val acc drops while train acc rises → overfitting)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(len(history["train_loss"]))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   label="Val loss",   linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Loss over training")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], label="Train accuracy", linewidth=2)
    ax2.plot(epochs, history["val_acc"],   label="Val accuracy",   linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy over training")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Saved: training_curves.png")


# ─────────────────────────────────────────────
# Step 3: Visualize sample predictions
# ─────────────────────────────────────────────

def plot_sample_predictions(nn, X_val, Y_val, num_samples=16):
    """
    Picks 16 random validation images and shows what the network predicted.
    Green title = correct, red title = wrong.
    Great for seeing what kinds of digits the network struggles with.
    """
    indices = np.random.choice(X_val.shape[1], num_samples, replace=False)
    X_sample = X_val[:, indices]
    Y_sample = Y_val[indices]

    predictions = nn.predict(X_sample)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img = X_sample[:, i].reshape(28, 28)
        ax.imshow(img, cmap="gray")
        pred = predictions[i]
        true = Y_sample[i]
        color = "green" if pred == true else "red"
        ax.set_title(f"Pred: {pred}  True: {true}", color=color, fontsize=9)
        ax.axis("off")

    plt.suptitle("Sample predictions (green=correct, red=wrong)", y=1.01)
    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: sample_predictions.png")


# ─────────────────────────────────────────────
# Step 4: Confusion matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(nn, X_val, Y_val):
    """
    A 10×10 grid where cell (i, j) = how many times the network
    predicted digit j when the true digit was i.

    Perfect classifier: all mass on the diagonal.
    Off-diagonal entries show which digits get confused with which.
    Common confusions: 4 vs 9, 3 vs 8, 7 vs 1.
    """
    preds = nn.predict(X_val)
    cm = np.zeros((10, 10), dtype=int)
    for true, pred in zip(Y_val, preds):
        cm[true, pred] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)

    # Add text annotations inside each cell
    for i in range(10):
        for j in range(10):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=8, color=color)

    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title("Confusion matrix (validation set)")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")


# ─────────────────────────────────────────────
# Main: run everything
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Load data
    print("=" * 50)
    print("Loading MNIST...")
    print("=" * 50)
    X_train, Y_train, X_val, Y_val = load_mnist()

    # Build and train network
    print("\n" + "=" * 50)
    print("Training neural network...")
    print("=" * 50)
    nn = NeuralNetwork(
        layer_sizes=(784, 128, 64, 10),
        learning_rate=0.1
    )

    history = nn.train(
        X_train, Y_train,
        X_val,   Y_val,
        epochs=300,
        batch_size=64
    )

    # Final evaluation
    print("\n" + "=" * 50)
    print("Final results")
    print("=" * 50)
    final_train_acc = get_accuracy(nn.forward(X_train), Y_train)
    final_val_acc   = get_accuracy(nn.forward(X_val),   Y_val)
    print(f"Train accuracy: {final_train_acc * 100:.2f}%")
    print(f"Val accuracy:   {final_val_acc   * 100:.2f}%")

    # Save weights so you don't have to retrain every time
    nn.save("model_weights.npz")

    # Plots
    print("\nGenerating plots...")
    plot_training_curves(history)
    plot_sample_predictions(nn, X_val, Y_val)
    plot_confusion_matrix(nn, X_val, Y_val)

    print("\nDone! Check training_curves.png, sample_predictions.png,")
    print("and confusion_matrix.png in your working directory.")