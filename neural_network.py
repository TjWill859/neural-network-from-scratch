"""
neural_network.py
-----------------
A feedforward neural network built from scratch using only NumPy.
Architecture: 784 → 128 → 64 → 10  (for MNIST digit classification)
"""

import numpy as np


# ─────────────────────────────────────────────
# Activation functions
# ─────────────────────────────────────────────

def relu(Z):
    """
    ReLU: max(0, Z) element-wise.
    Negative pre-activations get zeroed out — those neurons are "off".
    """
    return np.maximum(0, Z)


def relu_derivative(Z):
    """
    Derivative of ReLU.
    Where Z > 0, the gradient passes through unchanged (derivative = 1).
    Where Z <= 0, the gradient is blocked (derivative = 0).
    This is just a boolean mask cast to float.
    """
    return (Z > 0).astype(float)


def softmax(Z):
    """
    Softmax: turns raw scores into probabilities that sum to 1.
    We subtract the column max before exponentiating — this is called
    the 'numerically stable' trick. It prevents np.exp() from producing
    inf when Z contains large values. Mathematically equivalent to the
    plain formula but won't blow up.
    """
    shifted = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(shifted)
    return expZ / np.sum(expZ, axis=0, keepdims=True)


# ─────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────

def cross_entropy_loss(A_out, Y):
    """
    Cross-entropy loss for multi-class classification.

    A_out : (10, m) — predicted probabilities from softmax
    Y     : (10, m) — one-hot encoded true labels

    We clip A_out to avoid log(0) which would be -inf.
    The 1e-8 is tiny enough that it doesn't affect real predictions.
    """
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(np.clip(A_out, 1e-8, 1.0))) / m
    return loss


# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────

def one_hot_encode(Y, num_classes=10):
    """
    Converts a flat array of integer labels into a one-hot matrix.

    Example: label 3 with 10 classes → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    Y          : (m,)       — integer labels, e.g. [3, 7, 2, ...]
    Returns    : (10, m)    — one-hot matrix, one column per example
    """
    m = Y.shape[0]
    one_hot = np.zeros((num_classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot


def get_accuracy(A_out, Y_labels):
    """
    Computes classification accuracy.

    A_out    : (10, m) — predicted probabilities
    Y_labels : (m,)    — integer ground-truth labels (NOT one-hot)

    np.argmax along axis 0 picks the class with the highest probability
    for each column (each image), giving us the network's predicted digit.
    """
    predictions = np.argmax(A_out, axis=0)
    return np.mean(predictions == Y_labels)


# ─────────────────────────────────────────────
# The neural network class
# ─────────────────────────────────────────────

class NeuralNetwork:
    """
    A three-layer feedforward neural network.
    Layer sizes: 784 → 128 → 64 → 10

    All weights and biases live inside this object.
    Call train() to learn from data, predict() to run inference.
    """

    def __init__(self, layer_sizes=(784, 128, 64, 10), learning_rate=0.1):
        """
        Initialize weights with He initialization.

        He initialization sets weights to random values scaled by
        sqrt(2 / fan_in). This is specifically designed for ReLU networks
        — it keeps the variance of activations stable across layers so
        gradients neither vanish (become too small) nor explode (blow up)
        as they flow backward through the network.

        Using zeros would be wrong: all neurons would compute the same
        thing and learn identically — they'd never specialize.
        """
        self.lr = learning_rate
        n0, n1, n2, n3 = layer_sizes

        self.W1 = np.random.randn(n1, n0) * np.sqrt(2.0 / n0)  # (128, 784)
        self.b1 = np.zeros((n1, 1))                              # (128, 1)

        self.W2 = np.random.randn(n2, n1) * np.sqrt(2.0 / n1)  # (64, 128)
        self.b2 = np.zeros((n2, 1))                              # (64, 1)

        self.W3 = np.random.randn(n3, n2) * np.sqrt(2.0 / n2)  # (10, 64)
        self.b3 = np.zeros((n3, 1))                              # (10, 1)

        # These will be populated during the forward pass and read during
        # the backward pass — we store them on self so both methods can
        # access them without passing a giant tuple around.
        self.cache = {}

    # ── Forward pass ──────────────────────────

    def forward(self, X):
        """
        Run a forward pass through all three layers.

        X : (784, m) — pixel values for a batch of m images

        Each layer does:
          1. Z = W · A_prev + b   (linear combination)
          2. A = activation(Z)    (nonlinearity)

        We save every Z and A in self.cache because backprop needs them.
        Returns the final output A3: (10, m) probability distributions.
        """
        # Layer 1
        Z1 = self.W1 @ X + self.b1     # (128, m)
        A1 = relu(Z1)                   # (128, m)

        # Layer 2
        Z2 = self.W2 @ A1 + self.b2    # (64, m)
        A2 = relu(Z2)                   # (64, m)

        # Output layer
        Z3 = self.W3 @ A2 + self.b3    # (10, m)
        A3 = softmax(Z3)               # (10, m)

        # Store everything for the backward pass
        self.cache = {"X": X, "Z1": Z1, "A1": A1,
                               "Z2": Z2, "A2": A2,
                               "Z3": Z3, "A3": A3}
        return A3

    # ── Backward pass ─────────────────────────

    def backward(self, Y_onehot):
        """
        Run backpropagation to compute gradients for all weights.

        Y_onehot : (10, m) — one-hot encoded true labels

        We work from the output layer backward to the input layer,
        applying the chain rule at each step. The gradient at each layer
        depends on the gradient from the layer in front of it.
        """
        m = Y_onehot.shape[1]
        X  = self.cache["X"]
        Z1 = self.cache["Z1"];  A1 = self.cache["A1"]
        Z2 = self.cache["Z2"];  A2 = self.cache["A2"]
        Z3 = self.cache["Z3"];  A3 = self.cache["A3"]

        # ── Output layer gradient ──────────────────────────────────────
        # The combined derivative of softmax + cross-entropy simplifies
        # to just (prediction - truth). This is one of the most elegant
        # results in ML math — big errors produce large gradients, small
        # errors produce small gradients, automatically.
        dZ3 = A3 - Y_onehot                              # (10, m)
        dW3 = (1/m) * dZ3 @ A2.T                         # (10, 64)
        db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True) # (10, 1)

        # ── Hidden layer 2 gradient ────────────────────────────────────
        # Backpropagate the error signal from layer 3 through W3,
        # then kill gradients where ReLU was off (Z2 <= 0).
        dA2 = self.W3.T @ dZ3                             # (64, m)
        dZ2 = dA2 * relu_derivative(Z2)                  # (64, m)
        dW2 = (1/m) * dZ2 @ A1.T                         # (64, 128)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True) # (64, 1)

        # ── Hidden layer 1 gradient ────────────────────────────────────
        dA1 = self.W2.T @ dZ2                             # (128, m)
        dZ1 = dA1 * relu_derivative(Z1)                  # (128, m)
        dW1 = (1/m) * dZ1 @ X.T                          # (128, 784)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True) # (128, 1)

        self.grads = {"dW1": dW1, "db1": db1,
                      "dW2": dW2, "db2": db2,
                      "dW3": dW3, "db3": db3}

    # ── Weight update ──────────────────────────

    def update(self):
        """
        Gradient descent: nudge every weight opposite to its gradient.
        A large positive gradient means the weight is making the loss
        bigger — so we subtract, reducing its influence.
        """
        self.W1 -= self.lr * self.grads["dW1"]
        self.b1 -= self.lr * self.grads["db1"]
        self.W2 -= self.lr * self.grads["dW2"]
        self.b2 -= self.lr * self.grads["db2"]
        self.W3 -= self.lr * self.grads["dW3"]
        self.b3 -= self.lr * self.grads["db3"]

    # ── Full training loop ─────────────────────

    def train(self, X_train, Y_train, X_val, Y_val,
              epochs=300, batch_size=64):
        """
        Mini-batch stochastic gradient descent training loop.

        Instead of using all 60,000 images per update (too slow) or just
        1 image (too noisy), we use mini-batches of 64. Each epoch shuffles
        the data and iterates through all batches.

        X_train  : (784, 60000)  — training images
        Y_train  : (60000,)      — training labels (integers)
        X_val    : (784, 10000)  — validation images
        Y_val    : (10000,)      — validation labels (integers)
        """
        m = X_train.shape[1]
        history = {"train_loss": [], "val_loss": [],
                   "train_acc":  [], "val_acc":  []}

        for epoch in range(epochs):
            # Shuffle training data at the start of each epoch so
            # batches are different every time — prevents the network
            # from memorizing the order of examples.
            perm = np.random.permutation(m)
            X_shuffled = X_train[:, perm]
            Y_shuffled = Y_train[perm]

            # ── Mini-batch loop ────────────────────────────────────────
            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[:, start:end]
                Y_batch = Y_shuffled[start:end]
                Y_onehot = one_hot_encode(Y_batch)

                # The three-step dance: forward → backward → update
                self.forward(X_batch)
                self.backward(Y_onehot)
                self.update()

            # ── Epoch metrics (run on full sets, no update) ────────────
            A_train = self.forward(X_train)
            train_loss = cross_entropy_loss(A_train, one_hot_encode(Y_train))
            train_acc  = get_accuracy(A_train, Y_train)

            A_val = self.forward(X_val)
            val_loss = cross_entropy_loss(A_val, one_hot_encode(Y_val))
            val_acc  = get_accuracy(A_val, Y_val)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:>3} | "
                      f"Train loss: {train_loss:.4f}  acc: {train_acc:.3f} | "
                      f"Val loss: {val_loss:.4f}  acc: {val_acc:.3f}")

        return history

    # ── Inference ──────────────────────────────

    def predict(self, X):
        """
        Returns the predicted digit class for each image in X.
        argmax picks the index (0–9) with the highest probability.
        """
        A_out = self.forward(X)
        return np.argmax(A_out, axis=0)

    # ── Save / load weights ────────────────────

    def save(self, path="model_weights.npz"):
        """Save all weights to a .npz file so you don't have to retrain."""
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)
        print(f"Weights saved to {path}")

    def load(self, path="model_weights.npz"):
        """Load previously saved weights."""
        data = np.load(path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self.W3, self.b3 = data["W3"], data["b3"]
        print(f"Weights loaded from {path}")