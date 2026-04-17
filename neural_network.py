"""
Neural network from scratch — pure NumPy.
Architecture: 784 → 128 → 64 → 10
"""

import numpy as np


# ── Activations ───────────────────────────────────────────

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    # Subtract max per column for numerical stability
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)


# ── Loss ──────────────────────────────────────────────────

def cross_entropy_loss(A_out, Y):
    m = Y.shape[1]
    return -np.sum(Y * np.log(np.clip(A_out, 1e-8, 1.0))) / m


# ── Helpers ───────────────────────────────────────────────

def one_hot_encode(Y, num_classes=10):
    m = Y.shape[0]
    one_hot = np.zeros((num_classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot

def get_accuracy(A_out, Y_labels):
    return np.mean(np.argmax(A_out, axis=0) == Y_labels)


# ── Network ───────────────────────────────────────────────

class NeuralNetwork:

    def __init__(self, layer_sizes=(784, 128, 64, 10), learning_rate=0.1):
        self.lr = learning_rate
        n0, n1, n2, n3 = layer_sizes

        # He initialization — keeps variance stable through ReLU layers
        self.W1 = np.random.randn(n1, n0) * np.sqrt(2.0 / n0)
        self.b1 = np.zeros((n1, 1))
        self.W2 = np.random.randn(n2, n1) * np.sqrt(2.0 / n1)
        self.b2 = np.zeros((n2, 1))
        self.W3 = np.random.randn(n3, n2) * np.sqrt(2.0 / n2)
        self.b3 = np.zeros((n3, 1))

        self.cache = {}
        self.grads = {}

    def forward(self, X):
        Z1 = self.W1 @ X + self.b1
        A1 = relu(Z1)
        Z2 = self.W2 @ A1 + self.b2
        A2 = relu(Z2)
        Z3 = self.W3 @ A2 + self.b3
        A3 = softmax(Z3)

        self.cache = {"X": X, "Z1": Z1, "A1": A1,
                               "Z2": Z2, "A2": A2, "A3": A3}
        return A3

    def backward(self, Y_onehot):
        m  = Y_onehot.shape[1]
        X  = self.cache["X"]
        Z1 = self.cache["Z1"];  A1 = self.cache["A1"]
        Z2 = self.cache["Z2"];  A2 = self.cache["A2"]
        Z3 = self.cache["Z3"];  A3 = self.cache["A3"]

        # Softmax + cross-entropy gradient simplifies to A - Y
        dZ3 = A3 - Y_onehot
        dW3 = (1/m) * dZ3 @ A2.T
        db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)

        dA2 = self.W3.T @ dZ3
        dZ2 = dA2 * relu_derivative(Z2)
        dW2 = (1/m) * dZ2 @ A1.T
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = (1/m) * dZ1 @ X.T
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        self.grads = {"dW1": dW1, "db1": db1,
                      "dW2": dW2, "db2": db2,
                      "dW3": dW3, "db3": db3}

    def update(self):
        self.W1 -= self.lr * self.grads["dW1"]
        self.b1 -= self.lr * self.grads["db1"]
        self.W2 -= self.lr * self.grads["dW2"]
        self.b2 -= self.lr * self.grads["db2"]
        self.W3 -= self.lr * self.grads["dW3"]
        self.b3 -= self.lr * self.grads["db3"]

    def train(self, X_train, Y_train, X_val, Y_val,
              epochs=300, batch_size=64):
        m = X_train.shape[1]
        history = {"train_loss": [], "val_loss": [],
                   "train_acc":  [], "val_acc":  []}

        for epoch in range(epochs):
            # Shuffle each epoch so batches differ
            perm = np.random.permutation(m)
            X_shuffled = X_train[:, perm]
            Y_shuffled = Y_train[perm]

            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch  = X_shuffled[:, start:end]
                Y_onehot = one_hot_encode(Y_shuffled[start:end])

                self.forward(X_batch)
                self.backward(Y_onehot)
                self.update()

            A_train    = self.forward(X_train)
            train_loss = cross_entropy_loss(A_train, one_hot_encode(Y_train))
            train_acc  = get_accuracy(A_train, Y_train)

            A_val    = self.forward(X_val)
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

    def predict(self, X):
        return np.argmax(self.forward(X), axis=0)

    def save(self, path="model_weights.npz"):
        np.savez(path, W1=self.W1, b1=self.b1,
                       W2=self.W2, b2=self.b2,
                       W3=self.W3, b3=self.b3)
        print(f"Saved to {path}")

    def load(self, path="model_weights.npz"):
        data = np.load(path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self.W3, self.b3 = data["W3"], data["b3"]