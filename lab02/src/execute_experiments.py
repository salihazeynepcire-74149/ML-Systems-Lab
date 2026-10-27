import os
import sys
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modular_network.network import ModularDeepPerceptron
from modular_network.verifier import verify_gradients

def main():
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=74149)

    model = ModularDeepPerceptron(in_dim=64, hidden1=48, hidden2=24, num_classes=10, seed=74149)
    rel_err = verify_gradients(model, X_train[:32], y_train[:32])
    print(f"[CHECK] Gradient Relative Error: {rel_err:.4e}")

    start = time.perf_counter()
    epochs = 40
    batch_size = 32
    for ep in range(epochs):
        perm = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            b_idx = perm[i:i+batch_size]
            model.forward(X_train[b_idx], y_train[b_idx])
            model.backward()
            model.optimize(lr=0.07)
    duration = time.perf_counter() - start

    preds = np.argmax(model.forward(X_test), axis=1)
    acc = np.mean(preds == y_test) * 100.0
    print(f"[METRIC] Accuracy: {acc:.2f}% | Elapsed Time: {duration:.3f}s")

if __name__ == "__main__":
    main()
