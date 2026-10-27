import numpy as np

class AffineMap:
    def __init__(self, in_features, out_features, seed=42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.cache_input = None

    def forward(self, X):
        self.cache_input = X
        return np.dot(X, self.W) + self.b

    def backward(self, dOut):
        N = self.cache_input.shape[0]
        self.dW = np.dot(self.cache_input.T, dOut) / N
        self.db = np.sum(dOut, axis=0, keepdims=True) / N
        return np.dot(dOut, self.W.T)

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

class RectifiedLinearUnit:
    def __init__(self):
        self.active_mask = None

    def forward(self, Z):
        self.active_mask = (Z > 0)
        return np.maximum(0, Z)

    def backward(self, dOut):
        return dOut * self.active_mask

class VectorizedClassifier:
    def __init__(self, in_dim, hidden_dim, out_dim, seed=42):
        self.l1 = AffineMap(in_dim, hidden_dim, seed=seed)
        self.a1 = RectifiedLinearUnit()
        self.l2 = AffineMap(hidden_dim, hidden_dim, seed=seed+1)
        self.a2 = RectifiedLinearUnit()
        self.l3 = AffineMap(hidden_dim, out_dim, seed=seed+2)

    def forward(self, X, Y=None):
        out1 = self.l1.forward(X)
        act1 = self.a1.forward(out1)
        out2 = self.l2.forward(act1)
        act2 = self.a2.forward(out2)
        logits = self.l3.forward(act2)

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        loss = None
        if Y is not None:
            eps = 1e-12
            loss = -np.mean(np.sum(Y * np.log(probs + eps), axis=1))

        self.probs = probs
        self.Y = Y
        return logits, probs, loss

    def backward(self):
        dZ3 = self.probs - self.Y
        dH2 = self.l3.backward(dZ3)
        dZ2 = self.a2.backward(dH2)
        dH1 = self.l2.backward(dZ2)
        dZ1 = self.a1.backward(dH1)
        dX = self.l1.backward(dZ1)
        return dX

    def step(self, lr):
        self.l1.step(lr)
        self.l2.step(lr)
        self.l3.step(lr)
