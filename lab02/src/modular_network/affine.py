import numpy as np

class FullyConnectedLayer:
    """Affine projection layer with Gaussian initialization and momentum-ready gradients."""
    def __init__(self, in_features, out_features, seed=74149):
        rng = np.random.RandomState(seed)
        sigma = np.sqrt(2.0 / in_features)
        self.W = rng.randn(in_features, out_features) * sigma
        self.b = np.zeros((1, out_features))
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        self.inputs = None

    def forward(self, x):
        self.inputs = x
        return np.dot(x, self.W) + self.b

    def backward(self, delta):
        n = self.inputs.shape[0]
        self.grad_W = np.dot(self.inputs.T, delta) / n
        self.grad_b = np.sum(delta, axis=0, keepdims=True) / n
        return np.dot(delta, self.W.T)

    def apply_gradients(self, lr):
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b
