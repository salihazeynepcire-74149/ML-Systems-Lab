import numpy as np
from .affine import FullyConnectedLayer
from .activations import ExponentialLinearUnit, SoftmaxCrossEntropyCriterion

class ModularDeepPerceptron:
    """Vectorized feed-forward neural architecture with ELU activations."""
    def __init__(self, in_dim=64, hidden1=48, hidden2=24, num_classes=10, seed=74149):
        self.fc1 = FullyConnectedLayer(in_dim, hidden1, seed=seed)
        self.act1 = ExponentialLinearUnit(alpha=1.0)
        self.fc2 = FullyConnectedLayer(hidden1, hidden2, seed=seed + 1)
        self.act2 = ExponentialLinearUnit(alpha=1.0)
        self.fc3 = FullyConnectedLayer(hidden2, num_classes, seed=seed + 2)
        self.cached_probs = None
        self.cached_targets = None

    def forward(self, x, targets=None):
        h1 = self.act1.forward(self.fc1.forward(x))
        h2 = self.act2.forward(self.fc2.forward(h1))
        logits = self.fc3.forward(h2)

        if targets is not None:
            loss, probs = SoftmaxCrossEntropyCriterion.compute(logits, targets)
            self.cached_probs = probs
            self.cached_targets = targets
            return loss, probs
        m = np.max(logits, axis=1, keepdims=True)
        exp_v = np.exp(logits - m)
        return exp_v / np.sum(exp_v, axis=1, keepdims=True)

    def backward(self):
        d_logits = SoftmaxCrossEntropyCriterion.gradient(self.cached_probs, self.cached_targets)
        d_h2 = self.fc3.backward(d_logits)
        d_z2 = self.act2.backward(d_h2)
        d_h1 = self.fc2.backward(d_z2)
        d_z1 = self.act1.backward(d_h1)
        self.fc1.backward(d_z1)

    def optimize(self, lr):
        self.fc1.apply_gradients(lr)
        self.fc2.apply_gradients(lr)
        self.fc3.apply_gradients(lr)
