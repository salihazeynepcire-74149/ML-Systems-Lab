import numpy as np

class ExponentialLinearUnit:
    """ELU non-linear activation: alpha * (exp(x) - 1) for x <= 0; x for x > 0."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.cached_out = None

    def forward(self, z):
        self.cached_out = np.where(z > 0, z, self.alpha * (np.exp(np.clip(z, -20.0, 0.0)) - 1.0))
        return self.cached_out

    def backward(self, delta):
        grad = np.where(self.cached_out > 0, 1.0, self.cached_out + self.alpha)
        return delta * grad

class SoftmaxCrossEntropyCriterion:
    """Multiclass loss with log-sum-exp stability."""
    @staticmethod
    def compute(logits, targets):
        n = logits.shape[0]
        m = np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(logits - m)
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        loss = -np.sum(np.log(np.clip(probs[np.arange(n), targets], 1e-12, 1.0))) / n
        return loss, probs

    @staticmethod
    def gradient(probs, targets):
        n = probs.shape[0]
        d_logits = probs.copy()
        d_logits[np.arange(n), targets] -= 1.0
        return d_logits
