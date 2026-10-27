import numpy as np

def verify_gradients(network, x_batch, y_batch, epsilon=1e-5):
    """Calculates relative difference between analytical gradients and numerical approximations."""
    network.forward(x_batch, y_batch)
    network.backward()
    
    max_err = 0.0
    for layer in [network.fc1, network.fc2, network.fc3]:
        flat_w = layer.W.reshape(-1)
        flat_ana = layer.grad_W.reshape(-1)
        
        sample_indices = np.random.RandomState(42).choice(flat_w.size, size=min(10, flat_w.size), replace=False)
        for idx in sample_indices:
            orig = flat_w[idx]
            flat_w[idx] = orig + epsilon
            l_plus, _ = network.forward(x_batch, y_batch)
            flat_w[idx] = orig - epsilon
            l_minus, _ = network.forward(x_batch, y_batch)
            flat_w[idx] = orig
            
            num_grad = (l_plus - l_minus) / (2.0 * epsilon)
            diff = abs(flat_ana[idx] - num_grad)
            scale = max(abs(flat_ana[idx]), abs(num_grad), 1e-8)
            rel = diff / scale
            if rel > max_err:
                max_err = rel
    return max_err
