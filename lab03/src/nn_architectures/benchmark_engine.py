import time
import torch
from thop import profile as thop_profile

def evaluate_systems_footprint(model, input_shape=(1, 1, 28, 28), num_evals=75):
    """Audits parameter capacity, operational FLOPs, and runtime latency."""
    model.eval()
    sample = torch.randn(*input_shape)
    
    n_params = sum(p.numel() for p in model.parameters())
    macs, _ = thop_profile(model, inputs=(sample,), verbose=False)
    
    with torch.no_grad():
        for _ in range(15):
            _ = model(sample)
        
        t0 = time.perf_counter()
        for _ in range(num_evals):
            _ = model(sample)
        elapsed = time.perf_counter() - t0
        
    latency = (elapsed / num_evals) * 1000.0
    return {
        "parameters": n_params,
        "macs": int(macs),
        "mflops": (macs * 2) / 1e6,
        "latency_ms": latency
    }
