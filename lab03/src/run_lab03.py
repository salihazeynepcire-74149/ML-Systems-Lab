import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nn_architectures.perceptron import SalihaMLP
from nn_architectures.convnet import SalihaCNN
from nn_architectures.transformer import SalihaViT
from nn_architectures.benchmark_engine import evaluate_systems_footprint

def run_benchmarks():
    print("[INFO] Auditing Saliha Cire Vision Architectures...")
    models = {
        "ELU Perceptron": SalihaMLP(),
        "Deep 2-Stage CNN": SalihaCNN(),
        "CLS Vision Transformer": SalihaViT()
    }

    metrics = {}
    for name, net in models.items():
        res = evaluate_systems_footprint(net)
        metrics[name] = res
        print(f"[{name}] Parameters: {res['parameters']:,} | MACs: {res['macs']:,} | Latency: {res['latency_ms']:.3f} ms")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "systems_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[DONE] Benchmark results exported to {out_dir}")

if __name__ == "__main__":
    run_benchmarks()
