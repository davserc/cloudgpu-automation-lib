#!/usr/bin/env python3
"""Simple PyTorch GPU test."""

import sys
import time
from datetime import datetime
from pathlib import Path

import torch


class Logger:
    """Simple logger that writes to both stdout and file."""

    def __init__(self, log_path: str = "/root/logs/pytorch_test.log"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_path, "w")
        self.writeln(f"PyTorch Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def write(self, msg: str):
        print(msg, end="")
        self.file.write(msg)
        self.file.flush()

    def writeln(self, msg: str = ""):
        self.write(msg + "\n")

    def close(self):
        self.file.close()


def main():
    log = Logger()

    log.writeln("=" * 50)
    log.writeln("PyTorch GPU Test")
    log.writeln("=" * 50)

    # Check CUDA availability
    log.writeln(f"\nPyTorch version: {torch.__version__}")
    log.writeln(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        log.writeln("ERROR: CUDA not available!")
        log.writeln(f"\nLog saved to: {log.log_path}")
        log.close()
        return 1

    # GPU info
    log.writeln(f"CUDA version: {torch.version.cuda}")
    log.writeln(f"GPU count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        log.writeln(f"\nGPU {i}: {props.name}")
        log.writeln(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        log.writeln(f"  Compute capability: {props.major}.{props.minor}")

    # Simple benchmark
    log.writeln("\n" + "-" * 50)
    log.writeln("Running matrix multiplication benchmark...")

    device = torch.device("cuda")
    size = 4096

    # Warmup
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(10):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tflops = (2 * size**3 * 10) / elapsed / 1e12
    log.writeln(f"Matrix size: {size}x{size}")
    log.writeln(f"Time for 10 iterations: {elapsed:.3f}s")
    log.writeln(f"Performance: {tflops:.2f} TFLOPS")

    log.writeln("\n" + "=" * 50)
    log.writeln("SUCCESS: PyTorch GPU test passed!")
    log.writeln("=" * 50)

    log.writeln(f"\nLog saved to: {log.log_path}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
