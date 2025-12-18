#!/usr/bin/env python3
"""Simple PyTorch GPU test."""

import time

import torch


def main():
    print("=" * 50)
    print("PyTorch GPU Test")
    print("=" * 50)

    # Check CUDA availability
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return 1

    # GPU info
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")

    # Simple benchmark
    print("\n" + "-" * 50)
    print("Running matrix multiplication benchmark...")

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
    print(f"Matrix size: {size}x{size}")
    print(f"Time for 10 iterations: {elapsed:.3f}s")
    print(f"Performance: {tflops:.2f} TFLOPS")

    print("\n" + "=" * 50)
    print("SUCCESS: PyTorch GPU test passed!")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    exit(main())
