#!/usr/bin/env python3
"""Direct CUDA test using subprocess to call nvidia-smi and nvcc."""

import subprocess
import sys


def run_cmd(cmd, description):
    """Run a command and print output."""
    print(f"\n{description}")
    print("-" * 50)
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0 and result.stderr:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Command timed out")
        return False
    except Exception as e:
        print(f"Failed: {e}")
        return False


def main():
    print("=" * 50)
    print("CUDA Direct Test")
    print("=" * 50)

    all_passed = True

    # nvidia-smi
    if not run_cmd("nvidia-smi", "GPU Information (nvidia-smi)"):
        all_passed = False

    # CUDA version
    run_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader", "Driver Version")

    # nvcc version (if available)
    run_cmd("nvcc --version 2>/dev/null || echo 'nvcc not in PATH'", "NVCC Compiler")

    # GPU memory
    run_cmd(
        "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv",
        "GPU Memory Status",
    )

    # GPU utilization
    run_cmd(
        "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu --format=csv",
        "GPU Utilization & Temperature",
    )

    # Try a simple CUDA operation with Python
    print("\nPython CUDA Check")
    print("-" * 50)
    try:
        import torch

        if torch.cuda.is_available():
            # Simple CUDA operation
            x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            y = x * 2
            print(f"CUDA tensor operation: {x.tolist()} * 2 = {y.tolist()}")
            print("PyTorch CUDA: OK")
        else:
            print("PyTorch CUDA: Not available")
            all_passed = False
    except ImportError:
        print("PyTorch not installed")
    except Exception as e:
        print(f"PyTorch CUDA error: {e}")
        all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("SUCCESS: CUDA test passed!")
    else:
        print("WARNING: Some tests failed")
    print("=" * 50)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
