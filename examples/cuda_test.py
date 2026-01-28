#!/usr/bin/env python3
"""Direct CUDA test using subprocess to call nvidia-smi and nvcc."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


class Logger:
    """Simple logger that writes to both stdout and file."""

    def __init__(self, log_path: str = "/root/logs/cuda_test.log"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_path, "w")
        self.write(f"CUDA Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def write(self, msg: str):
        print(msg, end="")
        self.file.write(msg)
        self.file.flush()

    def writeln(self, msg: str = ""):
        self.write(msg + "\n")

    def close(self):
        self.file.close()


log = None


def run_cmd(cmd, description):
    """Run a command and print output."""
    log.writeln(f"\n{description}")
    log.writeln("-" * 50)
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.stdout:
            log.write(result.stdout)
        if result.returncode != 0 and result.stderr:
            log.writeln(f"Error: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log.writeln("Command timed out")
        return False
    except Exception as e:
        log.writeln(f"Failed: {e}")
        return False


def main():
    global log
    log = Logger()

    log.writeln("=" * 50)
    log.writeln("CUDA Direct Test")
    log.writeln("=" * 50)

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
    log.writeln("\nPython CUDA Check")
    log.writeln("-" * 50)
    try:
        import torch

        if torch.cuda.is_available():
            # Simple CUDA operation
            x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            y = x * 2
            log.writeln(f"CUDA tensor operation: {x.tolist()} * 2 = {y.tolist()}")
            log.writeln("PyTorch CUDA: OK")
        else:
            log.writeln("PyTorch CUDA: Not available")
    except ImportError:
        log.writeln("PyTorch not installed")
    except Exception as e:
        log.writeln(f"PyTorch CUDA error: {e}")

    log.writeln("\n" + "=" * 50)
    if all_passed:
        log.writeln("SUCCESS: CUDA test passed!")
    else:
        log.writeln("WARNING: Some tests failed")
    log.writeln("=" * 50)

    log.writeln(f"\nLog saved to: {log.log_path}")
    log.close()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
