from __future__ import annotations

DEFAULT_GPU_NAME = "RTX_4090"
DEFAULT_NUM_GPUS = 1
DEFAULT_MAX_PRICE = 0.50
DEFAULT_MIN_RELIABILITY = 0.95
DEFAULT_DOCKER_IMAGE = "davse/flask-yolo-service:latest"
DEFAULT_DISK_SPACE = 20.0
MIN_DISK_SPACE = 10.0

# Common Docker images for ML/AI workloads
# vastai/* images are pre-cached and start faster
DOCKER_IMAGES: dict[str, str] = {
    "pytorch": "vastai/pytorch",
    "tensorflow": "vastai/tensorflow",
    "cuda": "nvidia/cuda:12.1.0-runtime-ubuntu22.04",
    "cuda_devel": "nvidia/cuda:12.1.0-devel-ubuntu22.04",
    # Alternative images (not cached, slower to start)
    "pytorch_official": "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    "huggingface": "huggingface/transformers-pytorch-gpu:latest",
}
