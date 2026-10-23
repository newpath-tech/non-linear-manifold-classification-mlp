import torch
import logging

logger = logging.getLogger(__name__)


def get_compute_device(preferred: str = "cuda", gpu_id: int = 0) -> torch.device:
    """Acquire and configure compute device prioritizing GPU acceleration.

    Strictly adheres to performance directives: offloads model parameters,
    activations, and heavy mathematical tensor evaluations directly onto CUDA.

    Args:
        preferred: 'cuda' or 'cpu'.
        gpu_id: GPU device index.

    Returns:
        torch.device configured for execution.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def log_device_telemetry(device: torch.device) -> None:
    """Print and log device characteristics and memory allocations."""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        total_memory_gb = props.total_memory / (1024 ** 3)
        allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
        print("=" * 60)
        print(f"[HARDWARE ACCELERATION ACTIVATED]")
        print(f"Device Name        : {props.name}")
        print(f"Compute Capability : {props.major}.{props.minor}")
        print(f"Total VRAM         : {total_memory_gb:.2f} GB")
        print(f"Allocated VRAM     : {allocated_mb:.2f} MB")
        print(f"Reserved VRAM      : {reserved_mb:.2f} MB")
        print("=" * 60)
    else:
        print("[WARNING] Executing on CPU. GPU is not detected or inactive.")
