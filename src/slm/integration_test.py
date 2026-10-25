"""Small CUDA smoke test for validating the Colab integration workflow."""

import torch


def gpu_matrix_test(device: torch.device) -> dict[str, object]:
    """Multiply small matrices on CUDA and return JSON-serializable metrics.

    Args:
        device: CUDA device on which to execute the matrix multiplication.

    Returns:
        A dictionary containing the execution device, result shape, metrics,
        and test status.

    Raises:
        RuntimeError: If CUDA is unavailable or a non-CUDA device is given.
    """
    if device.type != "cuda":
        raise RuntimeError("gpu_matrix_test requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch runtime")

    left = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float32,
    ).to(device)
    right = torch.tensor(
        [[5.0, 6.0], [7.0, 8.0]],
        dtype=torch.float32,
    ).to(device)
    product = torch.matmul(left, right)

    torch.cuda.synchronize(device)
    is_finite = bool(torch.isfinite(product).all().item())
    return {
        "device": product.device.type,
        "shape": list(product.shape),
        "sum": float(product.sum().item()),
        "mean": float(product.mean().item()),
        "status": "PASS" if is_finite else "FAIL",
    }
