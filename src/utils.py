# Helper function for variance calculation ignoring NaNs
import torch


def simple_nanvar(tensor, dim=1, unbiased=True) -> torch.Tensor:
    """
    Calculates variance along a dimension, ignoring NaNs.
    Similar to MATLAB's var(..., dim) behavior with NaNs.

    Args:
        tensor (torch.Tensor): Input tensor.
        dim (int): Dimension along which to calculate variance.
        unbiased (bool): Whether to use the unbiased estimator (N-1 denominator).

    Returns:
        torch.Tensor: Variance along the specified dimension.
    """
    mask_valid = ~torch.isnan(tensor)
    N_valid = torch.sum(mask_valid, dim=dim, keepdim=True)

    # Calculate mean ignoring NaNs
    mean_val = torch.sum(
        torch.where(mask_valid, tensor, 0.0), dim=dim, keepdim=True
    ) / torch.clamp(N_valid, min=1)

    # Calculate sum of squared deviations ignoring NaNs
    sum_sq_dev = torch.sum(
        torch.where(mask_valid, (tensor - mean_val).pow(2), 0.0), dim=dim, keepdim=True
    )

    # Calculate variance denominator
    if unbiased:
        denominator = torch.clamp(N_valid - 1, min=1)
    else:
        denominator = torch.clamp(N_valid, min=1)

    variance = sum_sq_dev / denominator

    # Handle cases with 0 or 1 valid points
    variance[N_valid == 0] = float(
        "nan"
    )  # Or 0? Let's stick to NaN for undefined variance
    if unbiased:
        variance[N_valid <= 1] = 0.0  # Variance of a single point is 0

    # Adjust shape back by removing the kept dimension
    if dim is not None:
        variance = variance.squeeze(dim)
    elif (
        variance.numel() == 1
    ):  # Handle case where dim=None and input was scalar/vector
        variance = variance.squeeze()

    return variance
