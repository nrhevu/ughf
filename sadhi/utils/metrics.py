import torch

def mse(output, target):
    """Mean Squared Error."""
    return torch.mean((output - target) ** 2)

def mae(output, target):
    """Mean Absolute Error."""
    return torch.mean(torch.abs(output - target))
