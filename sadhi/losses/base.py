import torch.nn as nn
from abc import ABC, abstractmethod

class BaseLoss(nn.Module, ABC):
    """Abstract base class for custom loss functions."""
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, input, target):
        """
        Compute the loss.
        
        Args:
            input: The input tensor (predictions).
            target: The target tensor (ground truth).
            
        Returns:
            The computed loss value.
        """
        pass
