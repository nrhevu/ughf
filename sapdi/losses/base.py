from abc import abstractmethod

import torch
import torch.nn as nn


class LDistortion(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        pass


class LNaturalness(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, y: torch.Tensor):
        pass


class LSemantics(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, y: torch.Tensor):
        pass


class LAscs(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        pass
