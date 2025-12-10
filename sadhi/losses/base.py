import torch.nn as nn
from abc import abstractmethod


class LDistortion(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, y):
        pass


class LNaturalness(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, y):
        pass


class LSemantics(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, y):
        pass


class ASCS(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass
