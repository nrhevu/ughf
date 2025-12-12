import torch.nn as nn
from abc import abstractmethod


class Distortion(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, y):
        pass


class Naturalness(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, y):
        pass


class Semantics(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, y):
        pass


class ASCS(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, y):
        pass
