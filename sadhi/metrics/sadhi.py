import torch
import torch.nn as nn

from .base import ASCS, Distortion, Naturalness, Semantics


class SADHI(nn.Module):
    def __init__(
        self,
        w1,
        w2,
        ascs: ASCS,
        distortion: Distortion,
        naturalness: Naturalness,
        semantics: Semantics,
    ):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.ascs = ascs
        self.distortion = distortion
        self.naturalness = naturalness
        self.semantics = semantics

    def forward(self, x, y):
        return (1 - self.ascs(x, y)) * self.distortion(x, y) + self.ascs(x, y) * (
            self.w1 * self.naturalness(y) + self.w2 * self.semantics(y)
        )
