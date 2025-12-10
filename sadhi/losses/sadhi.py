import torch
import torch.nn as nn

from .base import ASCS, LDistortion, LNaturalness, LSemantics


class SADHI(nn.Module):
    def __init__(
        self,
        w1,
        w2,
        ascs: ASCS,
        distortion: LDistortion,
        naturalness: LNaturalness,
        semantics: LSemantics,
    ):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.ascs = ascs
        self.distortion = distortion
        self.naturalness = naturalness
        self.semantics = semantics

    def forward(self, x, y):
        return (1 - self.ascs(x)) * self.distortion(x, y) + self.ascs(x) * (
            self.w1 * self.naturalness(y) + self.w2 * self.semantics(y)
        )
