from typing import Literal

import torch
import torch.nn as nn
import torchvision
from sapdi.clip import clip
from torchvision.models.feature_extraction import create_feature_extractor

from .base import LDistortion


def charbonnier(z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    # rho(z) = sqrt(z^2 + eps^2) - eps
    return torch.sqrt(z * z + (eps * eps)) - eps


class FeatureDistortionLoss(nn.Module):
    def __init__(self, weights, eps: float = 1e-3, reduction: str = "mean"):
        """
        weights: list/tuple of w_l (len = L), w_l > 0
        eps: Charbonnier epsilon
        reduction: "mean" or "sum" over all elements of each layer's tensor
        """
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.eps = float(eps)
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, feats_x, feats_y):
        """
        feats_x, feats_y: list/tuple of feature maps, same length L
        """
        if len(feats_x) != len(feats_y):
            raise ValueError("feats_x and feats_y must have same length")
        if len(feats_x) != int(self.weights.numel()):
            raise ValueError("weights length must match number of feature layers")

        loss = feats_x[0].new_zeros(())
        for w, fx, fy in zip(self.weights, feats_x, feats_y):
            d = fx - fy
            p = charbonnier(d, self.eps)
            layer_val = p.mean() if self.reduction == "mean" else p.sum()
            loss = loss + w * layer_val
        return loss


class DistortionLossWithBackbone(LDistortion):
    def __init__(
        self,
        backbone_model: Literal["vgg16", "resnet101", "clip_vit_b32"],
        eps=1e-3,
        reduction="mean",
    ):
        """
        backbone: e.g. torchvision.models.vgg16(weights=...), resnet50(...)
        return_nodes: mapping {<node_name_in_backbone>: <user_key>} for feature extraction
        weights: list of w_l corresponding to the order of return_nodes.values()
        """
        super().__init__()

        # select backbone
        if backbone_model == "vgg16":
            backbone = torchvision.models.vgg16(weights="DEFAULT")
            return_nodes = {"classifier.3": "f1", "classifier.6": "f2"}
            weights = [1.0, 1.0]
        elif backbone_model == "resnet101":
            backbone = torchvision.models.resnet101(weights="DEFAULT")
            return_nodes = {"avgpool": "f1"}
            weights = [1.0]
        elif backbone_model == "clip_vit_b32":
            backbone = clip.load("ViT-B/32")[0]
            return_nodes = {"visual.l2": "f1"}
            weights = [1.0]
        else:
            raise ValueError("Invalid backbone model")

        self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.loss_fn = FeatureDistortionLoss(
            weights=weights, eps=eps, reduction=reduction
        )
        self._keys = list(return_nodes.values())

    @torch.no_grad()
    def _feats(self, img: torch.Tensor):
        out = self.backbone(img)
        return [out[k] for k in self._keys]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        feats_x = self._feats(x)
        feats_y = self._feats(y)
        return self.loss_fn(feats_x, feats_y)
