import torch
import torch.nn as nn
import pytest

from sadhi.losses.distortion import (
    charobnier,
    FeatureDistortionLoss,
    DistortionLossWithBackbone,
)


def test_charbonnier_basic():
    # Simple case where input is zero should return zero
    z = torch.zeros(5)
    eps = 1e-3
    out = charobnier(z, eps)
    assert torch.allclose(out, torch.zeros_like(z))

    # Compare with manual computation for random values
    z = torch.randn(10)
    eps = 1e-4
    expected = torch.sqrt(z * z + eps * eps) - eps
    out = charobnier(z, eps)
    assert torch.allclose(out, expected, atol=1e-7)


def test_feature_distortion_loss_mean_and_sum():
    # Create two feature maps of same shape
    fx = torch.randn(2, 3, 4, 4)
    fy = torch.randn(2, 3, 4, 4)
    # Use two layers with different weights
    weights = [0.6, 0.4]
    loss_fn_mean = FeatureDistortionLoss(weights=weights, eps=1e-3, reduction="mean")
    loss_fn_sum = FeatureDistortionLoss(weights=weights, eps=1e-3, reduction="sum")

    # Manually compute expected loss
    def manual(feats_x, feats_y, reduction):
        total = 0.0
        for w, fx_l, fy_l in zip(weights, feats_x, feats_y):
            d = fx_l - fy_l
            p = torch.sqrt(d * d + (1e-3) ** 2) - 1e-3
            layer_val = p.mean() if reduction == "mean" else p.sum()
            total += w * layer_val
        return total

    expected_mean = manual([fx, fy], [fx, fy], "mean")
    expected_sum = manual([fx, fy], [fx, fy], "sum")

    loss_mean = loss_fn_mean([fx, fy], [fx, fy])
    loss_sum = loss_fn_sum([fx, fy], [fx, fy])

    assert torch.allclose(loss_mean, expected_mean)
    assert torch.allclose(loss_sum, expected_sum)


class DummyBackbone(nn.Module):
    """A minimal backbone that returns two feature maps.

    The forward method returns a dictionary keyed by the names supplied in
    ``return_nodes``.  This mimics the behaviour of ``torchvision``'s
    ``create_feature_extractor`` without pulling in heavy model weights.
    """

    def __init__(self, out1, out2):
        super().__init__()
        self.out1 = out1
        self.out2 = out2

    def forward(self, x):
        # Ignore the input and return the pre‑computed tensors.
        return {"layer1": self.out1, "layer2": self.out2}


def test_distortion_loss_with_backbone_matches_feature_loss():
    # Prepare deterministic feature tensors
    feat_a = torch.randn(1, 8, 8)
    feat_b = torch.randn(1, 8, 8)
    # Dummy backbone will always return the same tensors for any input
    backbone = DummyBackbone(out1=feat_a, out2=feat_b)
    return_nodes = {"layer1": "f1", "layer2": "f2"}
    weights = [0.7, 0.3]
    loss_module = DistortionLossWithBackbone(
        backbone=backbone,
        return_nodes=return_nodes,
        weights=weights,
        eps=1e-3,
        reduction="mean",
    )

    # Create two random images (the content does not matter because the backbone ignores them)
    img_x = torch.randn(1, 3, 224, 224)
    img_y = torch.randn(1, 3, 224, 224)

    # Compute loss via the wrapper
    loss_wrapped = loss_module(img_x, img_y)

    # Compute loss directly using FeatureDistortionLoss for verification
    loss_direct = FeatureDistortionLoss(weights=weights, eps=1e-3, reduction="mean")
    loss_expected = loss_direct([feat_a, feat_b], [feat_a, feat_b])

    assert torch.allclose(loss_wrapped, loss_expected)
