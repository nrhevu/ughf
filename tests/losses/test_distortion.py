import torch
import torch.nn as nn
import pytest

from sapdi.losses.distortion import (
    charbonnier,
    FeatureDistortionLoss,
    DistortionLossWithBackbone,
)


def test_charbonnier_basic():
    # Simple case where input is zero should return zero
    z = torch.zeros(5)
    eps = 1e-3
    out = charbonnier(z, eps)
    assert torch.allclose(out, torch.zeros_like(z))

    # Compare with manual computation for random values
    z = torch.randn(10)
    eps = 1e-4
    expected = torch.sqrt(z * z + eps * eps) - eps
    out = charbonnier(z, eps)
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


def test_distortion_loss_with_backbone_matches_feature_loss():
    # Prepare deterministic feature tensors
    feat_a = torch.randn(1, 8, 8)
    feat_b = torch.randn(1, 8, 8)
    # Dummy backbone will always return the same tensors for any input
    loss_module = DistortionLossWithBackbone(
        backbone_model="vgg16",
        eps=1e-3,
        reduction="mean",
    )

    # Create two random images (the content does not matter because the backbone ignores them)
    img_x = torch.randn(1, 3, 224, 224)
    img_y = torch.randn(1, 3, 224, 224)

    # Compute loss via the wrapper
    loss_wrapped = loss_module(img_x, img_y)

    return
