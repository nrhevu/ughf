import torch
import pytest

from sapdi.losses.naturalness import NaturalnessLoss
from sapdi.metrics.afine import AFINEQhead
from sapdi.clip import clip


def test_naturalness_loss_returns_tensor():
    # Create dummy components
    clip_model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
    # AFINEQhead expects certain dimensions; we can instantiate with default args
    afine_qhead = AFINEQhead()

    # Instantiate the loss
    loss_fn = NaturalnessLoss(clip_model=clip_model, afine_qhead=afine_qhead)

    # Random input image tensor (batch, channels, H, W)
    y = torch.randn(2, 3, 224, 224)

    # Compute loss (actually a score tensor)
    result = loss_fn(y)

    # Verify the output is a torch.Tensor and has expected shape (batch, 1)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 1)
    # Ensure the result is finite (no NaNs/Infs)
    assert torch.isfinite(result).all()
