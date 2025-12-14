import torch
from sapdi.clip.model import CLIP
from sapdi.metrics.afine import AFINEQhead

from .base import LNaturalness


class NaturalnessLoss(LNaturalness):
    def __init__(self, clip_model: CLIP, afine_qhead: AFINEQhead):
        super().__init__()
        self.clip_model = clip_model
        self.afine_qhead = afine_qhead

    @torch.no_grad()
    def _feats(self, img: torch.Tensor):
        cls_x, feat_x = self.clip_model.encode_image(img)
        return feat_x

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        naturalness_y = self.afine_qhead(y, self._feats(y))

        return naturalness_y
