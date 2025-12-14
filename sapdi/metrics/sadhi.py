import torch
import torch.nn as nn
from sadhi.clip import clip

from .afine import AFINEDhead, AFINEQhead
from .base import ASCS, Distortion, Naturalness, Semantics


class SADHI(nn.Module):
    def __init__(
        self,
        w1,
        w2,
        afine_path,
        clip_path,
        ascs_module,
        semantics_module,
    ):
        super().__init__()
        self.w1 = w1
        self.w2 = w2

        # Load CLIP
        self.clip_model, _ = clip.load(clip_path, device="cpu", jit=False)
        finetuned_clip_checkpoint = torch.load(afine_path, map_location="cpu")[
            "finetuned_clip"
        ]
        self.clip_model.load_state_dict(finetuned_clip_checkpoint)

        # Load A-FINE fidelity term
        self.net_fidelity = AFINEDhead()
        self.net_fidelity.load_state_dict(
            torch.load(afine_path, map_location="cpu")["fidelity"], strict=True
        )

        # Load A-FINE naturalness term
        self.net_naturalness = AFINEQhead()
        self.net_naturalness.load_state_dict(
            torch.load(afine_path, map_location="cpu")["natural"], strict=True
        )

        # Load ASCS module
        self.ascs_module = ascs_module
        self.semantics_module = semantics_module

    def forward(self, x, y):
        # The height and width of all the images must be divisible by 32, since we utilize the pretrained CLIP ViT-B-32 model
        _, c, h, w = x.shape
        if h % 32 != 0:
            pad_h = 32 - h % 32
        else:
            pad_h = 0

        if w % 32 != 0:
            pad_w = 32 - w % 32
        else:
            pad_w = 0

        if pad_h > 0 or pad_w > 0:
            x = F.interpolate(
                x, size=(h + pad_h, w + pad_w), mode="bicubic", align_corners=False
            )
            y = F.interpolate(
                y, size=(h + pad_h, w + pad_w), mode="bicubic", align_corners=False
            )

        with torch.no_grad():
            # CLIP encoding
            cls_x, feat_x = self.clip_model.encode_image(x)
            cls_y, feat_y = self.clip_model.encode_image(y)

            # Calculate fidelity and naturalness
            distortion_xy = self.net_fidelity(x, y, feat_x, feat_y)
            naturalness_y = self.net_naturalness(y, feat_y)

            # ASCS
            ascsc_y = self.ascs_module(x, y)

        return distortion_xy, naturalness_y

        # return (1 - self.ascs_module(x, y)) * distortion_xy + self.ascs_module(x, y) * (
        #     self.w1 * naturalness_y + self.w2 * self.semantics_module(y)
        # )
