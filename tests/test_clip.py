import argparse
from pathlib import Path

import torch
import torchvision
from PIL import Image
from sadhi.clip import clip
from sadhi.metrics.base import ASCS, Semantics
from sadhi.metrics.sadhi import SADHI


def load_image(img_path: Path, preprocess):
    """Load an image, apply CLIP preprocessing and add a batch dimension."""
    img = Image.open(img_path).convert("RGB")
    return preprocess(img).unsqueeze(0)  # shape: [1, 3, H, W]


def main():
    parser = argparse.ArgumentParser(description="SADHI single‑image inference")
    parser.add_argument(
        "--afine-path",
        type=str,
        required=True,
        help="Path to the A‑FINE checkpoint (used for the heads)",
    )
    parser.add_argument(
        "--clip-path",
        type=str,
        required=True,
        help="Path to pretrained CLIP ViT‑B‑32.pt",
    )
    parser.add_argument(
        "--dis-img-path", type=str, required=True, help="Path to the distorted image"
    )
    parser.add_argument(
        "--ref-img-path", type=str, required=True, help="Path to the reference image"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load pretrained CLIP
    clip_model, _ = clip.load(args.clip_path, device="cpu", jit=False)
    # load our finetuned CLIP
    finetuned_clip_checkpoint = torch.load(args.afine_path, map_location="cpu")[
        "finetuned_clip"
    ]
    clip_model.load_state_dict(finetuned_clip_checkpoint)
    clip_model = clip_model.to(device)

    # Prepare inputs
    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    dis_tensor = load_image(Path(args.dis_img_path), preprocess).to(device)
    ref_tensor = load_image(Path(args.ref_img_path), preprocess).to(device)

    # The height and width of all the images must be divisible by 32, since we utilize the pretrained CLIP ViT-B-32 model
    _, c, h, w = dis_tensor.shape
    if h % 32 != 0:
        pad_h = 32 - h % 32
    else:
        pad_h = 0

    if w % 32 != 0:
        pad_w = 32 - w % 32
    else:
        pad_w = 0

    if pad_h > 0 or pad_w > 0:
        dis_tensor = F.interpolate(
            dis_tensor, size=(h + pad_h, w + pad_w), mode="bicubic", align_corners=False
        )
        ref_tensor = F.interpolate(
            ref_tensor, size=(h + pad_h, w + pad_w), mode="bicubic", align_corners=False
        )

    with torch.no_grad():
        cls_dis, feat_dis = clip_model.encode_image(dis_tensor)
        cls_ref, feat_ref = clip_model.encode_image(ref_tensor)

    print(cls_dis)
    print(cls_ref)
    print(feat_dis)
    print(feat_ref)


if __name__ == "__main__":
    main()
