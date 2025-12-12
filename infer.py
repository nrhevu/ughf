#!/usr/bin/env python3
"""Run SADHI inference on a single distorted‑reference image pair.

The script mirrors the behaviour of ``AFINE/QuickInference/afine.py`` but
uses the SADHI model defined in ``sadhi/metrics/sadhi.py``.

Usage::

    python infer.py \
        --afine_path <path/to/afine.pth> \
        --clip_path <path/to/ViT-B-32.pt> \
        --dis_img_path <path/to/distorted.png> \
        --ref_img_path <path/to/reference.png>

The script prints the distortion (fidelity) and naturalness scores.
"""

import argparse
from pathlib import Path

import torch
import torchvision
from PIL import Image
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
    parser.add_argument(
        "--w1",
        type=float,
        default=0.5,
        help="Weight for naturalness term (default 0.5)",
    )
    parser.add_argument(
        "--w2", type=float, default=0.5, help="Weight for semantics term (default 0.5)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate SADHI model (the class internally loads the A‑FINE heads)
    sadhi = SADHI(
        w1=args.w1,
        w2=args.w2,
        afine_path=args.afine_path,
        clip_path=args.clip_path,
        ascs_module=ASCS(),
        semantics_module=Semantics(),
    )
    sadhi.to(device)
    sadhi.eval()

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

    # Forward pass – returns (distortion, naturalness)
    with torch.no_grad():
        distortion_score, naturalness_score = sadhi(dis_tensor, ref_tensor)

    # Convert to scalars for printing
    print(distortion_score)
    print(naturalness_score)

    distortion = distortion_score.mean().item()
    naturalness = naturalness_score.mean().item()

    print(f"Distortion (fidelity) score : {distortion:.4f}")
    print(f"Naturalness score          : {naturalness:.4f}")


if __name__ == "__main__":
    main()
