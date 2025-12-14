#!/usr/bin/env python3
"""Test script for SADHI model.

This script walks through the validation dataset (default path
`SADHI-metrics/data/DiffIQA/Validation`), loads each label JSON file, builds
paths to the distorted and reference images, runs the SADHI model on the image
pair, and writes the results to a CSV file.

Usage:
    python test_sadhi.py \
        --data_dir /path/to/SADHI-metrics/data/DiffIQA/Validation \
        --afine_path /path/to/afine.pth \
        --clip_path /path/to/ViT-B-32.pt \
        --output sadhi_results.csv
"""

import argparse
import json
import csv
import subprocess
import torch
from pathlib import Path
from PIL import Image

# Import the SADHI model implementation
from sadhi.metrics.sadhi import SADHI

# CLIP utilities (the repository provides a wrapper under sadhi.CLIP)
from sadhi.CLIP import clip


def find_label_files(labels_dir: Path):
    """Return a sorted list of all JSON label files under ``labels_dir``."""
    return sorted(labels_dir.rglob("*.json"))


def load_label(json_path: Path):
    """Load a label JSON file and return the parsed dict."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_image_paths(base_dir: Path, label_data: dict, data_root: Path):
    """Construct absolute paths for distorted and reference images.

    The validation dataset layout is:
        .../data/DiffIQA/Validation/
            images/<distortion_folder>/<distorted>.png
            images/Original/<reference>.png
            labels/<type>/<distortion_folder>/<label>.json
    """
    dis_name = label_data.get("Picture_AI", {}).get("Name")
    ref_name = label_data.get("picture_Original", {}).get("Name")
    if not dis_name or not ref_name:
        raise ValueError("Missing image names in label file")

    distortion_folder = base_dir.name
    images_root = data_root / "images"
    dis_path = images_root / distortion_folder / dis_name
    ref_path = images_root / "Original" / ref_name
    return dis_path, ref_path


def load_image(img_path: Path, preprocess):
    """Load an image file and apply CLIP preprocessing, returning a torch tensor."""
    img = Image.open(img_path).convert("RGB")
    return preprocess(img).unsqueeze(0)  # add batch dimension


def main():
    parser = argparse.ArgumentParser(description="Test SADHI on validation set")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory of the validation data (e.g., SADHI-metrics/data/DiffIQA/Validation)",
    )
    parser.add_argument(
        "--afine_path",
        type=str,
        required=True,
        help="Path to afine.pth model checkpoint (used for A‑FINE heads)",
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        required=True,
        help="Path to pretrained CLIP ViT‑B‑32.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sadhi_results.csv",
        help="CSV file to store results",
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

    data_root = Path(args.data_dir).resolve()
    labels_dir = data_root / "labels"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found at {labels_dir}")

    # Load CLIP model and preprocessing function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load(args.clip_path, device="cpu", jit=False)
    clip_model = clip_model.to(device)
    clip_model.eval()

    # Instantiate SADHI model
    sadhi = SADHI(
        w1=args.w1,
        w2=args.w2,
        afine_path=args.afine_path,
        clip_path=args.clip_path,
    )
    sadhi.to(device)
    sadhi.eval()

    label_files = find_label_files(labels_dir)
    if not label_files:
        print("No label files found.")
        return

    with open(args.output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "label_file",
                "distorted_image",
                "reference_image",
                "label",
                "distortion_score",
                "naturalness_score",
            ]
        )
        for label_path in label_files:
            try:
                label_data = load_label(label_path)
                dis_path, ref_path = build_image_paths(
                    label_path.parent, label_data, data_root
                )
                # Load and preprocess images
                x = load_image(dis_path, preprocess).to(device)
                y = load_image(ref_path, preprocess).to(device)
                # Forward pass
                distortion_xy, naturalness_y = sadhi(x, y)
                # Convert tensors to scalars
                distortion_score = distortion_xy.mean().item()
                naturalness_score = naturalness_y.mean().item()
                writer.writerow(
                    [
                        str(label_path),
                        str(dis_path),
                        str(ref_path),
                        label_data.get("Picture_AI", {}).get("Label", ""),
                        f"{distortion_score:.4f}",
                        f"{naturalness_score:.4f}",
                    ]
                )
                print(
                    f"Processed {label_path.name}: distortion={distortion_score:.4f}, naturalness={naturalness_score:.4f}"
                )
            except Exception as e:
                print(f"Error processing {label_path}: {e}")


if __name__ == "__main__":
    main()
