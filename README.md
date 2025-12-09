# SADHI

A loss function designed for Image Quality Assessment (IQA) problems, balancing image fidelity, naturalness, and semantic context to optimize perceived image quality.

## Structure

- `configs/`: Configuration files (YAML).
- `data/`: Datasets.
- `notebooks/`: Jupyter notebooks for analysis.
- `src/`: Source code.
    - `losses/`: Custom loss implementations.
    - `models/`: Model architectures.
    - `utils/`: Utility functions.
- `tests/`: Unit tests.

## Setup

```bash
pip install uv
uv sync
```

## Usage

```bash
python src/train.py
```
