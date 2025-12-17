import argparse
import os
import random
from typing import Any, Dict, Optional

import kornia
import lightning.pytorch as pl
import losses
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import utils
from datasets import get_training_data, get_validation_data
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from net_u import HGDF as myNet
from scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader


def _maybe_int(value: Optional[str]) -> Optional[Any]:
    """Convert strings representing integers into ints, keep everything else unchanged."""
    if value is None:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


class HFDTModule(pl.LightningModule):
    """LightningModule encapsulating the restoration network, losses, and schedulers."""

    def __init__(
        self, start_lr: float, end_lr: float, num_epochs: int, warmup_epochs: int
    ) -> None:
        super().__init__()
        # hparams
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        
        # init model
        self.model = myNet()

        # init loss
        self.criterion_char = losses.CharbonnierLoss()
        self.criterion_edge = losses.EdgeLoss()
        self.criterion_fft = losses.fftLoss()
        self.mse = nn.MSELoss()
        
    def forward(self, x: torch.Tensor) -> Any:
        return self.model(x)

    def _compute_loss(
        self, restored: Any, target_pyr: Any, mask_pyr: Any
    ) -> Dict[str, torch.Tensor]:
        loss_fft = sum(self.criterion_fft(restored[i], target_pyr[i]) for i in range(3))
        loss_char = sum(
            self.criterion_char(restored[i], target_pyr[i]) for i in range(3)
        )
        loss_edge = sum(
            self.criterion_edge(restored[i], target_pyr[i]) for i in range(3)
        )
        loss_mse = sum(self.mse(restored[5 - i], mask_pyr[i]) for i in range(3))
        
        total = loss_char + 0.01 * loss_fft + 0.05 * loss_edge + 0.001 * loss_mse
        return {
            "total": total,
            "char": loss_char,
            "edge": loss_edge,
            "fft": loss_fft,
            "mse": loss_mse,
        }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        target = batch[0]
        inputs = batch[1]

        mask_pyr = kornia.geometry.transform.build_pyramid(torch.abs(inputs - target), 3)
        target_pyr = kornia.geometry.transform.build_pyramid(target, 3)
        restored = self(inputs)

        losses_dict = self._compute_loss(restored, target_pyr, mask_pyr)
        self.log(
            "train/loss", losses_dict["total"], on_step=True, prog_bar=True, logger=True
        )
        self.log(
            "train/char_loss",
            losses_dict["char"],
            on_step=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/edge_loss",
            losses_dict["edge"],
            on_step=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/fft_loss",
            losses_dict["fft"],
            on_step=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/mse_loss",
            losses_dict["mse"],
            on_step=True,
            prog_bar=False,
            logger=True,
        )
        return losses_dict["total"]

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        target = batch[0]
        inputs = batch[1]
        restored = self(inputs)

        psnr_values = [
            utils.torchPSNR(res, tar) for res, tar in zip(restored[0], target)
        ]
        psnr = torch.stack(psnr_values).mean()
        self.log(
            "val/psnr", psnr, prog_bar=True, logger=True, on_epoch=True, sync_dist=True
        )
        return psnr
    
    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.start_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        warmup_epochs = self.warmup_epochs
        total_epochs = self.num_epochs
        cosine_epochs = max(total_epochs - warmup_epochs, 1)
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cosine_epochs, eta_min=self.end_lr
        )
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=cosine
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


class ImageRestorationDataModule(pl.LightningDataModule):
    """LightningDataModule handling train/validation dataloaders."""

    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        patch_size: int,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", "validate", None):
            self.train_dataset = get_training_data(
                self.train_dir, {"patch_size": self.patch_size}
            )
            self.val_dataset = get_validation_data(
                self.val_dir, {"patch_size": self.patch_size}
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )


def load_pretrained_weights(
    module: HFDTModule, checkpoint_path: Optional[str]
) -> None:
    if not checkpoint_path:
        return
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Pretrained weights not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("state_dict", state)
    module.model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights from {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the restoration network with PyTorch Lightning."
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Directory of training images.",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        required=True,
        help="Directory of validation images.",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints/logs.",
    )
    parser.add_argument(
        "--pretrain_weights",
        type=str,
        default="",
        help="Path to pretrained weights (optional).",
    )
    parser.add_argument(
        "--patch_size", type=int, default=256, help="Training patch size."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2000, help="Maximum number of epochs."
    )
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size.")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of dataloader workers."
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=5,
        help="Validate every n epochs.",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=3, help="Warmup epochs for the scheduler."
    )
    parser.add_argument(
        "--start_lr", type=float, default=2e-4, help="Initial learning rate."
    )
    parser.add_argument(
        "--end_lr",
        type=float,
        default=1e-6,
        help="Final learning rate for cosine scheduler.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="default", help="Experiment name for logging."
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--accelerator", type=str, default="auto", help="Trainer accelerator setting."
    )
    parser.add_argument(
        "--devices", type=str, default="auto", help="Trainer devices setting."
    )
    parser.add_argument(
        "--strategy", type=str, default="auto", help="Trainer strategy setting."
    )
    parser.add_argument(
        "--precision", type=str, default="32", help="Trainer precision setting."
    )
    parser.add_argument(
        "--gradient_clip_val", type=float, default=0.0, help="Gradient clipping value."
    )
    parser.add_argument(
        "--log_every_n_steps", type=int, default=50, help="Logging frequency."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pl.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_dir = os.path.join(args.model_save_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    module = HFDTModule(
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
    )

    load_pretrained_weights(module, args.pretrain_weights)

    data_module = ImageRestorationDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir,
            filename="model-{epoch:04d}-{val_psnr:.4f}",
            monitor="val/psnr",
            save_top_k=1,
            mode="max",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = TensorBoardLogger(save_dir=run_dir, name="tensorboard")

    devices = _maybe_int(args.devices)
    precision = _maybe_int(args.precision)
    
    # count number of GPUs
    if args.devices == 'auto':
        num_devices = torch.cuda.device_count()
        if num_devices > 1 and args.strategy == 'auto': 
            args.strategy = 'ddp_find_unused_parameters_true'
                

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator=args.accelerator,
        devices=devices,
        strategy=args.strategy,
        precision=precision,
        default_root_dir=run_dir,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=args.log_every_n_steps,
    )

    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    main()
