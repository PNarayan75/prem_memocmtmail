import logging
import os
import sys

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)

import argparse
import datetime
import random

import numpy as np
import torch
from torch import optim

import trainer as Trainer
from configs.base import Config
from data.dataloader import build_train_test_dataset
from models import losses, networks, optims
from utils.configs import get_options
from utils.torch.callbacks import CheckpointsCallback


# ------------------------
# Reproducibility & Device
# ------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------
# CSV Logger (callable callback)
# ------------------------
class CSVLogger:
    """
    Callable callback that records per-epoch metrics into logs/history.csv
    and human-readable lines into logs/train.log. It is lenient about the
    signature the trainer uses to invoke callbacks.
    """
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, "history.csv")
        self.buf = {"train_loss": None, "train_acc": None, "val_loss": None, "val_acc": None}

    def _write_row(self, epoch: int, row: dict):
        import csv, math
        os.makedirs(self.log_dir, exist_ok=True)
        write_header = (not os.path.exists(self.csv_path)) or os.path.getsize(self.csv_path) == 0
        keys = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            if write_header:
                w.writeheader()
            w.writerow(row)

        # Plain text lines compatible with typical regex parsers
        tl, ta, vl, va = row["train_loss"], row["train_acc"], row["val_loss"], row["val_acc"]
        if tl is not None:
            logging.info(f"Epoch {epoch} - loss: {float(tl):.4f}")
        if ta is not None:
            logging.info(f"Epoch {epoch} - acc: {float(ta):.4f}")
        if (vl is not None) or (va is not None):
            logging.info(
                f"Validation: loss: {float(vl) if vl is not None else float('nan'):.4f} "
                f"acc: {float(va) if va is not None else float('nan'):.4f}"
            )

    def __call__(self, *args, **kwargs):
        # Accept multiple calling styles from the trainer
        epoch = kwargs.get("epoch")
        logs = kwargs.get("logs") or kwargs.get("stat") or {}

        # Positional pattern: (epoch, logs_dict)
        if epoch is None and len(args) >= 2 and isinstance(args[0], int) and isinstance(args[1], dict):
            epoch, logs = args[0], args[1]

        # Phase-style updates
        phase = kwargs.get("phase") or logs.get("phase")
        if phase in ("train", "training"):
            if "loss" in kwargs:
                logs["train_loss"] = kwargs["loss"]
            if "acc" in kwargs:
                logs["train_acc"] = kwargs["acc"]
        elif phase in ("val", "valid", "validation"):
            if "loss" in kwargs:
                logs["val_loss"] = kwargs["loss"]
            if "acc" in kwargs:
                logs["val_acc"] = kwargs["acc"]

        # Flatten any explicitly named metrics
        for k in ("train_loss", "train_acc", "val_loss", "val_acc"):
            if k in kwargs and kwargs[k] is not None:
                logs[k] = kwargs[k]

        # Update buffer
        for k in ("train_loss", "train_acc", "val_loss", "val_acc"):
            if k in logs and logs[k] is not None:
                try:
                    self.buf[k] = float(logs[k])
                except Exception:
                    pass

        # When epoch is provided, write a row
        if epoch is not None:
            row = {
                "epoch": int(epoch),
                "train_loss": self.buf["train_loss"],
                "train_acc": self.buf["train_acc"],
                "val_loss": self.buf["val_loss"],
                "val_acc": self.buf["val_acc"],
            }
            self._write_row(int(epoch), row)


# ------------------------
# Main
# ------------------------
def main(cfg: Config):
    logging.info("Initializing model...")
    # Model
    try:
        network = getattr(networks, cfg.model_type)(cfg)
        network.to(device)
    except AttributeError:
        raise NotImplementedError(f"Model {cfg.model_type} is not implemented")

    logging.info("Initializing checkpoint directory and dataset...")
    # Prepare the checkpoint directory
    cfg.checkpoint_dir = checkpoint_dir = os.path.join(
        os.path.abspath(cfg.checkpoint_dir),
        cfg.name,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    log_dir = os.path.join(checkpoint_dir, "logs")
    weight_dir = os.path.join(checkpoint_dir, "weights")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    # File logger to logs/train.log
    log_file = os.path.join(log_dir, "train.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    logging.getLogger().addHandler(fh)

    # Save cfg snapshot
    cfg.save(cfg)

    # Loss
    try:
        criterion = getattr(losses, cfg.loss_type)(cfg)
        criterion.to(device)
    except AttributeError:
        raise NotImplementedError(f"Loss {cfg.loss_type} is not implemented")

    # Trainer
    try:
        trainer = getattr(Trainer, cfg.trainer)(
            cfg=cfg,
            network=network,
            criterion=criterion,
            log_dir=cfg.checkpoint_dir,
        )
    except AttributeError:
        raise NotImplementedError(f"Trainer {cfg.trainer} is not implemented")

    # Data
    train_ds, test_ds = build_train_test_dataset(cfg)
    logging.info("Initializing trainer...")

    logging.info("Start training...")

    # Optimizer & Scheduler
    optimizer = optims.get_optim(cfg, network)
    lr_scheduler = None
    if cfg.learning_rate_step_size is not None:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.learning_rate_step_size,
            gamma=cfg.learning_rate_gamma,
        )

    # Callbacks
    ckpt_callback = CheckpointsCallback(
        checkpoint_dir=weight_dir,
        save_freq=cfg.save_freq,
        max_to_keep=cfg.max_to_keep,
        save_best_val=cfg.save_best_val,
        save_all_states=cfg.save_all_states,
    )
    csv_logger = CSVLogger(log_dir)

    # Resume
    if cfg.resume:
        trainer.load_all_states(cfg.resume_path)

    # Compile & Fit
    trainer.compile(optimizer=optimizer, scheduler=lr_scheduler)
    trainer.fit(train_ds, cfg.num_epochs, test_ds, callbacks=[ckpt_callback, csv_logger])


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, default="../src/configs/base.py")
    return parser.parse_args()


if __name__ == "__main__":
    # Root logger basic config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = arg_parser()
    cfg: Config = get_options(args.config)
    if cfg.resume and cfg.cfg_path is not None:
        resume = cfg.resume
        resume_path = cfg.resume_path
        cfg.load(cfg.cfg_path)
        cfg.resume = resume
        cfg.resume_path = resume_path

    main(cfg)
