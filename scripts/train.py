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

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CSVLogger:
    """Per-epoch metrics ko logs/history.csv + train.log me likhne ke liye."""
    def __init__(self, log_dir):
        self.csv_path = os.path.join(log_dir, "history.csv")

    def on_epoch_end(self, epoch, logs: dict):
        # logs: {"train_loss":..., "train_acc":..., "val_loss":..., "val_acc":...}
        import csv, math
        keys = ["epoch","train_loss","train_acc","val_loss","val_acc"]
        row = {
            "epoch": epoch + 1,
            "train_loss": float(logs.get("train_loss", "nan")),
            "train_acc":  float(logs.get("train_acc", "nan")),
            "val_loss":   float(logs.get("val_loss", "nan")),
            "val_acc":    float(logs.get("val_acc", "nan")),
        }
        write_header = (not os.path.exists(self.csv_path)) or os.path.getsize(self.csv_path)==0
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            if write_header: w.writeheader()
            w.writerow(row)

        # plain-text lines (eval.py regex ke compatible)
        if not any(map(math.isnan, [row["train_loss"], row["train_acc"]])):
            logging.info(f"Epoch {row['epoch']} - loss: {row['train_loss']:.4f}")
            logging.info(f"Epoch {row['epoch']} - acc: {row['train_acc']:.4f}")
        if not any(map(math.isnan, [row["val_loss"], row["val_acc"]])):
            logging.info(f"Validation: loss: {row['val_loss']:.4f} acc: {row['val_acc']:.4f}")

def main(cfg: Config):
    logging.info("Initializing model...")
    # Model
    try:
        network = getattr(networks, cfg.model_type)(cfg)
        network.to(device)
    except AttributeError:
        raise NotImplementedError("Model {} is not implemented".format(cfg.model_type))

    logging.info("Initializing checkpoint directory and dataset...")
    # Preapre the checkpoint directory
    cfg.checkpoint_dir = checkpoint_dir = os.path.join(
        os.path.abspath(cfg.checkpoint_dir),
        cfg.name,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    log_dir = os.path.join(checkpoint_dir, "logs")
    weight_dir = os.path.join(checkpoint_dir, "weights")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    logging.getLogger().addHandler(fh)
    cfg.save(cfg)

    try:
        criterion = getattr(losses, cfg.loss_type)(cfg)
        criterion.to(device)
    except AttributeError:
        raise NotImplementedError("Loss {} is not implemented".format(cfg.loss_type))

    try:
        trainer = getattr(Trainer, cfg.trainer)(
            cfg=cfg,
            network=network,
            criterion=criterion,
            log_dir=cfg.checkpoint_dir,
        )
    except AttributeError:
        raise NotImplementedError("Trainer {} is not implemented".format(cfg.trainer))

    train_ds, test_ds = build_train_test_dataset(cfg)
    logging.info("Initializing trainer...")

    logging.info("Start training...")

    optimizer = optims.get_optim(cfg, network)
    lr_scheduler = None
    if cfg.learning_rate_step_size is not None:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.learning_rate_step_size,
            gamma=cfg.learning_rate_gamma,
        )

    ckpt_callback = CheckpointsCallback(
        checkpoint_dir=weight_dir,
        save_freq=cfg.save_freq,
        max_to_keep=cfg.max_to_keep,
        save_best_val=cfg.save_best_val,
        save_all_states=cfg.save_all_states,
    )

    if cfg.resume:
        trainer.load_all_states(cfg.resume_path)

   
    csv_logger = CSVLogger(log_dir)
    trainer.compile(optimizer=optimizer, scheduler=lr_scheduler)
    trainer.fit(train_ds, cfg.num_epochs, test_ds, callbacks=[ckpt_callback, csv_logger])

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, default="../src/configs/base.py")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    cfg: Config = get_options(args.config)
    if cfg.resume and cfg.cfg_path is not None:
        resume = cfg.resume
        resume_path = cfg.resume_path
        cfg.load(cfg.cfg_path)
        cfg.resume = resume
        cfg.resume_path = resume_path

    main(cfg)
