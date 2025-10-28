import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)
import re

# Path Setup for Imports
lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)

from data.dataloader import build_train_test_dataset
from tqdm.auto import tqdm
from models import networks
from configs.base import Config

LABEL_MAP = ["Angry", "Happy", "Sad", "Neutral"]

def parse_train_log(log_path):
    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []
    current_epoch = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            # Epoch X - loss: Y
            loss_match = re.search(r"Epoch (\d+) - loss: ([\d.]+)", line)
            if loss_match:
                epoch = int(loss_match.group(1))
                loss = float(loss_match.group(2))
                if current_epoch is None:
                    current_epoch = epoch
                if epoch == current_epoch:
                    train_loss.append(loss)
                continue

            # Epoch X - acc: Y
            acc_match = re.search(r"Epoch (\d+) - acc: ([\d.]+)", line)
            if acc_match:
                epoch = int(acc_match.group(1))
                acc = float(acc_match.group(2))
                if epoch == current_epoch:
                    train_acc.append(acc)
                    epochs.append(epoch)
                continue

            # Validation: loss: X acc: Y
            val_match = re.search(r"Validation: loss: ([\d.]+) acc: ([\d.]+)", line)
            if val_match:
                val_loss.append(float(val_match.group(1)))
                val_acc.append(float(val_match.group(2)))
                current_epoch = None
                continue

    return epochs, train_loss, train_acc, val_loss, val_acc

def model_eval(cfg, ckpt_path, all_state_dict=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = getattr(networks, cfg.model_type)(cfg).to(device)
    weights = torch.load(ckpt_path, map_location=device)
    if all_state_dict:
        weights = weights["state_dict_network"]
    net.load_state_dict(weights, strict=False)
    net.eval()
    y_true, y_pred_cls, y_pred_logits = [], [], []
    _, test_ds = build_train_test_dataset(cfg)
    print(f"Test dataset size: {len(test_ds)}")  # Debug
    for input_ids, audio, label in tqdm(test_ds, desc="Test"):
        input_ids, audio, label = input_ids.to(device), audio.to(device), label.to(device)
        with torch.no_grad():
            out = net(input_ids, audio)[0].cpu().numpy()[0]
            y_pred_logits.append(out)
            y_pred_cls.append(np.argmax(out))
            y_true.append(label.cpu().numpy()[0])
    print(f"Eval results: {len(y_true)} samples")  # Debug
    return y_true, y_pred_cls, np.array(y_pred_logits)

def plot_auc_roc(ax, y_true, y_logits):
    if len(set(y_true)) < 2 or y_logits.shape[0] == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return
    for cls in sorted(set(y_true)):
        y_true_bin = np.array(y_true) == cls
        fpr, tpr, _ = roc_curve(y_true_bin, y_logits[:, cls])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{LABEL_MAP[cls]} (AUC={roc_auc:.2f})", linewidth=1.5)
    ax.plot([0,1],[0,1],"r--",label="Random", linewidth=1.2)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("AUC-ROC", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=6, loc="lower right", frameon=True)
    ax.set_aspect("equal", adjustable="box")

def plot_confmat(ax, y_true, y_pred):
    if len(y_true) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return
    cm = confusion_matrix(y_true, y_pred)
    cmn = (cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]) * 100
    sns.heatmap(
        cmn, cmap="YlOrBr", annot=True, fmt=".1f", ax=ax,
        square=True, linecolor="black", linewidths=0.5,
        annot_kws={"size": 10}, cbar=False
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Ground Truth", fontsize=10)
    ax.set_xticklabels(LABEL_MAP, fontsize=8)
    ax.set_yticklabels(LABEL_MAP, fontsize=8)
    ax.set_title("Confusion Matrix", fontsize=10, fontweight="bold")

def main(args):
    # Log and config
    epochs, train_loss, train_acc, val_loss, val_acc = parse_train_log(args.train_log)
    cfg = Config(); cfg.load(args.cfg)
    cfg.data_valid = args.test_set if args.test_set else "test.pkl"
    if args.data_root: cfg.data_root, cfg.data_name = args.data_root, args.data_name

    # Model eval
    y_true, y_pred_cls, y_pred_logits = model_eval(cfg, args.checkpoint, args.all_state_dict)

    # IEEE-style: uniform subplot sizes, bold titles, clean labels, tight layout
    fig, axs = plt.subplots(1, 4, figsize=(10, 2.5))
    plt.subplots_adjust(wspace=0.35, hspace=0.1)

    # Accuracy Curve
    if len(epochs) > 0:
        axs[0].plot(epochs, train_acc, label="Train", color="tab:blue", linewidth=1.5)
        axs[0].plot(epochs[:len(val_acc)], val_acc, label="Val", color="tab:orange", linewidth=1.5)  # Align lengths
    axs[0].set_title("Model Accuracy", fontsize=10, fontweight="bold")
    axs[0].set_xlabel("Epochs", fontsize=10)
    axs[0].set_ylabel("Accuracy", fontsize=10)
    axs[0].tick_params(axis="both", labelsize=9)
    axs[0].legend(fontsize=8, loc="best", frameon=True)
    axs[0].grid(True, linestyle="--", alpha=0.5)

    # Loss Curve
    if len(epochs) > 0:
        axs[1].plot(epochs, train_loss, label="Train", color="tab:blue", linewidth=1.5)
        axs[1].plot(epochs[:len(val_loss)], val_loss, label="Val", color="tab:orange", linewidth=1.5)  # Align lengths
    axs[1].set_title("Model Loss", fontsize=10, fontweight="bold")
    axs[1].set_xlabel("Epochs", fontsize=10)
    axs[1].set_ylabel("Loss", fontsize=10)
    axs[1].tick_params(axis="both", labelsize=9)
    axs[1].legend(fontsize=8, loc="best", frameon=True)
    axs[1].grid(True, linestyle="--", alpha=0.5)

    # AUC-ROC
    plot_auc_roc(axs[2], y_true, y_pred_logits)

    # Confusion Matrix
    plot_confmat(axs[3], y_true, y_pred_cls)

    # Save
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_model_report.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(f"{args.output_prefix}_model_report.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.show()

def arg_parser():
    parser = argparse.ArgumentParser(description="IEEE-style 4-in-1 model evaluation report.")
    parser.add_argument("--train_log", type=str, required=True, help="Path to train.log")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best model checkpoint")
    parser.add_argument("--cfg", type=str, required=True, help="Path to cfg.log")
    parser.add_argument("--output_prefix", type=str, default="model", help="Prefix for output files")
    parser.add_argument("--test_set", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--data_name", type=str, default=None)
    parser.add_argument("--all_state_dict", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    main(arg_parser())
# # import logging
# # import os
# # import sys

# # lib_path = os.path.abspath("").replace("scripts", "src")
# # sys.path.append(lib_path)
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# # )
# # import csv
# # import glob
# # import argparse
# # import torch
# # import json
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import seaborn as sns
# # from sklearn import svm
# # from sklearn.metrics import (
# #     balanced_accuracy_score,
# #     accuracy_score,
# #     confusion_matrix,
# #     f1_score,
# # )
# # from data.dataloader import build_train_test_dataset
# # from tqdm.auto import tqdm
# # from models import networks
# # from configs.base import Config
# # from collections import Counter
# # from typing import Tuple


# # def calculate_accuracy(y_true, y_pred) -> Tuple[float, float]:
# #     class_weights = {cls: 1.0 / count for cls, count in Counter(y_true).items()}
# #     bacc = float(
# #         balanced_accuracy_score(
# #             y_true, y_pred, sample_weight=[class_weights[cls] for cls in y_true]
# #         )
# #     )
# #     acc = float(accuracy_score(y_true, y_pred))
# #     return bacc, acc


# # def calculate_f1_score(y_true, y_pred) -> Tuple[float, float]:
# #     macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
# #     weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
# #     return macro_f1, weighted_f1


# # def eval(cfg, checkpoint_path, all_state_dict=True, cm=False):
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     network = getattr(networks, cfg.model_type)(cfg)
# #     network.to(device)

# #     # Build dataset
# #     _, test_ds = build_train_test_dataset(cfg)
# #     weight = torch.load(checkpoint_path, map_location=torch.device(device))
# #     if all_state_dict:
# #         weight = weight["state_dict_network"]

# #     network.load_state_dict(weight)
# #     network.eval()
# #     network.to(device)

# #     y_actu = []
# #     y_pred = []

# #     for every_test_list in tqdm(test_ds):
# #         input_ids, audio, label = every_test_list
# #         input_ids = input_ids.to(device)
# #         audio = audio.to(device)
# #         label = label.to(device)
# #         with torch.no_grad():
# #             output = network(input_ids, audio)[0]
# #             _, preds = torch.max(output, 1)
# #             y_actu.append(label.detach().cpu().numpy()[0])
# #             y_pred.append(preds.detach().cpu().numpy()[0])
# #     bacc, acc = calculate_accuracy(y_actu, y_pred)
# #     macro_f1, weighted_f1 = calculate_f1_score(y_actu, y_pred)
# #     if cm:
# #         cm = confusion_matrix(y_actu, y_pred)
# #         print("Confusion Matrix: \n", cm)
# #         cmn = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]) * 100

# #         ax = plt.subplots(figsize=(8, 5.5))[1]
# #         sns.heatmap(
# #             cmn,
# #             cmap="YlOrBr",
# #             annot=True,
# #             square=True,
# #             linecolor="black",
# #             linewidths=0.75,
# #             ax=ax,
# #             fmt=".2f",
# #             annot_kws={"size": 16},
# #         )
# #         ax.set_xlabel("Predicted", fontsize=18, fontweight="bold")
# #         ax.xaxis.set_label_position("bottom")
# #         label_names = ["Anger", "Happiness", "Sadness", "Neutral"]
# #         if cfg.num_classes != 4:
# #             with open(os.path.join(cfg.data_root, "classes.json"), "r") as f:
# #                 label_data = json.load(f)
# #                 label_names = label_data.keys()

# #         ax.xaxis.set_ticklabels(label_names, fontsize=16)
# #         ax.set_ylabel("Ground Truth", fontsize=18, fontweight="bold")
# #         ax.yaxis.set_ticklabels(label_names, fontsize=16)
# #         plt.tight_layout()
# #         plt.savefig(
# #             "confusion_matrix_" + cfg.name + cfg.data_valid + ".png",
# #             format="png",
# #             dpi=1200,
# #         )

# #     return bacc, acc, macro_f1, weighted_f1


# # def find_checkpoint_folder(path):
# #     candidate = os.listdir(path)
# #     if "logs" in candidate and "weights" in candidate and "cfg.log" in candidate:
# #         return [path]
# #     list_candidates = []
# #     for c in candidate:
# #         list_candidates += find_checkpoint_folder(os.path.join(path, c))
# #     return list_candidates


# # def main(args):
# #     logging.info("Finding checkpoints")
# #     list_checkpoints = find_checkpoint_folder(args.checkpoint_path)
# #     test_set = args.test_set if args.test_set is not None else "test.pkl"
# #     csv_path = os.path.basename(args.checkpoint_path) + "{}.csv".format(test_set)
# #     # field names
# #     fields = ["BACC", "ACC", "MACRO_F1", "WEIGHTED_F1", "Time", "Model", "Settings"]
# #     with open(csv_path, "a") as csvfile:
# #         writer = csv.DictWriter(csvfile, fieldnames=fields)
# #         writer.writeheader()
# #         for ckpt in list_checkpoints:
# #             meta_info = ckpt.split("/")
# #             time = meta_info[-1]
# #             settings = meta_info[-2]
# #             model_name = meta_info[-3]
# #             logging.info("Evaluating: {}/{}/{}".format(model_name, settings, time))
# #             cfg_path = os.path.join(ckpt, "cfg.log")
# #             if args.latest:
# #                 ckpt_path = glob.glob(os.path.join(ckpt, "weights", "*.pt"))
# #                 if len(ckpt_path) != 0:
# #                     ckpt_path = ckpt_path[0]
# #                     all_state_dict = True
# #                 else:
# #                     ckpt_path = glob.glob(os.path.join(ckpt, "weights", "*.pth"))[0]
# #                     all_state_dict = False

# #             else:
# #                 ckpt_path = os.path.join(ckpt, "weights/best_acc/checkpoint_0_0.pt")
# #                 all_state_dict = True
# #                 if not os.path.exists(ckpt_path):
# #                     ckpt_path = os.path.join(ckpt, "weights/best_acc/checkpoint_0.pth")
# #                     all_state_dict = False

# #             cfg = Config()
# #             cfg.load(cfg_path)
# #             # Change to test set
# #             cfg.data_valid = test_set
# #             if args.data_root is not None:
# #                 assert (
# #                     args.data_name is not None
# #                 ), "Change validation dataset requires data_name"
# #                 cfg.data_root = args.data_root
# #                 cfg.data_name = args.data_name

# #             bacc, acc, macro_f1, weighted_f1 = eval(
# #                 cfg, ckpt_path, all_state_dict=all_state_dict, cm=args.confusion_matrix
# #             )
# #             writer.writerows(
# #                 [
# #                     {
# #                         "BACC": round(bacc * 100, 2),
# #                         "ACC": round(acc * 100, 2),
# #                         "MACRO_F1": round(macro_f1 * 100, 2),
# #                         "WEIGHTED_F1": round(weighted_f1 * 100, 2),
# #                         "Time": time,
# #                         "Model": model_name,
# #                         "Settings": settings,
# #                     }
# #                 ]
# #             )
# #             logging.info(
# #                 "\nBACC | ACC | MACRO_F1 | WEIGHTED_F1 \n{:.2f} & {:.2f} & {:.2f} & {:.2f}".format(
# #                     round(bacc * 100, 2),
# #                     round(acc * 100, 2),
# #                     round(macro_f1 * 100, 2),
# #                     round(weighted_f1 * 100, 2),
# #                 )
# #             )


# # def arg_parser():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument(
# #         "-ckpt", "--checkpoint_path", type=str, help="path to checkpoint folder"
# #     )
# #     parser.add_argument(
# #         "-r",
# #         "--recursive",
# #         action="store_true",
# #         help="whether to travel child folder or not",
# #     )

# #     parser.add_argument(
# #         "-l",
# #         "--latest",
# #         action="store_true",
# #         help="whether to use latest weight or best weight",
# #     )

# #     parser.add_argument(
# #         "-t",
# #         "--test_set",
# #         type=str,
# #         default=None,
# #         help="name of testing set. Ex: test.pkl",
# #     )

# #     parser.add_argument(
# #         "-cm",
# #         "--confusion_matrix",
# #         action="store_true",
# #         help="whether to export consution matrix or not",
# #     )

# #     parser.add_argument(
# #         "--data_root",
# #         type=str,
# #         default=None,
# #         help="If want to change the validation dataset",
# #     )
# #     parser.add_argument(
# #         "--data_name", type=str, default=None, help="for changing validation dataset"
# #     )

# #     return parser.parse_args()


# # if __name__ == "__main__":
# #     args = arg_parser()
# #     if not args.recursive:
# #         cfg_path = os.path.join(args.checkpoint_path, "cfg.log")
# #         all_state_dict = True
# #         ckpt_path = os.path.join(
# #             args.checkpoint_path, "weights/best_acc/checkpoint_0_0.pt"
# #         )
# #         if not os.path.exists(ckpt_path):
# #             ckpt_path = os.path.join(
# #                 args.checkpoint_path, "weights/best_acc/checkpoint_0.pth"
# #             )
# #             all_state_dict = False

# #         cfg = Config()
# #         cfg.load(cfg_path)
# #         # Change to test set
# #         test_set = args.test_set if args.test_set is not None else "test.pkl"
# #         cfg.data_valid = test_set
# #         if args.data_root is not None:
# #             assert (
# #                 args.data_name is not None
# #             ), "Change validation dataset requires data_name"
# #             cfg.data_root = args.data_root
# #             cfg.data_name = args.data_name

# #         bacc, acc, macro_f1, weighted_f1 = eval(
# #             cfg,
# #             ckpt_path,
# #             cm=args.confusion_matrix,
# #             all_state_dict=all_state_dict,
# #         )
# #         logging.info(
# #             "\nBACC | ACC | MACRO_F1 | WEIGHTED_F1 \n{:.2f} & {:.2f} & {:.2f} & {:.2f}".format(
# #                 round(bacc * 100, 2),
# #                 round(acc * 100, 2),
# #                 round(macro_f1 * 100, 2),
# #                 round(weighted_f1 * 100, 2),
# #             )
# #         )

# #     else:
# #         main(args)
# import os
# import sys
# import argparse
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from sklearn.metrics import (
#     confusion_matrix,
#     roc_curve,
#     auc,
# )
# import re

# # Path Setup for Imports
# lib_path = os.path.abspath("").replace("scripts", "src")
# sys.path.append(lib_path)

# from data.dataloader import build_train_test_dataset
# from tqdm.auto import tqdm
# from models import networks
# from configs.base import Config

# LABEL_MAP = ["Angry", "Happy", "Sad", "Neutral"]

# def parse_train_log(log_path):
#     epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []
#     with open(log_path, "r") as f:
#         for line in f:
#             if "Epoch" in line and "loss:" in line and "acc:" not in line:
#                 epochs.append(int(re.search(r"Epoch (\d+)", line).group(1)))
#                 train_loss.append(float(re.search(r"loss: ([\d.]+)", line).group(1)))
#             elif "Epoch" in line and "acc:" in line and "loss:" not in line:
#                 train_acc.append(float(re.search(r"acc: ([\d.]+)", line).group(1)))
#             elif "Validation: loss:" in line:
#                 val_loss.append(float(re.search(r"loss: ([\d.]+)", line).group(1)))
#                 val_acc.append(float(re.search(r"acc: ([\d.]+)", line).group(1)))
#     return epochs, train_loss, train_acc, val_loss, val_acc

# def model_eval(cfg, ckpt_path, all_state_dict=True):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = getattr(networks, cfg.model_type)(cfg).to(device)
#     weights = torch.load(ckpt_path, map_location=device)
#     if all_state_dict:
#         weights = weights["state_dict_network"]
#     net.load_state_dict(weights, strict=False)
#     net.eval()
#     y_true, y_pred_cls, y_pred_logits = [], [], []
#     _, test_ds = build_train_test_dataset(cfg)
#     for input_ids, audio, label in tqdm(test_ds, desc="Test"):
#         input_ids, audio, label = input_ids.to(device), audio.to(device), label.to(device)
#         with torch.no_grad():
#             out = net(input_ids, audio)[0].cpu().numpy()[0]
#             y_pred_logits.append(out)
#             y_pred_cls.append(np.argmax(out))
#             y_true.append(label.cpu().numpy()[0])
#     return y_true, y_pred_cls, np.array(y_pred_logits)

# def plot_auc_roc(ax, y_true, y_logits):
#     for cls in sorted(set(y_true)):
#         y_true_bin = np.array(y_true) == cls
#         fpr, tpr, _ = roc_curve(y_true_bin, y_logits[:, cls])
#         roc_auc = auc(fpr, tpr)
#         ax.plot(fpr, tpr, label=f"{LABEL_MAP[cls]} (AUC={roc_auc:.2f})", linewidth=1.5)
#     ax.plot([0,1],[0,1],"r--",label="Random", linewidth=1.2)
#     ax.set_xlabel("False Positive Rate", fontsize=10)
#     ax.set_ylabel("True Positive Rate", fontsize=10)
#     ax.set_title("AUC-ROC", fontsize=10, fontweight="bold")
#     ax.grid(True, linestyle="--", alpha=0.5)
#     ax.legend(fontsize=6, loc="lower right", frameon=True)
#     ax.set_aspect("equal", adjustable="box")

# def plot_confmat(ax, y_true, y_pred):
#     cm = confusion_matrix(y_true, y_pred)
#     cmn = (cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]) * 100
#     sns.heatmap(
#         cmn, cmap="YlOrBr", annot=True, fmt=".1f", ax=ax,
#         square=True, linecolor="black", linewidths=0.5,
#         annot_kws={"size": 10}, cbar=False
#     )
#     ax.set_xlabel("Predicted", fontsize=10)
#     ax.set_ylabel("Ground Truth", fontsize=10)
#     ax.set_xticklabels(LABEL_MAP, fontsize=8)
#     ax.set_yticklabels(LABEL_MAP, fontsize=8)
#     ax.set_title("Confusion Matrix", fontsize=10, fontweight="bold")

# def main(args):
#     # Log and config
#     epochs, train_loss, train_acc, val_loss, val_acc = parse_train_log(args.train_log)
#     cfg = Config(); cfg.load(args.cfg)
#     cfg.data_valid = args.test_set if args.test_set else "test.pkl"
#     if args.data_root: cfg.data_root, cfg.data_name = args.data_root, args.data_name

#     # Model eval
#     y_true, y_pred_cls, y_pred_logits = model_eval(cfg, args.checkpoint, args.all_state_dict)

#     # IEEE-style: uniform subplot sizes, bold titles, clean labels, tight layout
#     fig, axs = plt.subplots(1, 4, figsize=(10, 2.5))
#     plt.subplots_adjust(wspace=0.35, hspace=0.1)


#     # Accuracy Curve
#     axs[0].plot(epochs, train_acc, label="Train", color="tab:blue", linewidth=1.5)
#     axs[0].plot(epochs, val_acc, label="Val", color="tab:orange", linewidth=1.5)
#     axs[0].set_title("Model Accuracy", fontsize=10, fontweight="bold")
#     axs[0].set_xlabel("Epochs", fontsize=10); axs[1].set_ylabel("Accuracy", fontsize=10)
#     axs[0].tick_params(axis="both", labelsize=9)
#     axs[0].legend(fontsize=8, loc="best", frameon=True)
#     axs[0].grid(True, linestyle="--", alpha=0.5)

#     # Loss Curve
#     axs[1].plot(epochs, train_loss, label="Train", color="tab:blue", linewidth=1.5)
#     axs[1].plot(epochs, val_loss, label="Val", color="tab:orange", linewidth=1.5)
#     axs[1].set_title("Model Loss ", fontsize=10, fontweight="bold")
#     axs[1].set_xlabel("Epochs", fontsize=10); axs[0].set_ylabel("Loss", fontsize=10)
#     axs[1].tick_params(axis="both", labelsize=9)
#     axs[1].legend(fontsize=8, loc="best", frameon=True)
#     axs[1].grid(True, linestyle="--", alpha=0.5)


#     # AUC-ROC
#     plot_auc_roc(axs[2], y_true, y_pred_logits)

#     # Confusion Matrix
#     plot_confmat(axs[3], y_true, y_pred_cls)

#     # Save
#     plt.tight_layout()
#     plt.savefig(f"{args.output_prefix}_model_report.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
#     plt.savefig(f"{args.output_prefix}_model_report.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
#     plt.show()

# def arg_parser():
#     parser = argparse.ArgumentParser(description="IEEE-style 4-in-1 model evaluation report.")
#     parser.add_argument("--train_log", type=str, required=True, help="Path to train.log")
#     parser.add_argument("--checkpoint", type=str, required=True, help="Path to best model checkpoint")
#     parser.add_argument("--cfg", type=str, required=True, help="Path to cfg.log")
#     parser.add_argument("--output_prefix", type=str, default="model", help="Prefix for output files")
#     parser.add_argument("--test_set", type=str, default=None)
#     parser.add_argument("--data_root", type=str, default=None)
#     parser.add_argument("--data_name", type=str, default=None)
#     parser.add_argument("--all_state_dict", action="store_true")
#     return parser.parse_args()

# if __name__ == "__main__":
#     main(arg_parser())
