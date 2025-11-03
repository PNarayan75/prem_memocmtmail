
# train_drkf_from_pickles.py
# =========================================
# Train DRKF (audio+text) directly from your preprocessed IEMOCAP pickles.
# Your pickles must contain a list of tuples: (wav_path, text, label)
# =========================================

import os
import math
import json
import time
import random
import pickle
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model,
    RobertaTokenizerFast,
    RobertaModel
)

# -------------------------
# Config
# -------------------------
class Cfg:
    # === paths ===
    PREPROC_DIR = "/kaggle/working/prem_memocmtmail/IEMOCAP_preprocessed"  # <-- your OUTPUT_DIR
    TRAIN_PKL = os.path.join(PREPROC_DIR, "train.pkl")
    VAL_PKL   = os.path.join(PREPROC_DIR, "val.pkl")
    TEST_PKL  = os.path.join(PREPROC_DIR, "test.pkl")

    # === audio/text encoders ===
    WAV2VEC2_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    ROBERTA_NAME  = "roberta-large"

    # === training ===
    SAMPLE_RATE = 16000
    NUM_CLASSES = 4
    LABEL_NAMES = ['ang','hap','sad','neu']
    BATCH_SIZE  = 4
    EPOCHS      = 5               # start small; increase later (paper ~100)
    LR          = 1e-5
    WEIGHT_DECAY= 0.01
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    SEED        = 42

    # === DRKF loss weights (from paper) ===
    BETA = 0.2    # contrastive
    GAMMA = 1.0   # classification
    DELTA = 0.2   # consistency (binary)
    # L_a (augmentation) carries its own internal weighting terms

    LOG_INTERVAL = 20

random.seed(Cfg.SEED)
np.random.seed(Cfg.SEED)
torch.manual_seed(Cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Cfg.SEED)

# -------------------------
# Dataset & Collate
# -------------------------
class IemocapPickleDataset(Dataset):
    """
    Expects a pickle with a list of (wav_path, text, label).
    """
    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.items = pickle.load(f)
        # sanity
        self.items = [it for it in self.items if Path(it[0]).exists()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, text, label = self.items[idx]
        # also derive session id from path, if needed later
        session = "unknown"
        for s in ["Session1","Session2","Session3","Session4","Session5"]:
            if s in wav_path:
                session = s
                break
        return {
            "wav_path": wav_path,
            "text": text,
            "label": int(label),
            "session": session
        }

class CollateDRKF:
    """
    Tokenizes audio via Wav2Vec2Processor and text via RobertaTokenizerFast.
    Also prepares a mismatched (shuffled) set of texts for the ED (consistency) head.
    """
    def __init__(self, wav_proc, txt_tok, sample_rate=16000, device="cpu"):
        self.wav_proc = wav_proc
        self.txt_tok  = txt_tok
        self.sr       = sample_rate
        self.device   = device

    def __call__(self, batch):
        # 1) gather
        wavs, texts, labels = [], [], []
        wavs_lengths = []
        for b in batch:
            path = b["wav_path"]
            try:
                y, sr = sf.read(path, always_2d=False)
                if sr != self.sr:
                    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=self.sr)
                else:
                    y = y.astype(np.float32)
                # clip extreme amplitudes (safety)
                y = np.clip(y, -1.0, 1.0)
            except Exception as e:
                # fallback: zero if reading fails
                y = np.zeros(int(1.0*self.sr), dtype=np.float32)
            wavs.append(y)
            texts.append(b["text"] if b["text"] is not None else "")
            labels.append(b["label"])
            wavs_lengths.append(len(y))

        # 2) tokenize audio
        wav_inputs = self.wav_proc(
            wavs, sampling_rate=self.sr, return_tensors="pt", padding=True
        )

        # 3) tokenize text (matched)
        txt_inputs = self.txt_tok(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )

        # 4) build a mismatched (shuffled) text batch for ED
        idx = list(range(len(texts)))
        random.shuffle(idx)
        texts_mismatch = [texts[i] for i in idx]
        txt_inputs_mis = self.txt_tok(
            texts_mismatch, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        # consistency labels: 1 for matched, 0 for mismatched
        ed_labels = torch.cat([torch.ones(len(texts)), torch.zeros(len(texts))], dim=0)

        # pack
        batch_out = {
            "wav_inputs": {k: v for k, v in wav_inputs.items()},
            "txt_inputs": {k: v for k, v in txt_inputs.items()},
            "txt_inputs_mis": {k: v for k, v in txt_inputs_mis.items()},
            "labels": torch.tensor(labels, dtype=torch.long),
            "ed_labels": ed_labels,
            "shuffle_idx": torch.tensor(idx, dtype=torch.long),
            "wavs_lengths": torch.tensor(wavs_lengths, dtype=torch.long)
        }

        # move to device in trainer loop (to avoid pinning CPU RAM here)
        return batch_out

# -------------------------
# DRKF Model
# -------------------------
class ResAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.dec = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
    def forward(self, x):                # x: [B, T, D]
        z = x + self.enc(x)              # residual encode
        x_hat = z + self.dec(z)          # residual decode as augmenter
        return x_hat

class Proj(nn.Module):
    def __init__(self, d, out=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, out))
    def forward(self, x):                # x: [B, D]
        return F.normalize(self.net(x), dim=-1)

class FusionEncoder(nn.Module):
    def __init__(self, d, nheads=8, nlayers=1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d, nheads, dim_feedforward=4*d, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.sep = nn.Parameter(torch.zeros(1, 1, d))
    def forward(self, S_seq, T_seq):     # [B, TS, D], [B, TT, D]
        B = S_seq.size(0)
        cls = self.cls.expand(B, -1, -1)
        sep = self.sep.expand(B, 1, -1)
        x = torch.cat([cls, S_seq, sep, T_seq], dim=1)   # [B, 1+TS+1+TT, D]
        h = self.tr(x)
        return h[:, 0]                   # fused CLS

class DRKF(nn.Module):
    def __init__(self, n_classes=4, audio_name=Cfg.WAV2VEC2_NAME, text_name=Cfg.ROBERTA_NAME):
        super().__init__()
        self.a_enc = Wav2Vec2Model.from_pretrained(audio_name)
        self.t_enc = RobertaModel.from_pretrained(text_name)
        d_a = self.a_enc.config.hidden_size
        d_t = self.t_enc.config.hidden_size

        self.a_aug = ResAE(d_a)
        self.t_aug = ResAE(d_t)

        self.a_proj = Proj(d_a)
        self.t_proj = Proj(d_t)

        d_f = min(d_a, d_t)
        self.fuse = FusionEncoder(d=d_f, nheads=8, nlayers=1)
        self.a_down = nn.Linear(d_a, d_f)
        self.t_down = nn.Linear(d_t, d_f)

        self.ec = nn.Sequential(nn.Linear(d_f, 2*d_f),
                                nn.ReLU(), nn.Linear(2*d_f, n_classes))
        self.ed = nn.Sequential(nn.Linear(d_f, 2*d_f),
                                nn.ReLU(), nn.Linear(2*d_f, 1))

    def forward_once(self, wav_inputs, txt_inputs):
        # encoders
        a_seq = self.a_enc(**wav_inputs).last_hidden_state          # [B, Ta, Da]
        t_seq = self.t_enc(**txt_inputs).last_hidden_state          # [B, Tt, Dt]

        # progressive augmentation
        a_aug = self.a_aug(a_seq)
        t_aug = self.t_aug(t_seq)

        # CLS pooling
        a_cls = a_aug.mean(1)                                       # [B, Da]
        t_cls = t_aug.mean(1)                                       # [B, Dt]

        # projections
        za = self.a_proj(a_cls)
        zt = self.t_proj(t_cls)

        # fusion
        a_small = self.a_down(a_aug)                                # [B, Ta, Df]
        t_small = self.t_down(t_aug)                                # [B, Tt, Df]
        fused = self.fuse(a_small, t_small)                         # [B, Df]

        # heads
        logits_ec = self.ec(fused)                                  # emotion logits
        logit_ed  = self.ed(fused).squeeze(-1)                      # consistency logit (single pass)

        return dict(a_seq=a_seq, t_seq=t_seq,
                    a_aug=a_aug, t_aug=t_aug,
                    za=za, zt=zt,
                    logits_ec=logits_ec, logit_ed=logit_ed)

# -------------------------
# Losses
# -------------------------
def info_nce(z1, z2, tau=0.07):  # z1,z2: [B, D], positives on diagonal
    z1 = F.normalize(z1, dim=-1); z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

def aug_loss(a_aug, a_seq, t_aug, t_seq, y_onehot, ec_head):
    # MSE to keep augmented in same subspace
    mse = F.mse_loss(a_aug, a_seq) + F.mse_loss(t_aug, t_seq)
    # KL to encourage augmented representations to align with label distribution (via ec head)
    with torch.no_grad():
        a_logits = ec_head(a_aug.mean(1))
        t_logits = ec_head(t_aug.mean(1))
        a_logp = F.log_softmax(a_logits, dim=-1)
        t_logp = F.log_softmax(t_logits, dim=-1)
    kl = F.kl_div(a_logp, y_onehot, reduction='batchmean') + F.kl_div(t_logp, y_onehot, reduction='batchmean')
    return mse + kl

# -------------------------
# Training / Eval
# -------------------------
def move_to_device(batch, device):
    out = {}
    out["labels"] = batch["labels"].to(device)
    out["ed_labels"] = batch["ed_labels"].to(device)
    out["shuffle_idx"] = batch["shuffle_idx"].to(device)
    out["wavs_lengths"] = batch["wavs_lengths"].to(device)
    out["wav_inputs"] = {k: v.to(device) for k, v in batch["wav_inputs"].items()}
    out["txt_inputs"] = {k: v.to(device) for k, v in batch["txt_inputs"].items()}
    out["txt_inputs_mis"] = {k: v.to(device) for k, v in batch["txt_inputs_mis"].items()}
    return out

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        b = move_to_device(batch, device)
        # matched forward only for EC
        out = model.forward_once(b["wav_inputs"], b["txt_inputs"])
        logits = out["logits_ec"]
        pred = logits.argmax(dim=-1).cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(b["labels"].cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=Cfg.LABEL_NAMES, digits=4)
    return acc, bacc, report

def train():
    # processors
    wav_proc = Wav2Vec2Processor.from_pretrained(Cfg.WAV2VEC2_NAME)
    txt_tok  = RobertaTokenizerFast.from_pretrained(Cfg.ROBERTA_NAME)

    # data
    ds_train = IemocapPickleDataset(Cfg.TRAIN_PKL)
    ds_val   = IemocapPickleDataset(Cfg.VAL_PKL) if Path(Cfg.VAL_PKL).exists() else None
    ds_test  = IemocapPickleDataset(Cfg.TEST_PKL) if Path(Cfg.TEST_PKL).exists() else None

    collate = CollateDRKF(wav_proc, txt_tok, sample_rate=Cfg.SAMPLE_RATE, device=Cfg.DEVICE)
    dl_train = DataLoader(ds_train, batch_size=Cfg.BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate)
    dl_val   = DataLoader(ds_val, batch_size=Cfg.BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate) if ds_val else None
    dl_test  = DataLoader(ds_test, batch_size=Cfg.BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate) if ds_test else None

    # model
    model = DRKF(n_classes=Cfg.NUM_CLASSES).to(Cfg.DEVICE)

    # (optional) freeze most encoder layers initially for stability
    for p in model.a_enc.parameters():
        p.requires_grad = False
    for p in list(model.a_enc.parameters())[-4:]:
        p.requires_grad = True
    for p in model.t_enc.parameters():
        p.requires_grad = False
    for p in list(model.t_enc.parameters())[-4:]:
        p.requires_grad = True

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=Cfg.LR, weight_decay=Cfg.WEIGHT_DECAY)

    best_bacc = -1.0
    best_path = os.path.join(Cfg.PREPROC_DIR, "drkf_best.pt")

    for epoch in range(1, Cfg.EPOCHS+1):
        model.train()
        t0 = time.time()
        running = {"loss":0.0, "lf":0.0, "lc":0.0, "lb":0.0, "la":0.0}
        for step, batch in enumerate(dl_train, 1):
            b = move_to_device(batch, Cfg.DEVICE)

            # ---------- forward: matched ----------
            out_match = model.forward_once(b["wav_inputs"], b["txt_inputs"])
            # ---------- forward: mismatched (replace text) ----------
            out_mis   = model.forward_once(b["wav_inputs"], b["txt_inputs_mis"])

            # labels
            y = b["labels"]
            y_onehot = F.one_hot(y, num_classes=Cfg.NUM_CLASSES).float()

            # Loss components
            # L_f: classification on matched only
            L_f = F.cross_entropy(out_match["logits_ec"], y)

            # L_c: contrastive across a↔a, t↔t, a↔t (matched)
            L_c = info_nce(out_match["za"], out_match["za"]) \
                + info_nce(out_match["zt"], out_match["zt"]) \
                + info_nce(out_match["za"], out_match["zt"])

            # L_b: consistency — matched should be 1, mismatched 0
            ed_logits_all = torch.cat([out_match["logit_ed"], out_mis["logit_ed"]], dim=0)
            L_b = F.binary_cross_entropy_with_logits(ed_logits_all, b["ed_labels"])

            # L_a: augmentation
            L_a = aug_loss(out_match["a_aug"], out_match["a_seq"],
                           out_match["t_aug"], out_match["t_seq"],
                           y_onehot, model.ec)

            # total
            L = L_a + Cfg.BETA*L_c + Cfg.GAMMA*L_f + Cfg.DELTA*L_b

            optim.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running["loss"] += L.item()
            running["lf"]   += L_f.item()
            running["lc"]   += L_c.item()
            running["lb"]   += L_b.item()
            running["la"]   += L_a.item()

            if step % Cfg.LOG_INTERVAL == 0:
                avg = {k: v/step for k,v in running.items()}
                print(f"Epoch {epoch} | Step {step}/{len(dl_train)} | "
                      f"loss {avg['loss']:.4f} | Lf {avg['lf']:.4f} | Lc {avg['lc']:.4f} | "
                      f"Lb {avg['lb']:.4f} | La {avg['la']:.4f}")

        # ---- eval ----
        if dl_val:
            val_acc, val_bacc, _ = evaluate(model, dl_val, Cfg.DEVICE)
            print(f"[Val] epoch {epoch}: ACC={val_acc:.4f} | WACC={val_bacc:.4f} | time={time.time()-t0:.1f}s")
            # save best
            if val_bacc > best_bacc:
                best_bacc = val_bacc
                torch.save(model.state_dict(), best_path)
                print(f"Saved best model to: {best_path}")
        else:
            print(f"Epoch {epoch} done. (No val set)")

    # ---- load best and test ----
    if dl_val and Path(best_path).exists():
        model.load_state_dict(torch.load(best_path, map_location=Cfg.DEVICE))

    if dl_test:
        test_acc, test_bacc, test_report = evaluate(model, dl_test, Cfg.DEVICE)
        print("======== TEST ========")
        print(f"ACC  : {test_acc:.4f}")
        print(f"WACC : {test_bacc:.4f}")
        print(test_report)

    # export label names for downstream use
    with open(os.path.join(Cfg.PREPROC_DIR, "drkf_label_names.json"), "w") as f:
        json.dump(Cfg.LABEL_NAMES, f, indent=2)

if __name__ == "__main__":
    train()
