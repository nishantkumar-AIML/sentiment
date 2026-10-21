#!/usr/bin/env python3
"""
Step 3: Transformer & Disentangled Framework Experiment Runner
=============================================================
Framework: Adversarial Affective Representation Disentanglement with Adaptive Prototype Margins
Paper: "Adversarial Affective Representation Disentanglement with Adaptive
        Prototype Margins for Long-Tailed Text Emotion Recognition"
Author: Nishant Kumar
Target Journal: IEEE Transactions on Affective Computing (TAFFC)

Executes deep transformer models in isolated processes to ensure clean memory
management and exact reproducibility.

Supported experiments (<exp_name>):
  - ModernBERT_Baseline: Standard ModernBERT fine-tuned with Cross-Entropy Loss (92.30% Acc, 0.8773 Macro-F1)
  - Proposed_Framework_Full: Full proposed disentangled architecture (92.90% Acc, 0.8866 Macro-F1)
  - Ablation_No_Ortho: Full framework without subspace orthogonality constraint (92.45% Acc, 0.8776 Macro-F1)
  - Ablation_No_Adv: Full framework without adversarial GRL emotion erasure (92.25% Acc, 0.8774 Macro-F1)
  - Auxiliary_Continuous_VAD: Framework with auxiliary continuous VAD regression (92.65% Acc, 0.8707 Macro-F1)

Usage:
  python3 step3_train_transformer.py <exp_name>
"""

import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from models.vad_causal_disentangler import CausalAffectiveDisentangler, AttentionPooling

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

EMOTION_NAMES = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]

class EmotionVADDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=64):
        self.texts = [item["text"] for item in data_list]
        self.labels = [item["label"] for item in data_list]
        self.vads = [item["vad"] for item in data_list]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "vad": torch.tensor(self.vads[idx], dtype=torch.float32)
        }

class StandardTransformerClassifier(nn.Module):
    def __init__(self, backbone, hidden_dim, num_classes=6, dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.pooler = AttentionPooling(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = self.pooler(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out["logits"]
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)

    per_class = {}
    for i, name in enumerate(EMOTION_NAMES):
        per_class[name] = {
            "precision": round(float(p_cls[i]), 4),
            "recall": round(float(r_cls[i]), 4),
            "f1": round(float(f1_cls[i]), 4),
            "support": int(supp_cls[i])
        }

    return {
        "accuracy": round(float(acc) * 100, 2),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_wt), 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist()
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 step3_train_transformer.py <exp_name>")
        sys.exit(1)

    exp_name = sys.argv[1]
    print(f"=== RUNNING SINGLE EXPERIMENT: {exp_name} ===", flush=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with open("dataset/vad_augmented_splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    with open("results/class_vad_anchors.json", "r", encoding="utf-8") as f:
        anchors = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    train_loader = DataLoader(EmotionVADDataset(splits["train"], tokenizer), batch_size=32, shuffle=True)
    val_loader = DataLoader(EmotionVADDataset(splits["val"], tokenizer), batch_size=32, shuffle=False)
    test_loader = DataLoader(EmotionVADDataset(splits["test"], tokenizer), batch_size=32, shuffle=False)

    torch.manual_seed(42)
    np.random.seed(42)

    backbone = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
    hidden_dim = backbone.config.hidden_size

    # Configure parameters based on experiment
    if exp_name == "ModernBERT_Baseline":
        model = StandardTransformerClassifier(backbone, hidden_dim).to(device)
        optimizer_grouped_parameters = [
            {"params": model.backbone.parameters(), "lr": 2e-5, "weight_decay": 0.01},
            {"params": model.classifier.parameters(), "lr": 1e-4, "weight_decay": 0.001}
        ]
    else:
        if exp_name in ["Proposed_Framework_Full", "Proposed_Disentangled", "Ablation_Static_Margins"]:
            # Full proposed disentangled framework with adaptive prototype margins (92.90% Acc, 0.8866 Macro-F1)
            kwargs = dict(alpha=0.5, gamma=0.0, tau=0.1, lambda_vad=0.0, lambda_ortho=0.3, lambda_adv=0.1)
        elif exp_name == "Ablation_No_Ortho":
            # Ablation: without subspace orthogonality constraint
            kwargs = dict(alpha=0.5, gamma=0.0, tau=0.1, lambda_vad=0.0, lambda_ortho=0.0, lambda_adv=0.1)
        elif exp_name == "Ablation_No_Adv":
            # Ablation: without adversarial emotion erasure GRL
            kwargs = dict(alpha=0.5, gamma=0.0, tau=0.1, lambda_vad=0.0, lambda_ortho=0.3, lambda_adv=0.0)
        elif exp_name in ["Auxiliary_Continuous_VAD", "Ablation_With_Continuous_VAD"]:
            # Ablation: with auxiliary continuous VAD regression loss
            kwargs = dict(alpha=0.5, gamma=2.0, tau=0.1, lambda_vad=0.05, lambda_ortho=0.3, lambda_adv=0.1)
        else:
            raise ValueError(f"Unknown experiment: {exp_name}. Choose from: ['ModernBERT_Baseline', 'Proposed_Framework_Full', 'Ablation_No_Ortho', 'Ablation_No_Adv', 'Auxiliary_Continuous_VAD']")

        model = CausalAffectiveDisentangler(
            backbone=backbone,
            hidden_dim=hidden_dim,
            proj_dim=256,
            num_classes=6,
            class_counts=anchors["class_counts"],
            class_vad_anchors=anchors["class_vad_anchors"],
            **kwargs
        ).to(device)
        optimizer_grouped_parameters = [
            {"params": model.backbone.parameters(), "lr": 2e-5, "weight_decay": 0.01},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")], "lr": 1e-4, "weight_decay": 0.001}
        ]

    opt = torch.optim.AdamW(optimizer_grouped_parameters)
    epochs = 3
    total_steps = len(train_loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    best_val_f1 = -1.0
    best_weights = None
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            opt.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["label"].to(device),
                vad_targets=batch["vad"].to(device)
            )
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += out["loss"].item()

        val_eval = evaluate_model(model, val_loader, device)
        if val_eval["macro_f1"] > best_val_f1:
            best_val_f1 = val_eval["macro_f1"]
            best_weights = copy.deepcopy(model.state_dict())
        print(f"  Epoch {ep+1}/{epochs} | Val Macro-F1: {val_eval['macro_f1']:.4f}", flush=True)

    model.load_state_dict(best_weights)
    test_eval = evaluate_model(model, test_loader, device)
    test_eval["duration_sec"] = round(time.time() - t0, 2)
    test_eval["best_val_macro_f1"] = round(best_val_f1, 4)

    print(f"\n[RESULT] {exp_name}: Acc = {test_eval['accuracy']}%, Macro-F1 = {test_eval['macro_f1']}, Surprise F1 = {test_eval['per_class']['Surprise']['f1']}", flush=True)

    # Save or update in results/sota_experiments_results.json
    results_path = "results/sota_experiments_results.json"
    data = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            data = {}

    data[exp_name] = test_eval
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[✓] Saved {exp_name} to {results_path}")

if __name__ == "__main__":
    main()
