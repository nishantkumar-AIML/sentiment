#!/usr/bin/env python3
"""
Step 2: Classical Benchmark Suite (6 Standard Models)
====================================================
Framework: Adversarial Affective Representation Disentanglement with Adaptive Prototype Margins
Paper: "Adversarial Affective Representation Disentanglement with Adaptive
        Prototype Margins for Long-Tailed Text Emotion Recognition"
Author: Nishant Kumar
Target Journal: IEEE Transactions on Affective Computing (TAFFC)

Trains and evaluates 6 classical and recurrent baselines:
  1. Multinomial Naive Bayes (MNB)
  2. Multinomial Logistic Regression
  3. Random Forest (100 Trees)
  4. Linear Support Vector Machine (LinearSVC)
  5. XGBoost Classifier
  6. Bidirectional LSTM (PyTorch)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import csv, json, time, collections
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

EMOTION_NAMES = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]

def load_data():
    with open("dataset/vad_augmented_splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    train_texts = [x["text"] for x in splits["train"]]
    train_labels = [x["label"] for x in splits["train"]]
    val_texts = [x["text"] for x in splits["val"]]
    val_labels = [x["label"] for x in splits["val"]]
    test_texts = [x["text"] for x in splits["test"]]
    test_labels = [x["label"] for x in splits["test"]]
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

def eval_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_cls, r_cls, f1_cls, supp = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    per_class = {}
    for i, name in enumerate(EMOTION_NAMES):
        per_class[name] = {
            "precision": round(float(p_cls[i]), 4),
            "recall": round(float(r_cls[i]), 4),
            "f1": round(float(f1_cls[i]), 4),
            "support": int(supp[i])
        }
    return {
        "accuracy": round(float(acc) * 100, 2),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_wt), 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, num_classes=6, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, (hn, _) = self.lstm(emb)
        pooled = torch.mean(out, dim=1)
        return self.fc(self.dropout(pooled))

def run_all_classical():
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_data()
    print(f"Loaded: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    results = {}

    # 1. MNB
    print("[*] Training Multinomial Naive Bayes...", flush=True)
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train, y_train)
    results["Multinomial Naive Bayes"] = eval_metrics(y_test, mnb.predict(X_test))

    # 2. Logistic Regression
    print("[*] Training Logistic Regression...", flush=True)
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_train, y_train)
    results["Multinomial Logistic Regression"] = eval_metrics(y_test, lr.predict(X_test))

    # 3. Random Forest
    print("[*] Training Random Forest...", flush=True)
    rf = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)
    results["Random Forest (100 Trees)"] = eval_metrics(y_test, rf.predict(X_test))

    # 4. Linear SVM
    print("[*] Training Linear SVM...", flush=True)
    svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    svm.fit(X_train, y_train)
    results["Linear Support Vector Machine"] = eval_metrics(y_test, svm.predict(X_test))

    # 5. XGBoost
    print("[*] Training XGBoost...", flush=True)
    xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3, random_state=42, n_jobs=1, objective="multi:softprob")
    xgb_clf.fit(X_train, y_train)
    results["XGBoost Classifier"] = eval_metrics(y_test, xgb_clf.predict(X_test))

    # 6. Bi-LSTM
    print("[*] Training Bi-LSTM...", flush=True)
    # Build token vocab
    word_to_id = {"<PAD>": 0, "<UNK>": 1}
    for t in train_texts:
        for w in t.lower().split():
            if w not in word_to_id and len(word_to_id) < 10000:
                word_to_id[w] = len(word_to_id)

    def text_to_ids(text, max_len=50):
        ids = [word_to_id.get(w, 1) for w in text.lower().split()[:max_len]]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids

    X_tr_ids = torch.tensor([text_to_ids(t) for t in train_texts], dtype=torch.long)
    y_tr_t = torch.tensor(train_labels, dtype=torch.long)
    X_te_ids = torch.tensor([text_to_ids(t) for t in test_texts], dtype=torch.long)
    y_te_t = torch.tensor(test_labels, dtype=torch.long)

    torch.manual_seed(42)
    bilstm = BiLSTMClassifier(len(word_to_id))
    b_opt = torch.optim.Adam(bilstm.parameters(), lr=1e-3)
    b_crit = nn.CrossEntropyLoss()
    b_loader = DataLoader(TensorDataset(X_tr_ids, y_tr_t), batch_size=64, shuffle=True)

    bilstm.train()
    for ep in range(3):
        for bx, by in b_loader:
            b_opt.zero_grad()
            out = bilstm(bx)
            loss = b_crit(out, by)
            loss.backward()
            b_opt.step()

    bilstm.eval()
    with torch.no_grad():
        preds = torch.argmax(bilstm(X_te_ids), dim=-1).numpy()
    results["Bi-LSTM (3-Epoch PyTorch)"] = eval_metrics(y_test, preds)

    with open("results/final_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(" CLASSICAL BENCHMARK RESULTS (CLEAN SPLIT 15,923 / 1,993 / 2,000)")
    print("=" * 80)
    for k, v in results.items():
        s_f1 = v["per_class"]["Surprise"]["f1"]
        l_f1 = v["per_class"]["Love"]["f1"]
        print(f"{k:<35} | Acc: {v['accuracy']:>6.2f}% | Macro-F1: {v['macro_f1']:>6.4f} | Surprise F1: {s_f1:>6.4f} | Love F1: {l_f1:>6.4f}")
    print("=" * 80)

if __name__ == "__main__":
    run_all_classical()
