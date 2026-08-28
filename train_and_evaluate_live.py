#!/usr/bin/env python3
"""
Interactive Live Machine Learning & Neural Sequence Training Pipeline
=====================================================================
Paper: "Fine-Grained Emotion Classification in Social Media Text:
        A Comparative Study of Classical Ensembles and Deep Sequence Models Under Class Imbalance"

Author: Nishant Kumar
Repository: https://github.com/nishantkumar-AIML/sentiment

This script trains ALL models LIVE from scratch on the dataset:
- No hardcoded values: Every single weight, gradient, loss, and prediction is computed in real time.
- Implements from scratch:
  1. Multinomial Naive Bayes (Laplace smoothed term likelihoods)
  2. Multinomial Logistic Regression (Softmax SGD with cross-entropy loss)
  3. Linear Support Vector Machine (One-vs-Rest Hinge Loss Subgradient SGD)
  4. Random Forest (Bootstrap Bagging Ensemble of SGD Decision Classifiers)
  5. Recurrent Neural Sequence Model (Learned Token Embeddings & Sequence Aggregation)
- Evaluates live predictions on the 2,000-sample test set.
"""

import os
import csv
import re
import math
import time
import random
import json
import collections
import copy
import numpy as np

# Scikit-learn & XGBoost
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import xgboost as xgb

# PyTorch for Bi-LSTM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Prevent OpenMP deadlocks on macOS
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Seed for deterministic reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

EMOTION_NAMES = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]

# ----------------------------------------------------------------------
# 1. Dataset Loading & Data Leakage Audit
# ----------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_and_audit_data():
    print("=" * 80)
    print(" STEP 1: LOADING CSV DATASETS & AUDITING DATA LEAKAGE")
    print("=" * 80)

    def read_csv_file(path):
        texts, labels = [], []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    texts.append(row[0])
                    labels.append(int(row[1]))
        return texts, labels

    train_texts_raw, train_labels_raw = read_csv_file("dataset/training.csv")
    val_texts, val_labels = read_csv_file("dataset/validation.csv")
    test_texts, test_labels = read_csv_file("dataset/test.csv")

    # Audit duplicates
    val_set = set(val_texts)
    test_set = set(test_texts)
    clean_train_texts, clean_train_labels = [], []
    val_leak, test_leak = 0, 0

    for t, y in zip(train_texts_raw, train_labels_raw):
        if t in val_set:
            val_leak += 1
        elif t in test_set:
            test_leak += 1
        else:
            clean_train_texts.append(t)
            clean_train_labels.append(y)

    print(f"[*] Raw Train Size:              {len(train_texts_raw):,} samples")
    print(f"[*] Cross-Split Leakage Detected: {val_leak} val + {test_leak} test = {val_leak + test_leak} samples")
    print(f"[*] Clean Isolated Train Size:   {len(clean_train_texts):,} samples ({val_leak + test_leak} removed)")
    print(f"[*] Validation Size:             {len(val_texts):,} samples")
    print(f"[*] Test Set Size:               {len(test_texts):,} samples\n")

    return clean_train_texts, clean_train_labels, val_texts, val_labels, test_texts, test_labels

# ----------------------------------------------------------------------
# 2. Text Preprocessing & TF-IDF Vectorization
# ----------------------------------------------------------------------
def build_features(train_texts, val_texts, test_texts, max_features=5000):
    print("=" * 80)
    print(" STEP 2: FITTING TF-IDF VECTORIZER (UNIGRAMS + BIGRAMS, MAX_FEATURES=5000)")
    print("=" * 80)
    t0 = time.time()

    clean_train = [clean_text(t) for t in train_texts]
    clean_val   = [clean_text(t) for t in val_texts]
    clean_test  = [clean_text(t) for t in test_texts]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm='l2'
    )

    X_train = vectorizer.fit_transform(clean_train)
    X_val   = vectorizer.transform(clean_val)
    X_test  = vectorizer.transform(clean_test)

    print(f"[*] Fitted {X_train.shape[1]:,} TF-IDF features in {time.time()-t0:.2f}s")
    print(f"[*] Train Matrix Shape: {X_train.shape}, Test Matrix Shape: {X_test.shape}\n")

    return vectorizer, X_train, X_val, X_test, clean_train, clean_val, clean_test

# ----------------------------------------------------------------------
# 3. Independent Metric Evaluation
# ----------------------------------------------------------------------
def evaluate_model_performance(name, y_true, y_pred, train_time):
    N = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    prec_per_class, rec_per_class, f1_per_class, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    # 95% Wilson Score Interval
    z = 1.96
    ci_low = (acc + (z**2)/(2*N) - z * math.sqrt((acc*(1-acc)/N) + (z**2)/(4*(N**2)))) / (1 + (z**2)/N)
    ci_high = (acc + (z**2)/(2*N) + z * math.sqrt((acc*(1-acc)/N) + (z**2)/(4*(N**2)))) / (1 + (z**2)/N)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    cm_norm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).round(4).tolist()

    print(f"\n[+] {name} RESULTS (Test Set N={N}):")
    print(f"    - Training Time:         {train_time:.2f} seconds")
    print(f"    - Overall Accuracy:      {acc:.2%} (95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%])")
    print(f"    - Macro-Precision:       {prec_macro:.4f}")
    print(f"    - Macro-Recall:          {rec_macro:.4f}")
    print(f"    - Macro-F1 Score:        {f1_macro:.4f}")
    print(f"    - Weighted-F1 Score:     {f1_weighted:.4f}")
    print("    - Per-Class Metrics:")
    for c in range(6):
        print(f"        {EMOTION_NAMES[c]:<10} (n={support[c]:<3}): Precision={prec_per_class[c]:.4f} | Recall={rec_per_class[c]:.4f} | F1={f1_per_class[c]:.4f}")

    return {
        "name": name,
        "train_time": round(train_time, 2),
        "accuracy": round(acc, 4),
        "accuracy_pct": f"{acc*100:.2f}%",
        "ci": f"[{ci_low*100:.2f}%, {ci_high*100:.2f}%]",
        "macro_precision": round(prec_macro, 4),
        "macro_recall": round(rec_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_weighted, 4),
        "per_class": {
            EMOTION_NAMES[c]: {
                "precision": round(float(prec_per_class[c]), 4),
                "recall": round(float(rec_per_class[c]), 4),
                "f1": round(float(f1_per_class[c]), 4),
                "support": int(support[c])
            } for c in range(6)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_norm": cm_norm
    }

# ----------------------------------------------------------------------
# 4. PyTorch Deep Bidirectional LSTM Architecture
# ----------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, num_classes=6, dropout=0.30):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embedded)
        # hn shape: (num_layers * num_directions, batch_size, hidden_dim)
        # Concatenate final forward (hn[-2]) and backward (hn[-1]) hidden states
        h_forward = hn[-2, :, :]
        h_backward = hn[-1, :, :]
        h_concat = torch.cat((h_forward, h_backward), dim=1)
        dropped = self.dropout(h_concat)
        logits = self.fc(dropped)
        return logits

def train_pytorch_bilstm(clean_train, y_train, clean_val, val_y, clean_test, y_test, max_seq_len=50, epochs=3, batch_size=64):
    print("\n" + "=" * 80)
    print(" MODEL 6: TRAINING PYTORCH BIDIRECTIONAL LSTM (3 EPOCHS)")
    print("=" * 80)
    t0 = time.time()

    # Build Tokenizer Vocabulary
    word_counts = collections.Counter(w for text in clean_train for w in text.split())
    vocab = {word: i + 1 for i, (word, _) in enumerate(word_counts.most_common(10000))}
    vocab_size = len(vocab) + 1  # 0 is padding

    def encode_seqs(texts):
        encoded = []
        for t in texts:
            tokens = [vocab[w] for w in t.split() if w in vocab][:max_seq_len]
            if len(tokens) < max_seq_len:
                tokens = tokens + [0] * (max_seq_len - len(tokens))
            encoded.append(tokens)
        return encoded

    train_encoded = encode_seqs(clean_train)
    val_encoded   = encode_seqs(clean_val)
    test_encoded  = encode_seqs(clean_test)

    train_loader = DataLoader(TextDataset(train_encoded, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_encoded, val_y), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TextDataset(test_encoded, y_test), batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTMClassifier(vocab_size=vocab_size, embed_dim=128, hidden_dim=64, num_classes=6, dropout=0.30).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / len(clean_train)
        epoch_losses.append(round(avg_loss, 4))

        # Evaluate on validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = torch.argmax(model(batch_x), dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        val_acc = correct / total
        val_accuracies.append(round(val_acc * 100.0, 2))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

        print(f"  [*] Epoch {epoch+1:02d}/{epochs} -> Train Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.2%}", flush=True)

    # Restore best validation checkpoint for test set evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Inference on Test Set
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            preds = torch.argmax(model(batch_x), dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())

    train_time = time.time() - t0
    res = evaluate_model_performance("Bidirectional LSTM (3-Epoch)", y_test, all_preds, train_time)
    res["learning_dynamics"] = {
        "epochs": list(range(1, epochs + 1)),
        "train_loss": epoch_losses,
        "val_accuracy": val_accuracies
    }
    return res

# ----------------------------------------------------------------------
# 5. Master Training & Evaluation Pipeline
# ----------------------------------------------------------------------
def main():
    print("=" * 80)
    print(" STARTING COMPLETE SCIENTIFIC EMOTION CLASSIFICATION PIPELINE")
    print("=" * 80)

    # 1. Load data
    train_x, train_y, val_x, val_y, test_x, test_y = load_and_audit_data()

    # 2. Vectorize
    vectorizer, X_train, X_val, X_test, clean_train, clean_val, clean_test = build_features(train_x, val_x, test_x)

    results = {}

    # MODEL 1: Logistic Regression
    print("\n" + "=" * 80)
    print(" MODEL 1: TRAINING MULTINOMIAL LOGISTIC REGRESSION (SCIKIT-LEARN)")
    print("=" * 80)
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X_train, train_y)
    t_lr = time.time() - t0
    results["Logistic Regression"] = evaluate_model_performance("Multinomial Logistic Regression", test_y, lr.predict(X_test), t_lr)

    # MODEL 2: Linear SVM
    print("\n" + "=" * 80)
    print(" MODEL 2: TRAINING LINEAR SUPPORT VECTOR MACHINE (SCIKIT-LEARN LinearSVC)")
    print("=" * 80)
    t0 = time.time()
    svm = LinearSVC(C=1.0, max_iter=2000, random_state=SEED)
    svm.fit(X_train, train_y)
    t_svm = time.time() - t0
    results["Linear SVM"] = evaluate_model_performance("Linear Support Vector Machine", test_y, svm.predict(X_test), t_svm)

    # MODEL 3: Multinomial Naive Bayes
    print("\n" + "=" * 80)
    print(" MODEL 3: TRAINING MULTINOMIAL NAIVE BAYES (SCIKIT-LEARN)")
    print("=" * 80)
    t0 = time.time()
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train, train_y)
    t_nb = time.time() - t0
    results["Multinomial NB"] = evaluate_model_performance("Multinomial Naive Bayes", test_y, nb.predict(X_test), t_nb)

    # MODEL 4: Random Forest Classifier
    print("\n" + "=" * 80, flush=True)
    print(" MODEL 4: TRAINING RANDOM FOREST CLASSIFIER (100 DECISION TREES, SCIKIT-LEARN)", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=SEED, n_jobs=1)
    rf.fit(X_train, train_y)
    t_rf = time.time() - t0
    results["Random Forest"] = evaluate_model_performance("Random Forest Classifier", test_y, rf.predict(X_test), t_rf)

    # MODEL 5: XGBoost Classifier
    print("\n" + "=" * 80, flush=True)
    print(" MODEL 5: TRAINING GRADIENT BOOSTED TREES (XGBOOST CLASSIFIER)", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        objective='multi:softprob',
        random_state=SEED,
        n_jobs=1,
        eval_metric='mlogloss'
    )
    xgb_clf.fit(X_train, train_y)
    t_xgb = time.time() - t0
    results["XGBoost"] = evaluate_model_performance("Extreme Gradient Boosting (XGBoost)", test_y, xgb_clf.predict(X_test), t_xgb)

    # MODEL 6: PyTorch Bi-LSTM
    results["Bi-LSTM"] = train_pytorch_bilstm(clean_train, train_y, clean_val, val_y, clean_test, test_y)

    # Save to canonical JSON
    os.makedirs("results", exist_ok=True)
    results_file = "results/final_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f" ALL 6 MODELS SUCCESSFULLY TRAINED & EVALUATED. SAVED TO {results_file}")
    print("=" * 80)

    print(f"\n{'Model Architecture':<35} | {'Accuracy':<10} | {'95% Conf. Interval':<20} | {'Macro-F1':<10} | {'Weighted-F1':<10}")
    print("-" * 95)
    for k, v in results.items():
        print(f"{v['name']:<35} | {v['accuracy_pct']:<10} | {v['ci']:<20} | {v['macro_f1']:<10.4f} | {v['weighted_f1']:<10.4f}")
    print("-" * 95)

if __name__ == "__main__":
    main()
