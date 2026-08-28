#!/usr/bin/env python3
"""
Class-Balancing Strategies Comparison Experiment (XGBoost Benchmark)
=====================================================================
Paper: "Fine-Grained Emotion Classification in Social Media Text:
        A Comparative Study of Classical Ensembles and Deep Sequence Models Under Class Imbalance"

This script evaluates the 4 class-balancing strategies on XGBoost (Table 6 in paper):
  1. Baseline (Unweighted Standard Loss)
  2. Inverse Class-Weighted Loss (w_c = N / (C * N_c))
  3. Random Oversampling (ROS)
  4. TF-IDF Synthetic Vector Interpolation (SMOTE-style k=5)
"""

import os
import csv
import re
import json
import collections
import numpy as np
import xgboost as xgb
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

SEED = 42
np.random.seed(SEED)

def load_clean_data():
    def read_csv(path):
        texts, labels = [], []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    clean_t = re.sub(r'https?://\S+|www\.\S+', '', row[0].lower())
                    clean_t = re.sub(r'@\w+', '', clean_t)
                    clean_t = re.sub(r'[^a-zA-Z\s]', ' ', clean_t)
                    clean_t = re.sub(r'\s+', ' ', clean_t).strip()
                    texts.append(clean_t)
                    labels.append(int(row[1]))
        return texts, labels

    train_texts_raw, train_labels_raw = read_csv("dataset/training.csv")
    val_texts, _ = read_csv("dataset/validation.csv")
    test_texts, test_labels = read_csv("dataset/test.csv")

    val_set = set(val_texts)
    test_set = set(test_texts)
    clean_train_texts, clean_train_labels = [], []
    for t, y in zip(train_texts_raw, train_labels_raw):
        if t not in val_set and t not in test_set:
            clean_train_texts.append(t)
            clean_train_labels.append(y)

    return clean_train_texts, clean_train_labels, test_texts, test_labels

def main():
    print("=" * 80, flush=True)
    print(" EXECUTING CLASS-BALANCING STRATEGIES COMPARISON (XGBOOST BENCHMARK)", flush=True)
    print("=" * 80, flush=True)
    train_texts, train_y, test_texts, test_y = load_clean_data()

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True, norm='l2')
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    results = {}

    # 1. Baseline
    print("[*] Running Strategy 1/4: Baseline (Unweighted)...", flush=True)
    clf_base = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3, objective='multi:softprob', random_state=SEED, n_jobs=1, eval_metric='mlogloss')
    clf_base.fit(X_train, train_y)
    preds_base = clf_base.predict(X_test)
    acc_base = accuracy_score(test_y, preds_base)
    _, _, f1_per_class_base, _ = precision_recall_fscore_support(test_y, preds_base, average=None, zero_division=0)
    _, _, mf1_base, _ = precision_recall_fscore_support(test_y, preds_base, average='macro', zero_division=0)
    results["Baseline (Unweighted)"] = {
        "accuracy": round(acc_base, 4),
        "accuracy_pct": f"{acc_base*100:.2f}%",
        "surprise_f1": round(f1_per_class_base[5], 4),
        "macro_f1": round(mf1_base, 4)
    }

    # 2. Class-Weighted Loss
    print("[*] Running Strategy 2/4: Class-Weighted Loss...", flush=True)
    counts = collections.Counter(train_y)
    N = len(train_y)
    num_classes = 6
    weights = {c: N / (num_classes * counts[c]) for c in range(num_classes)}
    sample_weights = np.array([weights[y] for y in train_y])

    clf_weighted = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3, objective='multi:softprob', random_state=SEED, n_jobs=1, eval_metric='mlogloss')
    clf_weighted.fit(X_train, train_y, sample_weight=sample_weights)
    preds_weighted = clf_weighted.predict(X_test)
    acc_w = accuracy_score(test_y, preds_weighted)
    _, _, f1_per_class_w, _ = precision_recall_fscore_support(test_y, preds_weighted, average=None, zero_division=0)
    _, _, mf1_w, _ = precision_recall_fscore_support(test_y, preds_weighted, average='macro', zero_division=0)
    results["Class-Weighted Loss"] = {
        "accuracy": round(acc_w, 4),
        "accuracy_pct": f"{acc_w*100:.2f}%",
        "surprise_f1": round(f1_per_class_w[5], 4),
        "macro_f1": round(mf1_w, 4)
    }

    # 3. Random Oversampling (ROS)
    print("[*] Running Strategy 3/4: Random Oversampling (ROS)...", flush=True)
    max_count = max(counts.values())
    ros_indices = []
    for c in range(num_classes):
        c_idx = [i for i, y in enumerate(train_y) if y == c]
        sampled_idx = np.random.choice(c_idx, size=max_count, replace=True)
        ros_indices.extend(sampled_idx)

    X_train_ros = X_train[ros_indices]
    y_train_ros = np.array(train_y)[ros_indices]

    clf_ros = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3, objective='multi:softprob', random_state=SEED, n_jobs=1, eval_metric='mlogloss')
    clf_ros.fit(X_train_ros, y_train_ros)
    preds_ros = clf_ros.predict(X_test)
    acc_ros = accuracy_score(test_y, preds_ros)
    _, _, f1_per_class_ros, _ = precision_recall_fscore_support(test_y, preds_ros, average=None, zero_division=0)
    _, _, mf1_ros, _ = precision_recall_fscore_support(test_y, preds_ros, average='macro', zero_division=0)
    results["Random Oversampling"] = {
        "accuracy": round(acc_ros, 4),
        "accuracy_pct": f"{acc_ros*100:.2f}%",
        "surprise_f1": round(f1_per_class_ros[5], 4),
        "macro_f1": round(mf1_ros, 4)
    }

    # 4. TF-IDF Synthetic Oversampling (SMOTE-style feature interpolation)
    print("[*] Running Strategy 4/4: TF-IDF Synthetic Oversampling...", flush=True)
    surprise_idx = [i for i, y in enumerate(train_y) if y == 5]
    X_surprise = X_train[surprise_idx].toarray()
    n_synth = 1000
    synth_samples = []
    for _ in range(n_synth):
        idx_a = np.random.randint(0, len(surprise_idx))
        idx_b = np.random.randint(0, len(surprise_idx))
        lam = np.random.uniform(0.1, 0.9)
        synth_vec = lam * X_surprise[idx_a] + (1 - lam) * X_surprise[idx_b]
        norm = np.linalg.norm(synth_vec)
        if norm > 0:
            synth_vec = synth_vec / norm
        synth_samples.append(synth_vec)

    X_train_smote = sp.vstack([X_train, sp.csr_matrix(np.array(synth_samples))])
    y_train_smote = np.concatenate([np.array(train_y), np.full(n_synth, 5)])

    clf_smote = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3, objective='multi:softprob', random_state=SEED, n_jobs=1, eval_metric='mlogloss')
    clf_smote.fit(X_train_smote, y_train_smote)
    preds_smote = clf_smote.predict(X_test)
    acc_smote = accuracy_score(test_y, preds_smote)
    _, _, f1_per_class_smote, _ = precision_recall_fscore_support(test_y, preds_smote, average=None, zero_division=0)
    _, _, mf1_smote, _ = precision_recall_fscore_support(test_y, preds_smote, average='macro', zero_division=0)
    results["TF-IDF Synthetic Oversampling"] = {
        "accuracy": round(acc_smote, 4),
        "accuracy_pct": f"{acc_smote*100:.2f}%",
        "surprise_f1": round(f1_per_class_smote[5], 4),
        "macro_f1": round(mf1_smote, 4)
    }

    print("\n" + "=" * 75, flush=True)
    print(f"{'Strategy':<25} | {'Overall Accuracy':<18} | {'Surprise F1':<14} | {'Macro-F1':<10}", flush=True)
    print("-" * 75, flush=True)
    for strat, v in results.items():
        print(f"{strat:<25} | {v['accuracy_pct']:>16}    | {v['surprise_f1']:>12.4f}  | {v['macro_f1']:>8.4f}", flush=True)
    print("-" * 75, flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/class_balancing_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("[*] Saved class-balancing results to results/class_balancing_results.json", flush=True)

if __name__ == "__main__":
    main()
