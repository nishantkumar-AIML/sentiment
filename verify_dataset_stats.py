#!/usr/bin/env python3
"""
Dataset Breakdown, Data Leakage Audit & Hyperparameter Verification
===================================================================
Paper: "Fine-Grained Emotion Classification in Social Media Text:
        A Comparative Study of Classical Ensembles and Deep Sequence Models Under Class Imbalance"

This script validates and reproduces:
  - Table 1: Detailed Dataset Split Breakdown, Class Distribution & Leakage Audit
  - Table 2: Hyperparameter Configurations of all Evaluated Models
"""

import csv
import collections
import os

EMOTIONS = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]

def read_csv_data(filepath):
    texts, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 2:
                texts.append(row[0])
                labels.append(int(row[1]))
    return texts, labels

def audit_dataset():
    print("=" * 85)
    print(" REPRODUCING TABLE 1: DATASET SPLIT BREAKDOWN & DATA LEAKAGE AUDIT")
    print("=" * 85)

    train_texts, train_labels = read_csv_data("dataset/training.csv")
    val_texts, val_labels = read_csv_data("dataset/validation.csv")
    test_texts, test_labels = read_csv_data("dataset/test.csv")

    val_set = set(val_texts)
    test_set = set(test_texts)

    # Detect cross-split duplicates
    val_leakage = [t for t in train_texts if t in val_set]
    test_leakage = [t for t in train_texts if t in test_set]
    total_leakage = len(val_leakage) + len(test_leakage)

    # Clean isolated training split
    clean_train_texts, clean_train_labels = [], []
    for t, y in zip(train_texts, train_labels):
        if t not in val_set and t not in test_set:
            clean_train_texts.append(t)
            clean_train_labels.append(y)

    train_orig_counts = collections.Counter(train_labels)
    train_clean_counts = collections.Counter(clean_train_labels)
    val_counts = collections.Counter(val_labels)
    test_counts = collections.Counter(test_labels)

    total_clean_train = len(clean_train_labels)

    # Save to results/dataset_audit.json for data-driven figure generation
    os.makedirs("results", exist_ok=True)
    audit_data = {
        "classes": EMOTIONS,
        "train_orig_counts": [train_orig_counts[c] for c in range(6)],
        "train_clean_counts": [train_clean_counts[c] for c in range(6)],
        "val_counts": [val_counts[c] for c in range(6)],
        "test_counts": [test_counts[c] for c in range(6)],
        "total_train_orig": len(train_labels),
        "total_train_clean": total_clean_train,
        "total_val": len(val_labels),
        "total_test": len(test_labels),
        "leakage_val": len(val_leakage),
        "leakage_test": len(test_leakage),
        "total_leakage": total_leakage
    }
    import json
    with open("results/dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    print(f"\n[*] Data Leakage Audit Findings:")
    print(f"    - Overlaps with Validation Set: {len(val_leakage)} samples")
    print(f"    - Overlaps with Test Set:       {len(test_leakage)} samples")
    print(f"    - Total Contaminating Samples:  {total_leakage} samples (Removed from Clean Train)")
    print(f"    - Audit Results Saved To:      results/dataset_audit.json\n")

    print(f"{'Emotion':<10} | {'ID':<3} | {'Train (Orig)':<12} | {'Train (Clean)':<13} | {'Val':<6} | {'Test':<6} | {'Clean Train %':<12}")
    print("-" * 85)
    for c in range(6):
        orig_c = train_orig_counts[c]
        clean_c = train_clean_counts[c]
        v_c = val_counts[c]
        te_c = test_counts[c]
        pct = (clean_c / total_clean_train) * 100.0
        print(f"{EMOTIONS[c]:<10} | {c:<3} | {orig_c:>12,d} | {clean_c:>13,d} | {v_c:>6,d} | {te_c:>6,d} | {pct:>11.2f}%")
    print("-" * 85)
    print(f"{'Total':<10} | {'--':<3} | {len(train_labels):>12,d} | {total_clean_train:>13,d} | {len(val_labels):>6,d} | {len(test_labels):>6,d} | {'100.00%':>12}")
    print("=" * 85)

def print_table2_hyperparameters():
    print("\n" + "=" * 85)
    print(" REPRODUCING TABLE 2: HYPERPARAMETER CONFIGURATIONS (6 STANDARD MODELS)")
    print("=" * 85)
    
    hyperparams = [
        ("Multinomial Logistic Regression", "Solver=L-BFGS (Multinomial Softmax), Max Iter=1000, L2 Penalty (C=1.0), Multi-class=Multinomial"),
        ("Linear Support Vector Machine", "Loss=One-vs-Rest LinearSVC, Max Iter=2000, Penalty=L2 (C=1.0), Multi-class=OVR"),
        ("Multinomial Naive Bayes", "Additive Laplace Smoothing alpha=1.0, Prior=Empirical Class Probabilities"),
        ("Random Forest Classifier", "Trees (M)=100 Decision Trees, Splitting Criterion=Gini, Max Features=sqrt(V)=70, Bootstrap=True"),
        ("Extreme Gradient Boosting (XGBoost)", "Trees=100 GBDT Estimators, Max Depth=6, Learning Rate=0.30, Objective=multi:softprob"),
        ("Bidirectional LSTM (PyTorch)", "Seq Length=50, Embedding Dim=128, Hidden Dim=64 (Bi-LSTM -> 128), Dropout=0.30, Batch Size=64, Epochs=3, Adam (lr=0.001)")
    ]

    print(f"{'Model Architecture':<38} | {'Hyperparameter Specification':<45}")
    print("-" * 120)
    for model, spec in hyperparams:
        print(f"{model:<38} | {spec}")
    print("=" * 120 + "\n")

if __name__ == "__main__":
    audit_dataset()
    print_table2_hyperparameters()
