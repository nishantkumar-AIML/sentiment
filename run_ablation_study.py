#!/usr/bin/env python3
"""
N-gram and Preprocessing Pipeline Ablation Experiment
=====================================================
Paper: "Fine-Grained Emotion Classification in Social Media Text:
        A Comparative Study of Classical Ensembles and Deep Sequence Models Under Class Imbalance"

This script executes the exact 5-step controlled ablation study (Table 5 in paper):
  P0: Raw text + Unigrams (5k)
  P1: Lowercase + Punctuation Stripping
  P2: P1 + Standard Stopwords (Blindly stripping negations)
  P3: P1 + Negation-Preserved Stopwords (Baseline)
  P4: P3 + Unigram + Bigram (1, 2)
"""

import os
import csv
import re
import json
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

SEED = 42

# 1. Load Clean Dataset
def load_data():
    def read_csv(path):
        texts, labels = [], []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    texts.append(row[0])
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

# 2. Stopword Lists
STANDARD_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
    "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and",
    "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now"
}

NEGATION_WORDS = {"not", "no", "never", "nor", "neither", "barely", "hardly", "scarcely", "without", "don", "didn", "wasn", "weren", "haven", "hasn", "hadn", "couldn", "shouldn", "wouldn", "mustn"}
NEGATION_PRESERVED_STOPWORDS = STANDARD_STOPWORDS - NEGATION_WORDS

# 3. Preprocessing Functions
def preprocess_p0(text):
    return text

def preprocess_p1(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_p2(text):
    cleaned = preprocess_p1(text)
    words = [w for w in cleaned.split() if w not in STANDARD_STOPWORDS]
    return " ".join(words)

def preprocess_p3(text):
    cleaned = preprocess_p1(text)
    words = [w for w in cleaned.split() if w not in NEGATION_PRESERVED_STOPWORDS]
    return " ".join(words)

def main():
    print("=" * 88, flush=True)
    print(" EXECUTING N-GRAM & PREPROCESSING PIPELINE ABLATION STUDY (RANDOM FOREST 100 TREES)", flush=True)
    print("=" * 88, flush=True)
    train_x, train_y, test_x, test_y = load_data()

    experiments = [
        ("P0", "Raw text + Unigrams (5k)", preprocess_p0, (1, 1), 5000),
        ("P1", "Lowercase + Punctuation Stripping", preprocess_p1, (1, 1), 5000),
        ("P2", "P1 + Blind Stopwords (Stripped Negations)", preprocess_p2, (1, 1), 5000),
        ("P3 (Baseline)", "P1 + Negation-Preserved Stopwords", preprocess_p3, (1, 1), 5000),
        ("P4", "P3 + Unigram + Bigram (1, 2)", preprocess_p3, (1, 2), 5000),
    ]

    ablation_results = {}
    print(f"{'Config':<15} | {'Preprocessing Protocol':<45} | {'Accuracy':<10} | {'Macro-F1':<10}", flush=True)
    print("-" * 88, flush=True)

    for cfg, desc, fn, ngrams, max_v in experiments:
        t0 = time.time()
        train_proc = [fn(t) for t in train_x]
        test_proc  = [fn(t) for t in test_x]

        vec = TfidfVectorizer(max_features=max_v, ngram_range=ngrams, sublinear_tf=True, norm='l2')
        X_tr = vec.fit_transform(train_proc)
        X_te = vec.transform(test_proc)

        clf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)
        clf.fit(X_tr, train_y)
        preds = clf.predict(X_te)

        acc = accuracy_score(test_y, preds)
        _, _, mf1, _ = precision_recall_fscore_support(test_y, preds, average='macro', zero_division=0)
        _, _, wf1, _ = precision_recall_fscore_support(test_y, preds, average='weighted', zero_division=0)

        ablation_results[cfg] = {
            "config": cfg,
            "description": desc,
            "accuracy": round(acc, 4),
            "accuracy_pct": f"{acc*100:.2f}%",
            "macro_f1": round(mf1, 4),
            "weighted_f1": round(wf1, 4),
            "time_sec": round(time.time() - t0, 2)
        }

        print(f"{cfg:<15} | {desc:<45} | {acc:>8.2%}  | {mf1:>8.4f}", flush=True)

    print("-" * 88, flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print("[*] Saved ablation results to results/ablation_results.json", flush=True)

if __name__ == "__main__":
    main()
