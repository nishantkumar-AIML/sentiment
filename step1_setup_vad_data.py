#!/usr/bin/env python3
"""
Step 1: Psychological VAD Lexicon Grounding & Dataset Anchor Setup
==================================================================
Grounds discrete emotion recognition in Russell's Circumplex & Mehrabian's PAD/VAD model
via the NRC Valence, Arousal, and Dominance (NRC-VAD) Lexicon (Mohammad, ACL 2018).

Author: Nishant Kumar
Framework: VAD-Anchored Causal Affective Disentanglement with Adaptive Repulsion Margins
"""

import os
import csv
import re
import json
import urllib.request
import zipfile
import io
import numpy as np

EMOTION_NAMES = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]

def clean_tokens(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    return re.findall(r'[a-zA-Z]+', text)

def download_and_extract_nrc_vad(target_dir="dataset/nrc_vad"):
    lexicon_txt = os.path.join(target_dir, "NRC-VAD-Lexicon.txt")
    if os.path.exists(lexicon_txt):
        print(f"[*] Found existing NRC-VAD Lexicon at {lexicon_txt}")
        return lexicon_txt

    os.makedirs(target_dir, exist_ok=True)
    url = "http://saifmohammad.com/WebDocs/VAD/NRC-VAD-Lexicon-Aug2018Release.zip"
    print(f"[*] Downloading official NRC-VAD Lexicon from: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for member in z.namelist():
            if member.endswith("NRC-VAD-Lexicon.txt"):
                data = z.read(member)
                with open(lexicon_txt, "wb") as f:
                    f.write(data)
                print(f"[+] Successfully extracted {lexicon_txt} ({len(data):,} bytes)")
                return lexicon_txt
    raise FileNotFoundError("NRC-VAD-Lexicon.txt not found inside downloaded archive.")

def load_nrc_vad_lexicon(lexicon_path):
    lexicon = {}
    with open(lexicon_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                word = parts[0].lower()
                # Continuous rescale from [0, 1] to [-1, 1]
                v = 2.0 * float(parts[1]) - 1.0
                a = 2.0 * float(parts[2]) - 1.0
                d = 2.0 * float(parts[3]) - 1.0
                lexicon[word] = np.array([v, a, d], dtype=np.float32)
    print(f"[*] Loaded {len(lexicon):,} lexical VAD vectors from {lexicon_path}")
    return lexicon

def text_to_vad(tokens, lexicon):
    matched = [lexicon[w] for w in tokens if w in lexicon]
    if matched:
        return np.mean(matched, axis=0)
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)

def main():
    lexicon_path = download_and_extract_nrc_vad()
    lexicon = load_nrc_vad_lexicon(lexicon_path)

    def read_csv(path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    data.append((row[0], int(row[1])))
        return data

    raw_train = read_csv("dataset/training.csv")
    raw_val = read_csv("dataset/validation.csv")
    test_data = read_csv("dataset/test.csv")

    test_texts = set(t for t, _ in test_data)

    # 1. Clean Validation Split:
    # Exclude 3 direct cross-split overlaps with test set
    # Exclude internal duplicate conflicting-label instances (2 texts with 4 contradictory labels)
    val_t_to_l = {}
    for t, l in raw_val:
        val_t_to_l.setdefault(t, set()).add(l)
    val_conflicts = {t for t, ls in val_t_to_l.items() if len(ls) > 1}

    clean_val = [
        (t, y) for t, y in raw_val
        if t not in test_texts and t not in val_conflicts
    ]
    print(f"[*] Raw Validation: {len(raw_val):,} -> Clean Validation: {len(clean_val):,} (7 conflicting/leakage samples excised)")

    # 2. Clean Training Split:
    # Exclude cross-split duplicates with evaluation partitions (val + test)
    # Exclude conflicting-label instances within training set
    # Deduplicate identical rows (keep first)
    all_val_texts = set(t for t, _ in raw_val)
    exclude_eval = test_texts | all_val_texts

    train_t_to_l = {}
    for t, l in raw_train:
        train_t_to_l.setdefault(t, set()).add(l)
    train_conflicts = {t for t, ls in train_t_to_l.items() if len(ls) > 1}

    clean_train = []
    seen_train = set()
    for t, y in raw_train:
        if t in exclude_eval:
            continue
        if t in train_conflicts:
            continue
        if t in seen_train:
            continue
        seen_train.add(t)
        clean_train.append((t, y))

    print(f"[*] Raw Training:   {len(raw_train):,} -> Clean Training:   {len(clean_train):,} (77 conflicting/leakage/duplicate samples excised)")
    print(f"[*] Frozen Test:    {len(test_data):,}")

    class_vad_sums = {c: np.zeros(3, dtype=np.float64) for c in range(6)}
    class_counts = {c: 0 for c in range(6)}

    processed_train = []
    for text, label in clean_train:
        tokens = clean_tokens(text)
        vad = text_to_vad(tokens, lexicon)
        class_vad_sums[label] += vad
        class_counts[label] += 1
        processed_train.append({
            "text": text,
            "label": label,
            "vad": vad.tolist()
        })

    class_vad_anchors = {}
    print("\n" + "=" * 80)
    print(" COMPUTED 3D PSYCHOLOGICAL VAD CLASS ANCHORS (CENTROIDS)")
    print("=" * 80)
    print(f"{'Class ID':<8} | {'Emotion':<10} | {'Count (n)':<10} | {'Valence (V)':<12} | {'Arousal (A)':<12} | {'Dominance (D)':<12}")
    print("-" * 80)
    for c in range(6):
        centroid = (class_vad_sums[c] / class_counts[c]).astype(np.float32)
        class_vad_anchors[c] = centroid.tolist()
        print(f"{c:<8} | {EMOTION_NAMES[c]:<10} | {class_counts[c]:<10} | {centroid[0]:<12.4f} | {centroid[1]:<12.4f} | {centroid[2]:<12.4f}")
    print("-" * 80)

    # Process clean validation and test splits
    processed_val = [{"text": t, "label": y, "vad": text_to_vad(clean_tokens(t), lexicon).tolist()} for t, y in clean_val]
    processed_test = [{"text": t, "label": y, "vad": text_to_vad(clean_tokens(t), lexicon).tolist()} for t, y in test_data]

    # Compute audit summary
    import collections
    train_counts_dict = collections.Counter(y for _, y in clean_train)
    val_counts_dict = collections.Counter(y for _, y in clean_val)
    test_counts_dict = collections.Counter(y for _, y in test_data)

    audit_summary = {
        "classes": EMOTION_NAMES,
        "clean_train_counts": [train_counts_dict[c] for c in range(6)],
        "clean_val_counts": [val_counts_dict[c] for c in range(6)],
        "test_counts": [test_counts_dict[c] for c in range(6)],
        "total_train_orig": len(raw_train),
        "total_train_clean": len(clean_train),
        "total_val_orig": len(raw_val),
        "total_val_clean": len(clean_val),
        "total_test": len(test_data),
        "train_removed_samples": len(raw_train) - len(clean_train),
        "val_removed_samples": len(raw_val) - len(clean_val),
        "imbalance_ratio": round(train_counts_dict[1] / train_counts_dict[5], 2)
    }

    os.makedirs("results", exist_ok=True)
    with open("results/dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    with open("results/class_vad_anchors.json", "w", encoding="utf-8") as f:
        json.dump({
            "emotion_names": EMOTION_NAMES,
            "class_counts": class_counts,
            "class_vad_anchors": class_vad_anchors
        }, f, indent=2)

    with open("dataset/vad_augmented_splits.json", "w", encoding="utf-8") as f:
        json.dump({
            "train": processed_train,
            "val": processed_val,
            "test": processed_test
        }, f)

    print("\n[+] Saved dataset audit to results/dataset_audit.json")
    print("[+] Saved class VAD anchors to results/class_vad_anchors.json")
    print("[+] Saved VAD-augmented splits to dataset/vad_augmented_splits.json")

if __name__ == "__main__":
    main()
