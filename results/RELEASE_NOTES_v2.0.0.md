# Adversarial Affective Representation Disentanglement with Adaptive Prototype Margins for Long-Tailed Text Emotion Recognition

> **Target Journal:** *IEEE Transactions on Affective Computing (TAFFC)*  
> **Camera-Ready Submission Release:** `v2.0.0`  
> **Target Branch:** `sota-affective-disentanglement`  
> **Author:** Nishant Kumar (Department of Artificial Intelligence & Machine Learning)  
> **Manuscript PDF:** Attached as `Fine_Grained_Emotion_Classification_Research_Paper.pdf` (Exactly 9 Pages)

---

## 🏆 SOTA Empirical Benchmark Highlights (Official Test Set, N = 2,000)

| Metric | ModernBERT Baseline | Proposed Disentangled Framework (Ours) | Relative Gain / Improvement |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 92.30% [91.04%, 93.43%] | **92.90% [91.69%, 93.99%]** | **+0.60%** (+12 test instances net gain) |
| **Macro-Averaged F1** | 0.8773 | **0.8866** | **+0.0093** (Statistically significant, $p=0.0428$) |
| **Surprise F1 (3.54% class)** | 0.7442 (Recall: 0.7273) | **0.7556 (Recall: 0.7727)** | **+4.54% absolute Recall expansion** |
| **Love F1 (8.06% class)** | 0.8227 (Recall: 0.7736) | **0.8447 (Recall: 0.8553)** | **+8.17% absolute Recall expansion** |
| **Weighted-F1** | 0.9222 | **0.9291** | **+0.0069** |

---

## 🔬 Core Methodological Innovations

1. **Structural Causal Formulation**:
   - Formalized via a Directed Acyclic Graph (DAG) $X \leftarrow (A, C)$, disentangling invariant affective intent $A$ from syntactic narrative context $C$.
   - Dual orthogonal latent projectors mapping ModernBERT hidden states into $z_a \in \mathbb{R}^{256}$ (pure affect) and $z_c \in \mathbb{R}^{256}$ (context/syntax).
2. **Adversarial Emotion Erasure (GRL) & Subspace Orthogonality**:
   - Minimax adversarial discriminator with Gradient Reversal Layer (GRL) ensures $z_c$ contains zero discernible emotion information.
   - Cosine penalty enforces strict orthogonality ($z_a \perp z_c$).
3. **Adaptive Prototype Margins with Generalization Bound (Proposition 1)**:
   - Dynamic class-dependent margin $\Delta_y = \alpha n_y^{-1/4}$ expands the Voronoi decision boundary for sample-scarce classes (*Surprise* and *Love*).
   - Proven via empirical Rademacher complexity and McDiarmid's concentration inequality.
4. **Dimensional Psychological Affect Grounding**:
   - Anchored to empirical 3D centroids in Russell's Circumplex & Mehrabian's PAD space via the NRC-VAD lexicon for semantic topological interpretation.
   - Final composite objective intentionally omits continuous MSE regression based on systematic ablation findings.

---

## 🔬 Certified Multi-Tier Dataset Audit

- **Clean Training Set**: **15,923 samples** (77 defective instances excised: 16 cross-split duplicates, 4 internal duplicates, 57 conflicting-label pairs across 30 unique texts).
- **Clean Validation Set**: **1,993 samples** (3 cross-split duplicates with test + 4 conflicting-label rows across 2 unique texts excised: $2,000 - 3 - 4 = 1,993$).
- **Official Test Set**: **2,000 samples** (Certified frozen benchmark partition; preserved intact).
- **Imbalance Ratio**: 9.47:1 (*Joy* $n=5,340$ vs. *Surprise* $n=564$).

---

## 💻 5-Step Reproduction Suite

```bash
# 1. Setup VAD Lexicon & Audit Dataset
python3 step1_setup_vad_data.py
python3 verify_dataset_stats.py

# 2. Benchmark Classical & Recurrent Models
python3 step2_benchmark_classical.py

# 3. Train Transformer Baseline & Proposed Framework
python3 step3_train_transformer.py ModernBERT_Baseline
python3 step3_train_transformer.py Proposed_Framework_Full

# 4. Verify Statistical Significance & Instance Predictions
python3 step4_verify_statistics.py

# 5. Generate Publication-Quality Vector PDF Figures
python3 step5_generate_figures.py
```

---

## 📦 Attached Release Asset

- **`Fine_Grained_Emotion_Classification_Research_Paper.pdf`**: Complete 9-page camera-ready IEEE Transactions on Affective Computing manuscript with embedded vector figures, formal proofs, and unified bibliography on Page 9.
