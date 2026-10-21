"""
PyTorch Implementation of VAD-Anchored Causal Affective Disentanglement
with Adaptive Repulsion Prototype Margins
========================================================================
Author: Nishant Kumar
Target Journal: IEEE Transactions on Affective Computing (TAFFC)

This module implements:
1. AttentionPooling: Length-invariant attention-weighted token pooling.
2. GradientReversal: GRL layer for adversarial domain/emotion invariance.
3. VADRepulsionPrototypeLoss: Novel loss function coupling sample scarcity (n_y^{-1/4})
   with continuous 3D psychological VAD proximity exp(-gamma * ||v_y - v_k||_2^2).
4. CausalAffectiveDisentangler: Complete end-to-end architecture splitting contextual
   representations into pure affective (z_a) and context/noise (z_c) subspaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class AttentionPooling(nn.Module):
    """
    Computes an attention-weighted combination of sequence token representations.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: [B, L, D]
        scores = self.attention(hidden_states) # [B, L, 1]
        if attention_mask is not None:
            mask = (1.0 - attention_mask.unsqueeze(-1).float()) * -10000.0
            scores = scores + mask
        weights = F.softmax(scores, dim=1) # [B, L, 1]
        pooled = torch.sum(hidden_states * weights, dim=1) # [B, D]
        return pooled

class VADRepulsionPrototypeLoss(nn.Module):
    """
    VAD-Guided Dynamic Repulsion Prototype Margin Loss
    Combines inverse class frequency n_y^{-1/4} with continuous 3D psychological VAD
    proximity exp(-gamma * ||v_y - v_k||_2^2).
    """
    def __init__(self, class_counts, class_vad_anchors, alpha=0.5, gamma=2.0, tau=0.1):
        super().__init__()
        self.num_classes = len(class_counts)
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma

        # Extract sample counts n_c
        counts = torch.tensor(
            [class_counts[c] if c in class_counts else class_counts[str(c)] for c in range(self.num_classes)],
            dtype=torch.float32
        )
        # Sample scarcity weights n_y^{-1/4}
        scarcity_weights = counts ** -0.25
        scarcity_weights = scarcity_weights / scarcity_weights.mean()

        # Extract 3D psychological VAD centroids: [K, 3]
        v = torch.tensor(
            [class_vad_anchors[c] if c in class_vad_anchors else class_vad_anchors[str(c)] for c in range(self.num_classes)],
            dtype=torch.float32
        )
        v_diff = v.unsqueeze(1) - v.unsqueeze(0) # [K, K, 3]
        dist_sq = torch.sum(v_diff ** 2, dim=-1) # [K, K]

        # Psychological overlap repulsion factor
        vad_repulsion = torch.exp(-self.gamma * dist_sq) # [K, K]

        # Pairwise angular repulsion margin matrix
        delta_matrix = self.alpha * (scarcity_weights.unsqueeze(1) * vad_repulsion)
        delta_matrix.fill_diagonal_(0.0)

        self.register_buffer("delta_matrix", delta_matrix)
        self.register_buffer("scarcity_weights", scarcity_weights)

    def forward(self, z_a, prototypes, labels):
        # Unit-norm projection onto hypersphere
        z_norm = F.normalize(z_a, p=2, dim=-1)
        p_norm = F.normalize(prototypes, p=2, dim=-1)

        # Cosine similarity matrix: s(z_a, P_k)
        cosine_sim = torch.matmul(z_norm, p_norm.t()) # [B, K]

        # Retrieve adaptive repulsion margins for current batch
        batch_deltas = self.delta_matrix[labels] # [B, K]

        # Enforce repulsive margin on competing classes (k != y)
        margin_sim = cosine_sim + batch_deltas
        logits = margin_sim / self.tau

        loss = F.cross_entropy(logits, labels)
        return loss, cosine_sim

class CausalAffectiveDisentangler(nn.Module):
    """
    Unified Architecture: Modern Transformer Backbone + Dual Causal Projectors + VAD Regressor + GRL Eraser
    """
    def __init__(self, backbone, hidden_dim, proj_dim=256, num_classes=6,
                 class_counts=None, class_vad_anchors=None,
                 alpha=0.5, gamma=2.0, tau=0.1,
                 lambda_vad=1.0, lambda_ortho=0.5, lambda_adv=0.2):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.num_classes = num_classes

        self.lambda_vad = lambda_vad
        self.lambda_ortho = lambda_ortho
        self.lambda_adv = lambda_adv

        # Length-invariant attention pooler
        self.pooler = AttentionPooling(hidden_dim)

        # 1. Affective Subspace Projector (z_a)
        self.proj_a = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim)
        )

        # 2. Context/Noise Subspace Projector (z_c)
        self.proj_c = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim)
        )

        # 3. Psychological VAD Regressor phi_VAD(z_a) -> [V, A, D]
        self.phi_vad = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Linear(proj_dim // 2, 3),
            nn.Tanh() # Continuous VAD scaled to [-1, 1]
        )

        # 4. Adversarial Emotion Erasure Head f_adv(z_c) with GRL
        self.grl = GradientReversal(alpha=1.0)
        self.f_adv = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(proj_dim // 2, num_classes)
        )

        # 5. Unit-norm learnable class prototypes
        self.prototypes = nn.Parameter(torch.randn(num_classes, proj_dim))
        nn.init.orthogonal_(self.prototypes)

        # 6. VAD-Guided Prototype Loss Module
        self.proto_loss_fn = VADRepulsionPrototypeLoss(
            class_counts=class_counts,
            class_vad_anchors=class_vad_anchors,
            alpha=alpha,
            gamma=gamma,
            tau=tau
        )

    def forward(self, input_ids, attention_mask=None, labels=None, vad_targets=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state # [B, L, D]

        h = self.pooler(hidden_states, attention_mask) # [B, D]
        z_a = self.proj_a(h) # Affective representation
        z_c = self.proj_c(h) # Context/Noise representation

        if labels is not None:
            # Prototype classification loss with dynamic repulsion
            loss_proto, cosine_sim = self.proto_loss_fn(z_a, self.prototypes, labels)

            # Subspace orthogonality loss
            z_a_norm = F.normalize(z_a, p=2, dim=-1)
            z_c_norm = F.normalize(z_c, p=2, dim=-1)
            ortho_sim = torch.sum(z_a_norm * z_c_norm, dim=-1)
            loss_ortho = torch.mean(torch.abs(ortho_sim))

            # Psychological VAD alignment loss
            loss_vad = torch.tensor(0.0, device=z_a.device)
            if vad_targets is not None:
                pred_vad = self.phi_vad(z_a)
                loss_vad = F.mse_loss(pred_vad, vad_targets)

            # Adversarial emotion erasure loss on context subspace
            z_c_reversed = self.grl(z_c)
            adv_logits = self.f_adv(z_c_reversed)
            loss_adv = F.cross_entropy(adv_logits, labels)

            total_loss = (loss_proto +
                          self.lambda_vad * loss_vad +
                          self.lambda_ortho * loss_ortho +
                          self.lambda_adv * loss_adv)

            return {
                "loss": total_loss,
                "loss_proto": loss_proto,
                "loss_vad": loss_vad,
                "loss_ortho": loss_ortho,
                "loss_adv": loss_adv,
                "logits": cosine_sim,
                "z_a": z_a,
                "z_c": z_c
            }
        else:
            z_norm = F.normalize(z_a, p=2, dim=-1)
            p_norm = F.normalize(self.prototypes, p=2, dim=-1)
            cosine_sim = torch.matmul(z_norm, p_norm.t())
            pred_vad = self.phi_vad(z_a)
            return {
                "logits": cosine_sim,
                "pred_vad": pred_vad,
                "z_a": z_a,
                "z_c": z_c
            }
