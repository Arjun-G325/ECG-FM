# """
# models/backbone.py — ECG Foundation Model backbone

# Architecture overview
# ─────────────────────
#   Input (B, 12, 5000)
#       │
#   CNNStem          →  (B, d_model, 250)   temporal downsampling
#       │  permute
#   TransformerEncoder → (B, 250, d_model)  global context
#       │
#   TransformerDecoder → (B, n_query, d_model)  cross-attention summary
#       │  mean
#   (B, d_model)  ──►  ProjectionHead  →  (B, proj_dim)  [pre-training]
#                   └►  get_representation()              [fine-tuning]
# """

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # ── CNN Stem ───────────────────────────────────────────────────────────────────

# class CNNStem(nn.Module):
#     """
#     Three strided Conv1d blocks that compress
#     (B, n_leads, 5000) → (B, d_model, 250).

#     Stride schedule: ×2, ×2, ×5  →  total ×20
#     """

#     def __init__(self, n_leads: int = 12, d_model: int = 256):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(n_leads,  64,      kernel_size=15, stride=2, padding=7),
#             nn.BatchNorm1d(64),  nn.GELU(),
#             nn.Conv1d(64,       128,     kernel_size=9,  stride=2, padding=4),
#             nn.BatchNorm1d(128), nn.GELU(),
#             nn.Conv1d(128,      d_model, kernel_size=5,  stride=5, padding=2),
#             nn.BatchNorm1d(d_model), nn.GELU(),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """x: (B, n_leads, T)  →  (B, d_model, T/20)"""
#         return self.net(x)


# # ── Positional Encoding ────────────────────────────────────────────────────────

# class SinusoidalPE(nn.Module):
#     """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

#     def __init__(self, d_model: int, max_len: int = 512):
#         super().__init__()
#         pe  = torch.zeros(max_len, d_model)
#         pos = torch.arange(0, max_len).unsqueeze(1).float()
#         div = torch.exp(
#             torch.arange(0, d_model, 2).float() * -(np.log(10_000.0) / d_model)
#         )
#         pe[:, 0::2] = torch.sin(pos * div)
#         pe[:, 1::2] = torch.cos(pos * div)
#         self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """x: (B, T, d_model)"""
#         return x + self.pe[:, : x.size(1)]


# # ── Transformer Encoder ────────────────────────────────────────────────────────

# class TransformerEncoder(nn.Module):
#     def __init__(
#         self,
#         d_model:    int = 256,
#         nhead:      int = 8,
#         num_layers: int = 4,
#         dim_ff:     int = 512,
#         dropout:    float = 0.1,
#     ):
#         super().__init__()
#         layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_ff,
#             dropout=dropout,
#             batch_first=True,
#             norm_first=True,       # pre-LN for training stability
#         )
#         self.encoder = nn.TransformerEncoder(
#             layer, num_layers=num_layers, enable_nested_tensor=False
#         )
#         self.pe = SinusoidalPE(d_model)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """x: (B, T, d_model)  →  (B, T, d_model)"""
#         return self.encoder(self.pe(x))


# # ── Transformer Decoder ────────────────────────────────────────────────────────

# class TransformerDecoder(nn.Module):
#     """
#     Learnable query tokens attend over encoder memory via cross-attention.
#     Produces a compact fixed-length summary regardless of input length.
#     """

#     def __init__(
#         self,
#         d_model:        int = 256,
#         nhead:          int = 8,
#         num_layers:     int = 2,
#         dim_ff:         int = 512,
#         dropout:        float = 0.1,
#         n_query_tokens: int = 16,
#     ):
#         super().__init__()
#         self.query_tokens = nn.Parameter(torch.randn(1, n_query_tokens, d_model))
#         layer = nn.TransformerDecoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_ff,
#             dropout=dropout,
#             batch_first=True,
#             norm_first=True,
#         )
#         self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
#         self.pe = SinusoidalPE(d_model)

#     def forward(self, memory: torch.Tensor) -> torch.Tensor:
#         """
#         memory: (B, T, d_model)  →  (B, n_query_tokens, d_model)
#         """
#         B = memory.size(0)
#         q = self.pe(self.query_tokens.expand(B, -1, -1))
#         return self.decoder(q, memory)


# # ── Projection Head ────────────────────────────────────────────────────────────

# class ProjectionHead(nn.Module):
#     """
#     Two-layer MLP + L2 normalisation.
#     Used only during contrastive pre-training; discarded at fine-tuning time.
#     """

#     def __init__(self, d_model: int = 256, proj_dim: int = 128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.GELU(),
#             nn.Linear(d_model, proj_dim),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return F.normalize(self.net(x), dim=-1)


# # ── ECG Foundation Model ───────────────────────────────────────────────────────

# class ECGFoundationModel(nn.Module):
#     """
#     Full backbone used during pre-training.

#     forward()              → (B, proj_dim)   normalised projections for NT-Xent
#     get_representation()   → (B, d_model)    feature vector for downstream tasks
#     encode()               → (B, T', d_model) full encoder output
#     """

#     def __init__(
#         self,
#         n_leads:        int   = 12,
#         d_model:        int   = 256,
#         nhead:          int   = 8,
#         enc_layers:     int   = 4,
#         dec_layers:     int   = 2,
#         dim_ff:         int   = 512,
#         dropout:        float = 0.1,
#         n_query_tokens: int   = 16,
#         proj_dim:       int   = 128,
#     ):
#         super().__init__()
#         self.stem    = CNNStem(n_leads, d_model)
#         self.encoder = TransformerEncoder(d_model, nhead, enc_layers, dim_ff, dropout)
#         self.decoder = TransformerDecoder(d_model, nhead, dec_layers, dim_ff, dropout, n_query_tokens)
#         self.proj    = ProjectionHead(d_model, proj_dim)
#         self.d_model = d_model

#     def encode(self, x: torch.Tensor) -> torch.Tensor:
#         """(B, n_leads, T) → (B, T', d_model)"""
#         return self.encoder(self.stem(x).permute(0, 2, 1))

#     def get_representation(self, x: torch.Tensor) -> torch.Tensor:
#         """(B, n_leads, T) → (B, d_model)  mean-pooled over query tokens"""
#         return self.decoder(self.encode(x)).mean(dim=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """(B, n_leads, T) → (B, proj_dim)  L2-normalised for contrastive loss"""
#         return self.proj(self.get_representation(x))
"""
models/backbone.py — ECG Foundation Model backbone  (v2)

What changed from v1 and why
─────────────────────────────
  1. LeadPatchEmbedding   replaces CNNStem
       • Each lead is tokenised independently via a shared linear patchifier.
       • A learnable lead-ID embedding is added per lead so the model knows
         *which* anatomical projection each token came from.
       • Any subset of leads can be passed at inference — robust to missing leads.

  2. d_model = 512  (was 256)
       • More capacity for simultaneous morphology + rhythm + inter-lead features.
       • 8 heads × 64-dim subspaces vs the previous 8 × 32.

  3. Learnable positional encoding  (was fixed sinusoidal)
       • Generalises to variable signal lengths and sampling rates.

  4. MAE-style masked pre-training  (replaces NT-Xent contrastive loss)
       • No augmentation pairs required — supervision comes from the signal itself.
       • Three masking strategies: temporal, lead, spatiotemporal (2-D blocks).
       • High mask ratio (default 75 %) forces global reasoning.
       • Lightweight decoder reconstructs raw patch values; discarded at fine-tune.

  5. ECGFinetuneModel  — thin wrapper that loads pre-trained weights and attaches
       a task-specific head for downstream supervised learning.

Architecture (pre-training)
────────────────────────────
  Input (B, n_leads, T)
      │
  LeadPatchEmbedding   →  (B, n_leads × n_patches, d_model)
      │  mask → keep visible tokens only
  TransformerEncoder   →  (B, n_vis, d_model)
      │  scatter back + fill masked positions with [MASK] token
  MAEDecoder           →  (B, n_leads × n_patches, patch_size)
      │
  MSE loss on masked positions only

Architecture (fine-tuning)
───────────────────────────
  Input (B, n_leads, T)   ← any lead subset
      │
  LeadPatchEmbedding  →  (B, n_leads × n_patches, d_model)   [no masking]
  TransformerEncoder  →  (B, n_leads × n_patches, d_model)
      │  mean pool
  (B, d_model)  →  task head (linear / MLP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ── Constants ──────────────────────────────────────────────────────────────────

MAX_LEADS = 12
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


# ── Lead-Aware Patch Embedding ─────────────────────────────────────────────────

class LeadPatchEmbedding(nn.Module):
    """
    Tokenise each ECG lead independently into non-overlapping patches,
    then sum three embeddings:
        token embedding   — linear projection of raw patch samples
        temporal PE       — learnable position per patch index
        lead-ID embedding — learnable identity per lead

    The patch projector is shared across all leads, so the model handles
    any subset of leads at inference without architectural changes.

    Input : (B, n_leads, T)
    Output: (B, n_leads * n_patches, d_model)
    """

    def __init__(
        self,
        patch_size:  int   = 50,    # samples per patch; 50 @ 500 Hz = 100 ms
        d_model:     int   = 512,
        max_leads:   int   = MAX_LEADS,
        max_patches: int   = 256,   # upper bound on temporal patches per lead
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model    = d_model

        # Shared linear patchifier — equivalent to Conv1d(1, d_model, patch_size, stride=patch_size)
        # but explicit: input is (…, patch_size), output is (…, d_model)
        self.patch_proj = nn.Linear(patch_size, d_model, bias=False)

        # Learnable temporal positional encoding (one vector per patch index)
        self.temporal_pe = nn.Embedding(max_patches, d_model)

        # Learnable per-lead identity embedding
        # Index 0-11 correspond to I, II, III, aVR, aVL, aVF, V1-V6
        self.lead_id_emb = nn.Embedding(max_leads, d_model)

        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:        torch.Tensor,             # (B, n_leads, T)
        lead_ids: Optional[torch.Tensor],   # (n_leads,) int indices; None → 0..n_leads-1
    ) -> torch.Tensor:                      # (B, n_leads * n_patches, d_model)

        B, L, T = x.shape
        P        = self.patch_size
        n_patches = T // P

        # Trim to a multiple of patch_size
        x = x[:, :, : n_patches * P]                          # (B, L, n_patches * P)
        x = x.reshape(B, L, n_patches, P)                     # (B, L, n_p, P)

        # Linear patch projection
        x = self.patch_proj(x)                                 # (B, L, n_p, d_model)

        # Add temporal positional encoding (broadcast over batch and leads)
        t_idx = torch.arange(n_patches, device=x.device)      # (n_p,)
        x = x + self.temporal_pe(t_idx)[None, None]            # (1, 1, n_p, d_model)

        # Add lead identity encoding (broadcast over batch and patches)
        if lead_ids is None:
            lead_ids = torch.arange(L, device=x.device)       # (L,)
        x = x + self.lead_id_emb(lead_ids)[None, :, None]     # (1, L, 1, d_model)

        # Flatten leads × patches into a single token sequence
        x = x.reshape(B, L * n_patches, self.d_model)         # (B, L*n_p, d_model)
        return self.dropout(self.norm(x))


# ── Masking Strategies ─────────────────────────────────────────────────────────

class MaskingStrategy:
    """
    Three masking modes for MAE-style pre-training.
    All return a boolean tensor of shape (B, n_leads, n_patches)
    where True marks tokens that should be predicted (masked).
    """

    @staticmethod
    def temporal(
        B: int, n_leads: int, n_patches: int,
        ratio: float = 0.75,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Randomly mask individual patches.
        Each sample gets exactly `ratio` fraction of its tokens masked.
        This is the default MAE strategy and gives the model freedom to
        develop global context because nearby patches may be visible or not.
        """
        noise  = torch.rand(B, n_leads * n_patches, device=device)
        thresh = torch.quantile(noise, ratio, dim=1, keepdim=True)
        flat   = noise >= thresh                                   # (B, N)  True = masked
        return flat.reshape(B, n_leads, n_patches)

    @staticmethod
    def lead(
        B: int, n_leads: int, n_patches: int,
        n_masked_leads: int = 4,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Mask entire leads.
        Forces the model to reconstruct one lead from the others —
        directly learning inter-lead geometric dependencies.
        E.g. masking lead II and asking the model to reconstruct it
        from I and III tests whether it learned Einthoven's triangle.
        """
        mask = torch.zeros(B, n_leads, n_patches, dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(n_leads, device=device)[:n_masked_leads]
            mask[b, idx] = True
        return mask

    @staticmethod
    def spatiotemporal(
        B: int, n_leads: int, n_patches: int,
        lead_ratio: float = 0.4,
        time_ratio: float = 0.6,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Mask contiguous 2-D rectangular blocks in the (lead × time) grid.
        Harder than random masking — the model cannot use immediate
        neighbours in either the temporal or lead dimension.
        """
        mask = torch.zeros(B, n_leads, n_patches, dtype=torch.bool, device=device)
        n_l = max(1, int(n_leads   * lead_ratio))
        n_t = max(1, int(n_patches * time_ratio))
        for b in range(B):
            l0 = torch.randint(0, max(1, n_leads   - n_l + 1), ()).item()
            t0 = torch.randint(0, max(1, n_patches - n_t + 1), ()).item()
            mask[b, l0 : l0 + n_l, t0 : t0 + n_t] = True
        return mask


# ── Transformer Encoder ────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """
    Standard pre-LN transformer encoder.
    During pre-training it receives only the *visible* (unmasked) tokens,
    which is cheaper than processing the full sequence and forces the model
    to be robust to missing context.
    """

    def __init__(
        self,
        d_model:    int   = 512,
        nhead:      int   = 8,
        num_layers: int   = 8,
        dim_ff:     int   = 2048,
        dropout:    float = 0.1,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, norm_first=True,   # pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        return self.encoder(x)


# ── MAE Decoder ────────────────────────────────────────────────────────────────

class MAEDecoder(nn.Module):
    """
    Lightweight decoder used only during pre-training.

    Receives the full token sequence — encoder outputs at visible positions
    and learnable [MASK] tokens at masked positions — and reconstructs the
    raw signal samples for every patch.

    Kept intentionally shallow (fewer layers, smaller dim than encoder) so
    that reconstruction pressure falls on the encoder representations, not
    the decoder. This is the MAE design principle: a powerful encoder +
    weak decoder forces rich latent representations.

    Discarded entirely at fine-tuning time.
    """

    def __init__(
        self,
        enc_dim:    int   = 512,
        dec_dim:    int   = 256,
        nhead:      int   = 8,
        num_layers: int   = 4,
        dim_ff:     int   = 1024,
        dropout:    float = 0.1,
        patch_size: int   = 50,
    ):
        super().__init__()
        # Project from encoder dimension to (narrower) decoder dimension
        self.enc_to_dec = nn.Linear(enc_dim, dec_dim, bias=True)

        layer = nn.TransformerEncoderLayer(
            d_model=dec_dim, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(dec_dim)
        # Final linear head: each token predicts its raw patch samples
        self.head = nn.Linear(dec_dim, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, enc_dim)  full sequence (enc outputs + mask tokens)
        → (B, N, patch_size)
        """
        x = self.enc_to_dec(x)        # (B, N, dec_dim)
        x = self.norm(self.decoder(x))
        return self.head(x)            # (B, N, patch_size)


# ── ECG Foundation Model (pre-training) ───────────────────────────────────────

class ECGFoundationModel(nn.Module):
    """
    Full model used during masked signal modelling pre-training.

    Key methods
    ───────────
      forward()              → (loss, pred, mask)   MAE pre-training step
      encode()               → (B, N, d_model)      full token sequence, no masking
      get_representation()   → (B, d_model)          mean-pooled; used at fine-tuning
    """

    def __init__(
        self,
        patch_size:    int   = 50,          # 100 ms per patch at 500 Hz
        d_model:       int   = 512,
        nhead:         int   = 8,
        enc_layers:    int   = 8,
        dec_layers:    int   = 4,
        dim_ff:        int   = 2048,
        dropout:       float = 0.1,
        max_leads:     int   = MAX_LEADS,
        max_patches:   int   = 256,
        dec_dim:       int   = 256,
        mask_ratio:    float = 0.75,
        mask_strategy: str   = "temporal",  # "temporal" | "lead" | "spatiotemporal"
    ):
        super().__init__()
        self.patch_size    = patch_size
        self.mask_ratio    = mask_ratio
        self.mask_strategy = mask_strategy
        self.d_model       = d_model

        self.embed   = LeadPatchEmbedding(patch_size, d_model, max_leads, max_patches, dropout)
        self.encoder = TransformerEncoder(d_model, nhead, enc_layers, dim_ff, dropout)
        self.decoder = MAEDecoder(d_model, dec_dim, nhead, dec_layers, dim_ff // 2, dropout, patch_size)

        # Learnable [MASK] token — inserted at masked positions before the decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    # ── internal helpers ───────────────────────────────────────────────────────

    def _make_mask(
        self, B: int, L: int, n_patches: int, device: torch.device
    ) -> torch.Tensor:
        if self.mask_strategy == "lead":
            n_masked = max(1, int(L * self.mask_ratio))
            return MaskingStrategy.lead(B, L, n_patches, n_masked, device)
        elif self.mask_strategy == "spatiotemporal":
            return MaskingStrategy.spatiotemporal(B, L, n_patches, device=device)
        else:
            return MaskingStrategy.temporal(B, L, n_patches, self.mask_ratio, device)

    # ── public API ─────────────────────────────────────────────────────────────

    def encode(
        self,
        x:        torch.Tensor,
        lead_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        (B, n_leads, T)  →  (B, n_leads * n_patches, d_model)
        No masking; used at fine-tuning time.
        `lead_ids` allows passing an arbitrary lead subset, e.g.
            lead_ids = torch.tensor([0, 1, 6])  for leads I, II, V1
        """
        return self.encoder(self.embed(x, lead_ids))

    def get_representation(
        self,
        x:        torch.Tensor,
        lead_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        (B, n_leads, T)  →  (B, d_model)
        Global mean pool over all tokens.  Drop-in feature extractor for
        downstream classification / regression heads.
        """
        return self.encode(x, lead_ids).mean(dim=1)

    def forward(
        self,
        x:        torch.Tensor,                   # (B, n_leads, T)
        lead_ids: Optional[torch.Tensor] = None,  # (n_leads,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MAE pre-training forward pass.

        Returns
        ───────
        loss  : scalar  MSE computed only on masked patches
        pred  : (B, N, patch_size)   full reconstruction (all positions)
        mask  : (B, n_leads, n_patches)  bool, True = position was masked
        """
        B, L, T    = x.shape
        P          = self.patch_size
        n_patches  = T // P
        N          = L * n_patches                   # total tokens per sample

        # ── Step 1: embed all tokens ──────────────────────────────────────────
        tokens = self.embed(x, lead_ids)             # (B, N, d_model)

        # ── Step 2: generate mask ─────────────────────────────────────────────
        mask      = self._make_mask(B, L, n_patches, x.device)   # (B, L, n_p) bool
        flat_mask = mask.reshape(B, N)                             # (B, N)

        # Sort indices: unmasked first (0), masked last (1)
        ids_shuffle  = torch.argsort(flat_mask.float(), dim=1)    # (B, N)
        ids_restore  = torch.argsort(ids_shuffle, dim=1)          # (B, N)
        n_vis        = int((~flat_mask[0]).sum().item())

        # ── Step 3: gather visible tokens and encode ──────────────────────────
        vis_ids  = ids_shuffle[:, :n_vis]                          # (B, n_vis)
        visible  = torch.gather(
            tokens, 1, vis_ids.unsqueeze(-1).expand(-1, -1, self.d_model)
        )                                                          # (B, n_vis, d_model)
        enc_vis  = self.encoder(visible)                           # (B, n_vis, d_model)

        # ── Step 4: rebuild full sequence for the decoder ─────────────────────
        # Visible positions → encoder outputs; masked positions → [MASK] token
        mask_tokens = self.mask_token.expand(B, N - n_vis, self.d_model)
        full = torch.cat([enc_vis, mask_tokens], dim=1)            # (B, N, d_model)
        # Unshuffle back to original token order
        full = torch.gather(
            full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, self.d_model)
        )                                                          # (B, N, d_model)

        # ── Step 5: decode → patch reconstructions ───────────────────────────
        pred = self.decoder(full)                                  # (B, N, patch_size)

        # ── Step 6: MSE loss on masked patches only ───────────────────────────
        target = x[:, :, : n_patches * P].reshape(B, N, P)        # (B, N, patch_size)
        loss   = F.mse_loss(pred[flat_mask], target[flat_mask])

        return loss, pred, mask


# ── Fine-tuning Wrapper ────────────────────────────────────────────────────────

class ECGFinetuneModel(nn.Module):
    """
    Wraps a pre-trained ECGFoundationModel with a task-specific head.

    Usage
    ─────
      # Load pre-trained backbone
      backbone = ECGFoundationModel()
      backbone.load_state_dict(torch.load("pretrained.pt"))

      # Attach head and fine-tune
      model = ECGFinetuneModel(backbone, num_classes=5)  # e.g. rhythm classes

      # Optionally freeze the encoder for linear probing
      model = ECGFinetuneModel(backbone, num_classes=5, freeze_encoder=True)
    """

    def __init__(
        self,
        backbone:        ECGFoundationModel,
        num_classes:     int,
        hidden_dim:      int  = 256,
        dropout:         float = 0.1,
        freeze_encoder:  bool  = False,
    ):
        super().__init__()
        self.backbone = backbone

        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False

        d = backbone.d_model
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x:        torch.Tensor,
        lead_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        (B, n_leads, T)  →  (B, num_classes)  logits
        """
        rep = self.backbone.get_representation(x, lead_ids)  # (B, d_model)
        return self.head(rep)