"""
loss.py — Loss functions for ECG Foundation Model  (v2)

What changed from v1 and why
─────────────────────────────
  Removed — NTXentLoss:
    NT-Xent / SimCLR contrastive loss is no longer used.
    MAE pre-training loss (MSE on masked patches) is computed *inside*
    ECGFoundationModel.forward() and returned as the first element of the
    (loss, pred, mask) tuple.  pretrain.py calls loss.backward() directly
    — no external loss class is needed at all for pre-training.

  Removed — FinetuneLoss:
    The multi-task head (age / gender / bmi / sbp / dbp) no longer exists.
    ECGFinetuneModel.forward() returns a single logits tensor (B, num_classes).
    The fine-tuning loss is therefore either CrossEntropyLoss (classification)
    or MSELoss (regression), selected at runtime by finetune.py based on
    args.finetune_task.  No custom class is needed for the common case.

  Added — LabelSmoothingCrossEntropy:
    A drop-in replacement for nn.CrossEntropyLoss with label smoothing.
    Useful when fine-tuning on small or noisy ECG label sets where the
    model can overfit to hard labels quickly.  Activated by passing
    smoothing > 0.0 in ECGFinetuneModel or finetune.py.

  Added — FocalLoss:
    Down-weights easy negatives and focuses training on hard, mis-classified
    samples.  Important for imbalanced cardiac datasets where normal sinus
    rhythm vastly outnumbers rare pathologies (e.g. WPW, Brugada).

  Added — RMSELoss:
    Thin wrapper around MSELoss that returns root-MSE.  Used when the
    regression target has a natural unit (e.g. heart rate in bpm, QTc in ms)
    and you want the logged loss to be interpretable in that unit.

  NTXentLoss is kept but clearly marked deprecated so any external scripts
  that still import it do not break immediately.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Pre-training note ──────────────────────────────────────────────────────────
# MAE reconstruction loss is computed inside ECGFoundationModel.forward().
# There is nothing to import here for pre-training.


# ── Deprecated: kept for backward compatibility only ──────────────────────────

class NTXentLoss(nn.Module):
    """
    DEPRECATED — no longer used by this codebase.

    NT-Xent (SimCLR) contrastive loss was used in v1 pre-training.
    MAE masked signal modelling replaced it in v2.
    This class is retained so that any external scripts or notebooks that
    import it do not raise ImportError immediately.  It will be removed in
    a future version.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.T = temperature
        import warnings
        warnings.warn(
            "NTXentLoss is deprecated and will be removed in a future release. "
            "MAE pre-training loss is now computed inside ECGFoundationModel.forward().",
            DeprecationWarning,
            stacklevel=2,
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B   = z1.size(0)
        z   = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.T
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e4)
        labels = torch.cat(
            [torch.arange(B, 2 * B), torch.arange(0, B)]
        ).to(z.device)
        return F.cross_entropy(sim, labels)


# ── Fine-tuning losses ─────────────────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Replaces hard one-hot targets with soft targets:
        y_soft = (1 - smoothing) * y_hard  +  smoothing / num_classes

    Benefits for ECG fine-tuning:
    • Prevents overconfident predictions on small or noisy label sets.
    • Improves calibration — predicted probabilities better reflect true
      uncertainty, which matters in clinical decision support.
    • Slight regularisation effect when fine-tuning a large pre-trained
      backbone on a few hundred labelled examples.

    Args:
        smoothing : float in [0, 1).  0.0 = standard cross-entropy.
                    Typical values: 0.05–0.15 for ECG classification.
        reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C)  raw unnormalised scores
            targets : (B,)    integer class indices
        Returns:
            scalar loss (or per-sample if reduction='none')
        """
        num_classes = logits.size(-1)
        log_probs   = F.log_softmax(logits, dim=-1)           # (B, C)

        # Hard-target NLL (standard cross-entropy term)
        nll = F.nll_loss(log_probs, targets, reduction=self.reduction)

        # Smooth term: mean log-prob across all classes
        smooth = -log_probs.mean(dim=-1)
        if self.reduction == "mean":
            smooth = smooth.mean()
        elif self.reduction == "sum":
            smooth = smooth.sum()
        # else: 'none' → keep (B,)

        return (1.0 - self.smoothing) * nll + self.smoothing * smooth


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) for imbalanced ECG classification.

    Standard cross-entropy treats every sample equally regardless of how
    easily it is classified.  In cardiac datasets, the normal class typically
    comprises 60–90 % of recordings, so the gradient is dominated by easy
    negatives and rare pathologies are under-weighted.

    Focal loss applies a modulating factor (1 - p_t)^γ that reduces the
    loss contribution of well-classified samples and focuses training on
    the hard, mis-classified minority classes.

    Args:
        gamma  : focusing parameter γ ≥ 0.
                 0 = standard cross-entropy.
                 2 is the value used in the original paper.
        alpha  : optional (C,) class-weight tensor for additional class
                 balancing on top of the focal modulation.  Pass
                 torch.tensor([w0, w1, …, wC-1]) where wi is the weight
                 for class i.  None = no class weighting.
        reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma:     float               = 2.0,
        alpha:     torch.Tensor | None = None,
        reduction: str                 = "mean",
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be ≥ 0, got {gamma}")
        self.gamma     = gamma
        self.reduction = reduction
        # register as buffer so it moves to the right device with .to(device)
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C)  raw unnormalised scores
            targets : (B,)    integer class indices
        Returns:
            scalar focal loss
        """
        log_probs = F.log_softmax(logits, dim=-1)          # (B, C)
        probs     = log_probs.exp()                         # (B, C)

        # Gather the log-prob and prob of the true class for each sample
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        pt     = probs.gather(1,     targets.unsqueeze(1)).squeeze(1)  # (B,)

        # Focal modulation
        focal_weight = (1.0 - pt) ** self.gamma                        # (B,)
        loss         = -focal_weight * log_pt                           # (B,)

        # Optional per-class alpha weighting
        if self.alpha is not None:
            at   = self.alpha.gather(0, targets)                        # (B,)
            loss = at * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'


class RMSELoss(nn.Module):
    """
    Root Mean Squared Error loss for regression fine-tuning.

    Wraps MSELoss so that the logged loss is in the same unit as the
    prediction target rather than its square.

    Examples
    ────────
      Heart rate prediction   : RMSE in bpm
      QTc interval prediction : RMSE in ms
      Age estimation          : RMSE in years

    Args:
        eps : small constant added before sqrt to avoid zero-gradient at
              perfect predictions.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : (B,) or (B, 1)  model predictions
            target : (B,) or (B, 1)  ground-truth values
        Returns:
            scalar RMSE
        """
        return torch.sqrt(F.mse_loss(pred, target) + self.eps)


# ── Convenience factory ────────────────────────────────────────────────────────

def build_finetune_loss(args) -> nn.Module:
    """
    Construct the appropriate fine-tuning loss from parsed args.

    Reads:
        args.finetune_task   : 'classification' | 'regression'
        args.label_smoothing : float (default 0.0)
        args.focal_gamma     : float (default 0.0 = disabled)
        args.focal_alpha     : comma-separated class weights, e.g. '1.0,2.5'
                               or None (default)

    Usage in finetune.py:
        criterion = build_finetune_loss(args)
        loss = criterion(logits, gt)
    """
    task = getattr(args, "finetune_task", "classification")

    if task == "regression":
        return RMSELoss()

    # Classification — choose between standard CE, label-smoothed CE, and Focal
    smoothing  = getattr(args, "label_smoothing", 0.0)
    focal_gamma = getattr(args, "focal_gamma",    0.0)
    focal_alpha_str = getattr(args, "focal_alpha", None)

    if focal_gamma > 0.0:
        alpha = None
        if focal_alpha_str is not None:
            alpha = torch.tensor([float(w) for w in focal_alpha_str.split(",")])
        return FocalLoss(gamma=focal_gamma, alpha=alpha)

    if smoothing > 0.0:
        return LabelSmoothingCrossEntropy(smoothing=smoothing)

    return nn.CrossEntropyLoss()