"""
training/finetune.py — fine-tuning loop for ECGFinetuneModel  (v3)

What changed from v1 and why
─────────────────────────────
  v2 constructed the loss inline with a hard if/elif:
      if classification: nn.CrossEntropyLoss()
      elif regression:   nn.MSELoss()
  This bypassed FocalLoss and LabelSmoothingCrossEntropy added to loss.py,
  so --focal_gamma and --label_smoothing had no effect at runtime.

  v3 replaces the inline block with build_finetune_loss(args), which reads
  args.focal_gamma, args.label_smoothing, and args.focal_alpha and returns
  the right criterion automatically.  All other logic is unchanged.

  Regression logging updated: RMSELoss already returns root-MSE, so the
  old manual sqrt is removed to avoid logging the square root of RMSE.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loss import build_finetune_loss


def finetune(
    ft_model: nn.Module,
    loader:   DataLoader,
    args,
    device:   torch.device,
) -> nn.Module:
    """
    Run fine-tuning for args.finetune_epochs.

    Args:
        ft_model : ECGFinetuneModel (or DataParallel-wrapped).
                   forward() returns logits (B, num_classes) for
                   classification or (B, 1) for regression.
        loader   : DataLoader yielding batches with keys:
                     'ecg'      — (B, n_leads, T)
                     'label'    — (B,) LongTensor   [classification]
                     'target'   — (B,) FloatTensor  [regression]
                     'lead_ids' — (n_leads,) optional
        args     : namespace from args.get_args()
                   relevant keys:
                     finetune_task    : 'classification' | 'regression'
                     label_smoothing  : float  (default 0.0)
                     focal_gamma      : float  (default 0.0 = disabled)
                     focal_alpha      : str    (default None)
        device   : torch.device

    Returns:
        trained model (still on device)
    """
    ft_model.to(device)

    # Only optimise parameters that require grad — honours freeze_encoder=True
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, ft_model.parameters()),
        lr           = args.finetune_lr,
        weight_decay = args.weight_decay,
    )

    # Cosine decay — fine-tuning runs are short so no warmup needed
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = args.finetune_epochs,
        eta_min = getattr(args, "min_lr", 1e-6),
    )

    # ── Loss — single call covers CE / label-smoothed CE / Focal / RMSE ──────
    criterion     = build_finetune_loss(args)
    finetune_task = getattr(args, "finetune_task", "classification")

    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)

    print(
        f"  [Finetune] task={finetune_task}"
        f"  criterion={criterion.__class__.__name__}"
        f"  lr={args.finetune_lr:.2e}"
    )

    for epoch in range(1, args.finetune_epochs + 1):
        ft_model.train()

        total_loss    = 0.0
        total_correct = 0
        total_samples = 0

        for batch in loader:
            ecg = batch["ecg"].to(device, non_blocking=True)         # (B, L, T)

            # lead_ids: optional for variable-lead batches
            lead_ids = batch.get("lead_ids", None)
            if lead_ids is not None:
                lead_ids = lead_ids.to(device, non_blocking=True)

            # Ground-truth: 'label' for classification, 'target' for regression
            if finetune_task == "classification":
                gt = batch["label"].to(device, non_blocking=True)    # (B,) LongTensor
            else:
                gt = batch["target"].float().to(device, non_blocking=True)  # (B,)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=args.use_amp):
                logits = ft_model(ecg, lead_ids)                      # (B, num_classes) or (B, 1)

                if finetune_task == "regression":
                    loss = criterion(logits.squeeze(1), gt)
                else:
                    loss = criterion(logits, gt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ft_model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss    += loss.item()
            total_samples += ecg.size(0)

            if finetune_task == "classification":
                preds          = logits.argmax(dim=1)
                total_correct += (preds == gt).sum().item()

        scheduler.step()

        n        = len(loader)
        avg_loss = total_loss / max(1, n)

        if finetune_task == "classification":
            acc = 100.0 * total_correct / max(1, total_samples)
            print(
                f"  [Finetune] Epoch {epoch:>3}/{args.finetune_epochs}"
                f"  loss={avg_loss:.4f}"
                f"  acc={acc:.2f}%"
                f"  lr={scheduler.get_last_lr()[0]:.2e}"
            )
        else:
            # avg_loss is already RMSE — RMSELoss returns root-MSE directly
            print(
                f"  [Finetune] Epoch {epoch:>3}/{args.finetune_epochs}"
                f"  rmse={avg_loss:.4f}"
                f"  lr={scheduler.get_last_lr()[0]:.2e}"
            )

    return ft_model