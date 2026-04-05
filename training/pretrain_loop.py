"""
training/pretrain.py — MAE masked signal modelling pre-training loop

What changed from v1 and why
─────────────────────────────
  v1 used NT-Xent (SimCLR-style) contrastive learning:
    • Required two augmented views per sample → augment_ecg()
    • Required NTXentLoss with a temperature hyperparameter
    • Loss was computed externally on L2-normalised projection vectors

  v2 uses MAE masked signal modelling:
    • No augmentation pairs — the mask IS the self-supervised signal
    • Loss is computed inside ECGFoundationModel.forward() as MSE on
      masked patches only; pretrain() just calls loss.backward()
    • Mask strategy (temporal / lead / spatiotemporal) is set on the
      model at construction time via args.mask_strategy
    • Warmup scheduler added — MAE training benefits from a short
      linear warmup before cosine decay, especially at d_model=512
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def pretrain(
    model:  nn.Module,
    loader: DataLoader,
    args,
    device: torch.device,
) -> nn.Module:
    """
    Run MAE masked signal modelling pre-training for args.pretrain_epochs.

    Args:
        model  : ECGFoundationModel (or DataParallel-wrapped).
                 forward() must return (loss, pred, mask).
        loader : DataLoader yielding batches with key 'ecg' and optionally
                 'lead_ids' (LongTensor of shape (n_leads,)).
        args   : namespace from args.get_args()
        device : torch.device

    Returns:
        trained model (still on device)
    """
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.pretrain_lr,
        weight_decay = args.weight_decay,
        betas        = (0.9, 0.95),   # MAE paper recommendation
    )

    # Linear warmup for warmup_epochs, then cosine decay to args.min_lr
    warmup_epochs = getattr(args, "warmup_epochs", 10)
    min_lr        = getattr(args, "min_lr", 1e-6)
    total_epochs  = args.pretrain_epochs

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # linear ramp from 0 → 1
            return (epoch + 1) / warmup_epochs
        # cosine decay from 1 → min_lr / pretrain_lr
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine   = 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item())
        floor    = min_lr / args.pretrain_lr
        return floor + (1.0 - floor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler("cuda", enabled=args.use_amp)

    # Resolve inner model when wrapped in DataParallel
    # (needed to read mask_strategy for logging)
    inner = model.module if isinstance(model, torch.nn.DataParallel) else model

    print(f"  [Pretrain] strategy={inner.mask_strategy}"
          f"  mask_ratio={inner.mask_ratio}"
          f"  warmup={warmup_epochs} epochs")

    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss    = 0.0
        total_batches = 0

        for batch in loader:
            ecg = batch["ecg"].to(device, non_blocking=True)   # (B, L, T)

            # lead_ids is optional — present when the batch contains a
            # variable or partial set of leads (e.g. from a 6-lead device)
            lead_ids = batch.get("lead_ids", None)
            if lead_ids is not None:
                lead_ids = lead_ids.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=args.use_amp):
                # loss is MSE on masked patches only — computed inside model
                loss, _pred, _mask = model(ecg, lead_ids)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss    += loss.item()
            total_batches += 1

        scheduler.step()

        avg_loss = total_loss / max(1, total_batches)
        print(
            f"  [Pretrain] Epoch {epoch:>3}/{total_epochs}"
            f"  loss={avg_loss:.4f}"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
        )

    return model