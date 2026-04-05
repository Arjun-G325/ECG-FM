"""
main.py — ECG Foundation Model: pre-training → fine-tuning → inference demo

Usage examples
──────────────
  # run on real Indian ECG data (auto-detects shape, labels, num_classes)
  python main.py --data_path ecg_data/Indian_ECG_175k

  # explicit GPU, 8 workers
  python main.py --data_path ecg_data/Indian_ECG_175k --device cuda:0 --num_workers 8

  # quick smoke-test with synthetic data (no --data_path)
  python main.py --pretrain_epochs 2 --finetune_epochs 2 --n_samples 200

  # freeze backbone (linear probing)
  python main.py --data_path ecg_data/Indian_ECG_175k --freeze_backbone

  # focal loss for class imbalance
  python main.py --data_path ecg_data/Indian_ECG_175k --focal_gamma 2.0

  # spatiotemporal masking
  python main.py --data_path ecg_data/Indian_ECG_175k --mask_strategy spatiotemporal
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

from args import get_args
from ecg_data.dataset import IndianECGDataset, ECGDataset
from models import ECGFoundationModel, ECGFinetuneModel
from training import pretrain, finetune


# ── Dataset ────────────────────────────────────────────────────────────────────

def build_dataset(args):
    if args.data_path:
        common = dict(
            data_dir  = args.data_path,
            seq_len   = args.seq_len,
            n_leads   = args.n_leads,
            normalise = True,
        )
        train_ds = IndianECGDataset(**common, split="train")
        val_ds   = IndianECGDataset(**common, split="val")

        if train_ds.num_classes != args.num_classes:
            print(f"[INFO] Overriding --num_classes {args.num_classes} → "
                  f"{train_ds.num_classes}  (detected from labels)")
            args.num_classes = train_ds.num_classes

        args.finetune_task = (
            "classification" if train_ds.is_classification else "regression"
        )
        print(f"[INFO] finetune_task auto-set to '{args.finetune_task}'")
        return train_ds, val_ds

    ds = ECGDataset(
        n_samples   = args.n_samples,
        n_leads     = args.n_leads,
        seq_len     = args.seq_len,
        num_classes = args.num_classes,
    )
    return ds, ds


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(args) -> ECGFoundationModel:
    return ECGFoundationModel(
        patch_size    = args.patch_size,
        d_model       = args.d_model,
        nhead         = args.nhead,
        enc_layers    = args.enc_layers,
        dec_layers    = args.dec_layers,
        dec_dim       = args.dec_dim,
        dim_ff        = args.dim_ff,
        dropout       = args.dropout,
        max_leads     = args.n_leads,
        max_patches   = args.max_patches,
        mask_ratio    = args.mask_ratio,
        mask_strategy = args.mask_strategy,
    )


def maybe_wrap_data_parallel(model, args):
    if args.multi_gpu:
        n = torch.cuda.device_count()
        if n < 2:
            print(f"[WARN] --multi_gpu: only {n} device(s) visible — skipping.")
        else:
            print(f"[INFO] DataParallel across {n} GPUs.")
            model = torch.nn.DataParallel(model)
    return model


# ── Hardware summary ───────────────────────────────────────────────────────────

def print_hardware_info(args):
    device = args.device
    print(f"[HW]  Device             : {device}")
    if device.type == "cuda":
        idx       = device.index or 0
        props     = torch.cuda.get_device_properties(idx)
        total_gib = props.total_memory / (1024 ** 3)
        print(f"[HW]  GPU name           : {props.name}")
        print(f"[HW]  GPU total memory   : {total_gib:.2f} GiB")
        print(f"[HW]  CUDA visible       : {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
        print(f"[HW]  AMP enabled        : {args.use_amp}")
        print(f"[HW]  cuDNN benchmark    : {torch.backends.cudnn.benchmark}")
        print(f"[HW]  TF32               : {torch.backends.cuda.matmul.allow_tf32}")
        if args.cuda_max_memory_gb is not None:
            frac = min(args.cuda_max_memory_gb / total_gib, 1.0)
            print(f"[HW]  CUDA mem limit     : {args.cuda_max_memory_gb:.2f} GiB ({frac*100:.1f}%)")
        elif args.cuda_memory_fraction is not None:
            print(f"[HW]  CUDA mem limit     : fraction {args.cuda_memory_fraction:.3f}")
        else:
            print(f"[HW]  CUDA mem limit     : unlimited")
        if args.cuda_reserve_gb is not None:
            print(f"[HW]  CUDA reserved      : {args.cuda_reserve_gb:.2f} GiB")
    print(f"[HW]  DataLoader workers : {args.num_workers}")
    print(f"[HW]  Intra-op threads   : {torch.get_num_threads()}")
    print(f"[HW]  Inter-op threads   : {torch.get_num_interop_threads()}")
    if args.cpu_max_memory_gb is not None:
        print(f"[HW]  CPU mem limit      : {args.cpu_max_memory_gb:.2f} GiB")
    else:
        print(f"[HW]  CPU mem limit      : unlimited")


# ── Inference demo ─────────────────────────────────────────────────────────────

def inference_demo(ft_model, dataset, device, args):
    ft_model.eval()
    sample_ecg = dataset.ecg[:4].to(device)
    with torch.no_grad():
        logits = ft_model(sample_ecg)

    finetune_task = getattr(args, "finetune_task", "classification")
    hdr = f"{'':14} {'Sample 1':>10} {'Sample 2':>10} {'Sample 3':>10} {'Sample 4':>10}"
    sep = "-" * 54
    print(f"\n{hdr}\n{sep}")

    if finetune_task == "classification":
        pred_labels   = logits.argmax(dim=1).cpu().numpy()
        actual_labels = dataset.label[:4].numpy()
        names = getattr(dataset, "class_names", None)
        def fmt(i): return names[int(i)] if names else str(int(i))
        print(f"{'actual':14} " + "  ".join(f"{fmt(v):>10}" for v in actual_labels))
        print(f"{'predicted':14} " + "  ".join(f"{fmt(v):>10}" for v in pred_labels))
        correct = (pred_labels == actual_labels).sum()
        print(f"\n  Demo accuracy : {correct}/4")
    else:
        pred_vals   = logits.squeeze(1).cpu().numpy().round(3)
        actual_vals = dataset.target[:4].numpy().round(3)
        print(f"{'actual':14} " + "  ".join(f"{v:>10.3f}" for v in actual_vals))
        print(f"{'predicted':14} " + "  ".join(f"{v:>10.3f}" for v in pred_vals))
        rmse = float(np.sqrt(((pred_vals - actual_vals) ** 2).mean()))
        print(f"\n  Demo RMSE : {rmse:.4f}")
    print(sep)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    SAVE_DIR = args.save_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print(" ECG Foundation Model  (v2 — MAE pre-training)")
    print("=" * 60)
    print_hardware_info(args)
    print(f"[INFO] Save dir  : {SAVE_DIR}")
    print(f"[INFO] Data path : {args.data_path or 'synthetic'}")

    # 1. Dataset
    print(f"\n[1/4] Loading dataset ...")
    train_ds, val_ds = build_dataset(args)
    pretrain_loader = DataLoader(train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=args.pin_memory, drop_last=True)
    finetune_loader = DataLoader(train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=args.pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    print(f"      Train : {len(train_ds)}  Val : {len(val_ds)}")
    print(f"      num_classes   : {args.num_classes}")
    print(f"      finetune_task : {args.finetune_task}")

    # 2. Model
    print("\n[2/4] Instantiating ECGFoundationModel ...")
    foundation   = build_model(args)
    total_params = sum(p.numel() for p in foundation.parameters()) / 1e6
    print(f"      Parameters : {total_params:.2f} M")
    with torch.no_grad():
        _loss, _pred, _mask = foundation(torch.randn(2, args.n_leads, args.seq_len))
    n_patches = args.seq_len // args.patch_size
    print(f"      Sanity loss : {_loss.item():.4f}")
    print(f"      Pred shape  : {tuple(_pred.shape)}")
    print(f"      Mask shape  : {tuple(_mask.shape)}")
    foundation = maybe_wrap_data_parallel(foundation, args)

    # 3. Pre-training
    print(f"\n[3/4] Pre-training  ({args.pretrain_epochs} epochs) ...")
    foundation = pretrain(foundation, pretrain_loader, args, args.device)
    backbone = foundation.module if isinstance(foundation, torch.nn.DataParallel) else foundation
    pretrain_path = os.path.join(SAVE_DIR, "ecg_pretrained.pt")
    torch.save(backbone.state_dict(), pretrain_path)
    print(f"      Saved → {pretrain_path}")

    # 4. Fine-tuning
    print(f"\n[4/4] Fine-tuning  ({args.finetune_epochs} epochs) ...")
    ft_model = ECGFinetuneModel(
        backbone       = backbone,
        num_classes    = args.num_classes,
        freeze_encoder = args.freeze_backbone,
    )
    ft_model = maybe_wrap_data_parallel(ft_model, args)
    ft_model = finetune(ft_model, finetune_loader, args, args.device)
    ft_backbone = ft_model.module if isinstance(ft_model, torch.nn.DataParallel) else ft_model
    finetune_path = os.path.join(SAVE_DIR, "ecg_finetuned.pt")
    torch.save(ft_backbone.state_dict(), finetune_path)
    print(f"      Saved → {finetune_path}")

    # Inference demo
    print("\n── Inference demo (val split) ──")
    inference_demo(ft_backbone, val_ds, args.device, args)