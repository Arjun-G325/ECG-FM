"""
args.py — centralised argument parser for ECG Foundation Model  (v2)

What changed from v1 and why
─────────────────────────────
  Removed (no longer exist in the codebase):
    --n_query_tokens   TransformerDecoder with learnable queries is gone
    --proj_dim         ProjectionHead / NT-Xent projection is gone
    --temperature      NT-Xent temperature is gone

  Updated defaults (aligned with d_model=512 foundation model scale):
    --d_model          256  → 512
    --enc_layers         4  → 8
    --dec_layers         2  → 4
    --dim_ff           512  → 2048

  Added — Model Architecture:
    --patch_size       samples per temporal patch (50 @ 500 Hz = 100 ms)
    --dec_dim          MAE decoder hidden dim (narrower than encoder by design)
    --max_patches      upper bound on temporal patches per lead

  Added — Pre-training:
    --mask_ratio       fraction of tokens masked (MAE; default 0.75)
    --mask_strategy    temporal | lead | spatiotemporal
    --warmup_epochs    linear LR warmup before cosine decay
    --min_lr           cosine decay floor

  Added — Fine-tuning:
    --finetune_task    classification | regression
    --num_classes      output classes for the task head
"""

import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description="ECG Foundation Model — Pre-training + Fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ────────────────────────────────────────────────────────────────
    data = parser.add_argument_group("Dataset")
    data.add_argument("--n_samples",  type=int,  default=1000, help="number of ECG samples")
    data.add_argument("--n_leads",    type=int,  default=12,   help="number of ECG leads")
    data.add_argument("--seq_len",    type=int,  default=5000, help="timesteps per sample (500 Hz × 10 s)")
    data.add_argument("--data_path",  type=str,  default=None, help="path to real dataset (None = synthetic)")

    # ── Model Architecture ─────────────────────────────────────────────────────
    arch = parser.add_argument_group("Model Architecture")
    arch.add_argument(
        "--patch_size",
        type=int, default=50,
        help="samples per temporal patch (50 @ 500 Hz = 100 ms window)",
    )
    arch.add_argument(
        "--d_model",
        type=int, default=512,
        help="transformer hidden dim; 512 gives 64-dim subspaces per head at nhead=8",
    )
    arch.add_argument(
        "--nhead",
        type=int, default=8,
        help="attention heads (d_model must be divisible by nhead)",
    )
    arch.add_argument(
        "--enc_layers",
        type=int, default=8,
        help="encoder transformer layers",
    )
    arch.add_argument(
        "--dec_layers",
        type=int, default=4,
        help="MAE decoder transformer layers (kept shallow; reconstruction pressure lands on encoder)",
    )
    arch.add_argument(
        "--dec_dim",
        type=int, default=256,
        help="MAE decoder hidden dim (narrower than d_model by design)",
    )
    arch.add_argument(
        "--dim_ff",
        type=int, default=2048,
        help="feedforward dim inside transformer layers (typically 4 × d_model)",
    )
    arch.add_argument(
        "--dropout",
        type=float, default=0.1,
        help="dropout rate",
    )
    arch.add_argument(
        "--max_patches",
        type=int, default=256,
        help="upper bound on temporal patches per lead for positional embedding table size",
    )

    # ── Pre-training ───────────────────────────────────────────────────────────
    pre = parser.add_argument_group("Pre-training")
    pre.add_argument(
        "--pretrain_epochs",
        type=int, default=100,
        help="total pre-training epochs",
    )
    pre.add_argument(
        "--pretrain_lr",
        type=float, default=1.5e-4,
        help="peak learning rate after warmup (MAE paper: 1.5e-4 × batch/256)",
    )
    pre.add_argument(
        "--warmup_epochs",
        type=int, default=10,
        help="linear LR warmup epochs before cosine decay begins",
    )
    pre.add_argument(
        "--min_lr",
        type=float, default=1e-6,
        help="cosine decay floor; also used as eta_min in fine-tuning scheduler",
    )
    pre.add_argument(
        "--mask_ratio",
        type=float, default=0.75,
        help="fraction of tokens masked during MAE pre-training (0.75 from MAE paper)",
    )
    pre.add_argument(
        "--mask_strategy",
        type=str, default="temporal",
        choices=["temporal", "lead", "spatiotemporal"],
        help=(
            "masking strategy: "
            "'temporal' = random patches within each lead (default); "
            "'lead' = entire leads masked, tests inter-lead geometry; "
            "'spatiotemporal' = 2-D rectangular blocks across leads × time (hardest)"
        ),
    )

    # ── Fine-tuning ────────────────────────────────────────────────────────────
    fine = parser.add_argument_group("Fine-tuning")
    fine.add_argument(
        "--finetune_epochs",
        type=int, default=20,
        help="fine-tuning epochs",
    )
    fine.add_argument(
        "--finetune_lr",
        type=float, default=3e-5,
        help="fine-tuning peak learning rate",
    )
    fine.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="freeze encoder weights during fine-tuning (linear probing mode)",
    )
    fine.add_argument(
        "--finetune_task",
        type=str, default="classification",
        choices=["classification", "regression"],
        help=(
            "downstream task type: "
            "'classification' uses CrossEntropyLoss + accuracy; "
            "'regression' uses MSELoss + RMSE"
        ),
    )
    fine.add_argument(
        "--num_classes",
        type=int, default=5,
        help=(
            "number of output classes for the task head. "
            "For regression set to 1. "
            "Examples: 2=normal/abnormal, 5=rhythm classes, 26=PTB-XL superclasses"
        ),
    )

    # ── Training ───────────────────────────────────────────────────────────────
    train = parser.add_argument_group("Training")
    train.add_argument("--batch_size",   type=int,   default=64,   help="batch size (larger batches improve MAE stability)")
    train.add_argument("--weight_decay", type=float, default=0.05, help="AdamW weight decay (MAE paper uses 0.05)")
    train.add_argument("--grad_clip",    type=float, default=1.0,  help="gradient clip norm")

    # ── Hardware / Compute ─────────────────────────────────────────────────────
    hw = parser.add_argument_group("Hardware / Compute")
    hw.add_argument(
        "--device",
        type=str, default="auto",
        help="compute device: 'auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', …",
    )
    hw.add_argument(
        "--cuda_devices",
        type=str, default=None, metavar="IDS",
        help="visible CUDA device IDs, comma-separated (sets CUDA_VISIBLE_DEVICES). E.g. '0,1,2'.",
    )
    hw.add_argument(
        "--multi_gpu",
        action="store_true",
        help="wrap model in DataParallel across all visible CUDA devices",
    )
    hw.add_argument(
        "--num_workers",
        type=int, default=0,
        help="DataLoader worker processes (0 = main process only; safe on Windows)",
    )
    hw.add_argument(
        "--num_cores",
        type=int, default=None, metavar="N",
        help="CPU cores for intra-op parallelism (torch.set_num_threads)",
    )
    hw.add_argument(
        "--num_interop_cores",
        type=int, default=None, metavar="N",
        help="CPU cores for inter-op parallelism (torch.set_num_interop_threads)",
    )
    hw.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction, default=None,
        help="pin DataLoader memory for faster GPU transfers (auto-True on CUDA)",
    )
    hw.add_argument(
        "--fp16",
        action="store_true",
        help="force AMP mixed-precision even on CPU (experimental)",
    )
    hw.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction, default=True,
        help="allow TF32 on Ampere+ GPUs (slightly lower precision, faster matmuls)",
    )
    hw.add_argument(
        "--cudnn_benchmark",
        action=argparse.BooleanOptionalAction, default=True,
        help="enable cuDNN auto-tuner; best when input sizes are fixed",
    )

    # ── Memory Limits ──────────────────────────────────────────────────────────
    mem = parser.add_argument_group("Memory Limits")
    mem.add_argument(
        "--cuda_max_memory_gb",
        type=float, default=None, metavar="GB",
        help="soft GPU memory ceiling in GiB via set_per_process_memory_fraction()",
    )
    mem.add_argument(
        "--cuda_memory_fraction",
        type=float, default=None, metavar="F",
        help="alternative to --cuda_max_memory_gb: direct fraction of total GPU memory in (0, 1]",
    )
    mem.add_argument(
        "--cuda_reserve_gb",
        type=float, default=None, metavar="GB",
        help="pre-allocate a CUDA memory pool of this many GiB at startup to reduce fragmentation",
    )
    mem.add_argument(
        "--cpu_max_memory_gb",
        type=float, default=None, metavar="GB",
        help="soft CPU RAM ceiling in GiB via RLIMIT_AS (POSIX only; ignored on Windows)",
    )

    # ── Misc ───────────────────────────────────────────────────────────────────
    misc = parser.add_argument_group("Misc")
    misc.add_argument("--seed",     type=int, default=42,   help="global random seed")
    misc.add_argument("--save_dir", type=str, default=None, help="checkpoint directory (default: script folder)")

    args = parser.parse_args()
    _validate_args(args)
    _resolve_hardware(args)
    return args


def _validate_args(args):
    """Catch configuration errors early before any model or data is built."""

    if args.d_model % args.nhead != 0:
        raise ValueError(
            f"--d_model ({args.d_model}) must be divisible by --nhead ({args.nhead}). "
            f"Try --d_model {args.nhead * (args.d_model // args.nhead + 1)} or "
            f"--nhead {args.d_model // args.nhead}."
        )

    if args.dec_dim > args.d_model:
        raise ValueError(
            f"--dec_dim ({args.dec_dim}) should be ≤ --d_model ({args.d_model}). "
            "The MAE decoder is intentionally narrower than the encoder."
        )

    if args.seq_len % args.patch_size != 0:
        raise ValueError(
            f"--seq_len ({args.seq_len}) must be divisible by --patch_size ({args.patch_size}). "
            f"Nearest valid seq_len: {(args.seq_len // args.patch_size) * args.patch_size}."
        )

    n_patches = args.seq_len // args.patch_size
    if n_patches > args.max_patches:
        raise ValueError(
            f"seq_len / patch_size = {n_patches} exceeds --max_patches ({args.max_patches}). "
            f"Increase --max_patches to at least {n_patches}."
        )

    if not (0.0 < args.mask_ratio < 1.0):
        raise ValueError(f"--mask_ratio must be in (0, 1), got {args.mask_ratio}.")

    if args.finetune_task == "regression" and args.num_classes != 1:
        print(
            f"[WARN] --finetune_task regression but --num_classes={args.num_classes}. "
            "Setting --num_classes to 1."
        )
        args.num_classes = 1

    if args.warmup_epochs >= args.pretrain_epochs:
        raise ValueError(
            f"--warmup_epochs ({args.warmup_epochs}) must be < "
            f"--pretrain_epochs ({args.pretrain_epochs})."
        )


def _resolve_hardware(args):
    """Validate hardware args and resolve the final torch.device."""
    import os

    if args.cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    if args.device == "auto":
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(args.device)

    if args.pin_memory is None:
        args.pin_memory = args.device.type == "cuda"

    if args.num_cores is not None:
        torch.set_num_threads(args.num_cores)
    if args.num_interop_cores is not None:
        torch.set_num_interop_threads(args.num_interop_cores)

    if args.device.type == "cuda":
        torch.backends.cudnn.benchmark     = args.cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32    = args.tf32

    args.use_amp = (args.device.type == "cuda") or args.fp16

    _apply_memory_limits(args)


def _apply_memory_limits(args):
    """
    Enforce CUDA and CPU memory ceilings declared on the command line.
    Unchanged from v1 — no memory-limit logic was affected by the model changes.
    """
    import math

    if args.device.type == "cuda" and torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        fraction: float | None = None

        if args.cuda_max_memory_gb is not None:
            if args.cuda_max_memory_gb <= 0:
                raise ValueError("--cuda_max_memory_gb must be > 0")
            total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            fraction  = min(args.cuda_max_memory_gb / total_gib, 1.0)
            if args.cuda_max_memory_gb > total_gib:
                print(
                    f"[WARN] --cuda_max_memory_gb {args.cuda_max_memory_gb:.1f} GiB "
                    f"exceeds device total {total_gib:.1f} GiB — clamping to 1.0"
                )

        elif args.cuda_memory_fraction is not None:
            if not (0 < args.cuda_memory_fraction <= 1.0):
                raise ValueError("--cuda_memory_fraction must be in (0, 1]")
            fraction = args.cuda_memory_fraction

        if fraction is not None:
            for dev_idx in range(n_devices):
                torch.cuda.set_per_process_memory_fraction(fraction, device=dev_idx)
            limit_str = (
                f"{args.cuda_max_memory_gb:.2f} GiB"
                if args.cuda_max_memory_gb is not None
                else f"fraction {fraction:.3f}"
            )
            print(f"[MEM] CUDA memory limit : {limit_str}  (across {n_devices} device(s))")

        if args.cuda_reserve_gb is not None and args.cuda_reserve_gb > 0:
            reserve_bytes = int(args.cuda_reserve_gb * (1024 ** 3))
            reserve_mib   = reserve_bytes // (1024 ** 2)
            dev_idx = args.device.index or 0
            try:
                t = torch.empty(
                    reserve_mib * 1024 * 256, dtype=torch.uint8, device=args.device
                )
                del t
                torch.cuda.empty_cache()
                print(f"[MEM] CUDA reserved pool  : {args.cuda_reserve_gb:.2f} GiB "
                      f"on device {dev_idx}")
            except RuntimeError as exc:
                print(f"[WARN] CUDA reservation failed ({exc}); continuing without it.")

    if args.cpu_max_memory_gb is not None:
        if args.cpu_max_memory_gb <= 0:
            raise ValueError("--cpu_max_memory_gb must be > 0")
        try:
            import resource
            limit_bytes = int(args.cpu_max_memory_gb * (1024 ** 3))
            overhead    = 4 * (1024 ** 3)
            hard_bytes  = limit_bytes + overhead
            resource.setrlimit(resource.RLIMIT_AS, (hard_bytes, resource.RLIM_INFINITY))
            print(
                f"[MEM] CPU address-space limit : {args.cpu_max_memory_gb:.2f} GiB "
                f"(+{overhead // (1024**3)} GiB overhead = "
                f"{hard_bytes / (1024**3):.2f} GiB RLIMIT_AS)"
            )
        except ImportError:
            print("[WARN] --cpu_max_memory_gb ignored: 'resource' module not available (Windows).")
        except ValueError as exc:
            print(f"[WARN] --cpu_max_memory_gb could not be applied: {exc}")