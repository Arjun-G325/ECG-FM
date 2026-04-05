# ECG Foundation Model

Encoder-decoder transformer with contrastive pre-training and multi-task fine-tuning for 12-lead ECG signals.

## Project Structure

```
ecg_foundation/
├── main.py              # entry point — orchestrates all stages
├── args.py              # argument parser (model, training, hardware)
├── loss.py              # NTXentLoss, FinetuneLoss
├── requirements.txt
│
├── data/
│   ├── __init__.py
│   └── dataset.py       # SyntheticECGDataset + augment_ecg()
│
├── models/
│   ├── __init__.py
│   ├── backbone.py      # CNNStem, TransformerEncoder/Decoder, ECGFoundationModel
│   └── finetune.py      # ECGFinetuneModel (task heads)
│
└── training/
    ├── __init__.py
    ├── pretrain.py      # contrastive pre-training loop
    └── finetune.py      # multi-task fine-tuning loop
```

## Quick Start

```bash
pip install -r requirements.txt

# auto-detect device, synthetic data
python main.py

# explicit GPU
python main.py --device cuda:0 --num_workers 4

# multi-GPU (DataParallel)
python main.py --cuda_devices 0,1,2,3 --multi_gpu

# CPU only, pin threads
python main.py --device cpu --num_cores 8 --num_interop_cores 2

# freeze backbone during fine-tuning
python main.py --freeze_backbone --pretrain_epochs 20 --finetune_epochs 10

```
## Hardware / Compute Flags
# Limit to 6 GiB GPU, 24 GiB CPU RAM
python main.py --cuda_max_memory_gb 6 --cpu_max_memory_gb 24

# Use 80% of GPU memory, pre-reserve 3 GiB
python main.py --cuda_memory_fraction 0.80 --cuda_reserve_gb 3

# CPU-only run, cap at 8 GiB RAM
python main.py --device cpu --cpu_max_memory_gb 8

| Flag | Default | Description |
|---|---|---|
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0`, `cuda:1`, … |
| `--cuda_devices` | all | Comma-separated IDs → sets `CUDA_VISIBLE_DEVICES`. E.g. `0,2` |
| `--multi_gpu` | off | Wrap model in `DataParallel` across all visible GPUs |
| `--num_workers` | `0` | DataLoader worker processes |
| `--num_cores` | auto | Intra-op CPU threads (`torch.set_num_threads`) |
| `--num_interop_cores` | auto | Inter-op CPU threads (`torch.set_num_interop_threads`) |
| `--pin_memory` | auto | Pin DataLoader memory (default True on CUDA) |
| `--fp16` | off | Force AMP on CPU (experimental) |
| `--tf32` / `--no-tf32` | on | TF32 matmuls on Ampere+ GPUs |
| `--cudnn_benchmark` / `--no-cudnn_benchmark` | on | cuDNN auto-tuner (best for fixed input sizes) |

## Pipeline

```
SyntheticECGDataset  (B, 12, 5000)
        │
   augment_ecg() ──► view1, view2
        │
   ECGFoundationModel
   ├── CNNStem              (B,12,5000) → (B,256,250)
   ├── TransformerEncoder   (B,250,256) → (B,250,256)
   ├── TransformerDecoder   (B,250,256) → (B,16,256)   cross-attention summary
   └── ProjectionHead       (B,256)     → (B,128)      L2-normalised
        │
   NTXentLoss  (pre-training)
        │
   ECGFinetuneModel
   └── heads: age, gender, bmi, sbp, dbp
        │
   FinetuneLoss  (MSE + CrossEntropy, per-task weights)
```
