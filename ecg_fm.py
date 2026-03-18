# ECG Foundation Model
# encoder-decoder with cross-attention and contrastive pre-training
# fine-tuning heads for age, gender, BMI, systolic/diastolic BP
# data shape: (1000, 12, 5000) -> batch, leads, timesteps

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


# argument parser
def get_args():
    parser = argparse.ArgumentParser(
        description="ECG Foundation Model — Pre-training + Fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # dataset
    data = parser.add_argument_group("Dataset")
    data.add_argument("--n_samples", type=int, default=1000, help="number of ECG samples")
    data.add_argument("--n_leads",   type=int, default=12,   help="number of ECG leads")
    data.add_argument("--seq_len",   type=int, default=5000, help="timesteps per sample")

    # model architecture
    arch = parser.add_argument_group("Model Architecture")
    arch.add_argument("--d_model",        type=int,   default=256,  help="transformer hidden dim")
    arch.add_argument("--nhead",          type=int,   default=8,    help="attention heads (d_model must be divisible by nhead)")
    arch.add_argument("--enc_layers",     type=int,   default=4,    help="encoder layers")
    arch.add_argument("--dec_layers",     type=int,   default=2,    help="decoder layers")
    arch.add_argument("--dim_ff",         type=int,   default=512,  help="feedforward dim")
    arch.add_argument("--dropout",        type=float, default=0.1,  help="dropout rate")
    arch.add_argument("--n_query_tokens", type=int,   default=16,   help="decoder query tokens")
    arch.add_argument("--proj_dim",       type=int,   default=128,  help="contrastive projection dim")

    # pre-training
    pre = parser.add_argument_group("Pre-training")
    pre.add_argument("--pretrain_epochs", type=int,   default=5,    help="pre-training epochs")
    pre.add_argument("--pretrain_lr",     type=float, default=1e-4, help="pre-training learning rate")
    pre.add_argument("--temperature",     type=float, default=0.07, help="NT-Xent temperature")

    # fine-tuning
    fine = parser.add_argument_group("Fine-tuning")
    fine.add_argument("--finetune_epochs",  type=int,   default=5,    help="fine-tuning epochs")
    fine.add_argument("--finetune_lr",      type=float, default=3e-5, help="fine-tuning learning rate")
    fine.add_argument("--freeze_backbone",  action="store_true",      help="freeze backbone during fine-tuning")

    # training
    train = parser.add_argument_group("Training")
    train.add_argument("--batch_size",   type=int,   default=16,   help="batch size")
    train.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    train.add_argument("--grad_clip",    type=float, default=1.0,  help="gradient clip norm")
    train.add_argument("--num_workers",  type=int,   default=0,    help="dataloader workers (0 = safe on Windows)")

    # misc
    misc = parser.add_argument_group("Misc")
    misc.add_argument("--seed",     type=int, default=42,   help="random seed")
    misc.add_argument("--save_dir", type=str, default=None, help="checkpoint save directory (default: script folder)")

    return parser.parse_args()


# synthetic ECG dataset
# replace this with your real dataset when ready
class SyntheticECGDataset(Dataset):
    def __init__(self, n_samples=1000, n_leads=12, seq_len=5000):
        super().__init__()
        self.n_samples = n_samples

        # build fake ECG signals with gaussian noise and pseudo QRS bumps
        signals = []
        for _ in range(n_samples):
            lead_signals = []
            for _ in range(n_leads):
                base = 0.05 * np.random.randn(seq_len)
                for beat in np.arange(0.5, 10, np.random.uniform(0.7, 1.3)):
                    idx   = int(beat / 10 * seq_len)
                    width = np.random.randint(15, 35)
                    amp   = np.random.uniform(0.8, 1.5)
                    win   = np.arange(max(0, idx - width), min(seq_len, idx + width))
                    base[win] += amp * np.exp(-0.5 * ((win - idx) / (width / 3)) ** 2)
                lead_signals.append(base)
            signals.append(lead_signals)

        self.ecg = torch.tensor(np.array(signals, dtype=np.float32))  # (N, 12, 5000)

        # random labels — not realistic, just for demo
        self.age    = torch.rand(n_samples) * 60 + 18       # 18-78
        self.gender = torch.randint(0, 2, (n_samples,))     # 0=M 1=F
        self.bmi    = torch.rand(n_samples) * 25 + 15       # 15-40
        self.sbp    = torch.rand(n_samples) * 80 + 90       # systolic 90-170
        self.dbp    = torch.rand(n_samples) * 50 + 60       # diastolic 60-110

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "ecg":    self.ecg[idx],
            "age":    self.age[idx],
            "gender": self.gender[idx],
            "bmi":    self.bmi[idx],
            "sbp":    self.sbp[idx],
            "dbp":    self.dbp[idx],
        }


# augmentations for contrastive learning
# returns two different views of the same batch
def augment_ecg(x: torch.Tensor):
    B, L, T = x.shape

    def _aug(sig):
        # gaussian noise
        sig = sig + 0.02 * torch.randn_like(sig)
        # random amplitude scale
        scale = torch.empty(B, 1, 1, device=sig.device).uniform_(0.8, 1.2)
        sig = sig * scale
        # random time crop then resize back
        crop_len = int(T * np.random.uniform(0.7, 0.95))
        start    = np.random.randint(0, T - crop_len)
        sig = sig[:, :, start:start + crop_len]
        sig = F.interpolate(sig, size=T, mode="linear", align_corners=False)
        # randomly zero out 1-2 leads
        n_mask     = np.random.randint(1, 3)
        mask_leads = np.random.choice(L, n_mask, replace=False)
        sig[:, mask_leads, :] = 0.0
        return sig

    return _aug(x.clone()), _aug(x.clone())


# CNN stem: compresses (B, 12, 5000) -> (B, d_model, 250)
class CNNStem(nn.Module):
    def __init__(self, n_leads=12, d_model=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_leads, 64,     kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),  nn.GELU(),
            nn.Conv1d(64,  128,        kernel_size=9,  stride=2, padding=4),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, d_model,    kernel_size=5,  stride=5, padding=2),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )
        # 5000 -> 2500 -> 1250 -> 250

    def forward(self, x):
        return self.net(x)


# sinusoidal positional encoding
class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# transformer encoder
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_ff=512, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        # enable_nested_tensor=False avoids the norm_first warning
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.pe = SinusoidalPE(d_model)

    def forward(self, x):
        return self.encoder(self.pe(x))


# transformer decoder with cross-attention
# uses learnable query tokens to summarise the encoder memory
class TransformerDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=2, dim_ff=512,
                 dropout=0.1, n_query_tokens=16):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, n_query_tokens, d_model))
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.pe = SinusoidalPE(d_model)

    def forward(self, memory):
        B = memory.size(0)
        q = self.pe(self.query_tokens.expand(B, -1, -1))
        return self.decoder(q, memory)


# projection head used only during contrastive pre-training
class ProjectionHead(nn.Module):
    def __init__(self, d_model=256, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# full foundation model
# forward() -> used during pre-training (returns projection)
# get_representation() -> used during fine-tuning (returns feature vector)
class ECGFoundationModel(nn.Module):
    def __init__(self, n_leads=12, d_model=256, nhead=8, enc_layers=4,
                 dec_layers=2, dim_ff=512, dropout=0.1,
                 n_query_tokens=16, proj_dim=128):
        super().__init__()
        self.stem    = CNNStem(n_leads, d_model)
        self.encoder = TransformerEncoder(d_model, nhead, enc_layers, dim_ff, dropout)
        self.decoder = TransformerDecoder(d_model, nhead, dec_layers, dim_ff, dropout, n_query_tokens)
        self.proj    = ProjectionHead(d_model, proj_dim)
        self.d_model = d_model

    def encode(self, x):
        # (B, 12, 5000) -> (B, 250, d_model)
        return self.encoder(self.stem(x).permute(0, 2, 1))

    def get_representation(self, x):
        # (B, 12, 5000) -> (B, d_model) single vector
        return self.decoder(self.encode(x)).mean(dim=1)

    def forward(self, x):
        # (B, 12, 5000) -> (B, proj_dim) normalised for contrastive loss
        return self.proj(self.get_representation(x))


# NT-Xent loss (SimCLR style)
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature

    def forward(self, z1, z2):
        B   = z1.size(0)
        z   = torch.cat([z1, z2], dim=0)          # (2B, d)
        sim = torch.mm(z, z.T) / self.T            # (2B, 2B)

        # mask out self-similarity
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e4)               # -1e4 is safe for fp16

        # positives: each sample paired with its augmented twin
        labels = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)
        return F.cross_entropy(sim, labels)


# fine-tuning model with task heads on top of the backbone
class ECGFinetuneModel(nn.Module):
    def __init__(self, foundation: ECGFoundationModel, freeze_backbone=False):
        super().__init__()
        self.backbone = foundation
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        d = foundation.d_model
        self.head_age    = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_gender = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 2))
        self.head_bmi    = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_sbp    = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_dbp    = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        rep = self.backbone.get_representation(x)
        return {
            "age":    self.head_age(rep).squeeze(-1),
            "gender": self.head_gender(rep),
            "bmi":    self.head_bmi(rep).squeeze(-1),
            "sbp":    self.head_sbp(rep).squeeze(-1),
            "dbp":    self.head_dbp(rep).squeeze(-1),
        }


# pre-training loop
def pretrain(model, loader, args, device):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs)
    criterion = NTXentLoss(temperature=args.temperature)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            ecg = batch["ecg"].to(device)
            view1, view2 = augment_ecg(ecg)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = criterion(model(view1), model(view2))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(loader)
        print(f"  [Pretrain] Epoch {epoch}/{args.pretrain_epochs}  loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    return model


# fine-tuning loop
def finetune(ft_model, loader, args, device):
    ft_model.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, ft_model.parameters()),
        lr=args.finetune_lr, weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    mse = nn.MSELoss()
    ce  = nn.CrossEntropyLoss()

    for epoch in range(1, args.finetune_epochs + 1):
        ft_model.train()
        total_loss = 0.0
        for batch in loader:
            ecg    = batch["ecg"].to(device)
            age    = batch["age"].to(device)
            gender = batch["gender"].to(device)
            bmi    = batch["bmi"].to(device)
            sbp    = batch["sbp"].to(device)
            dbp    = batch["dbp"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds = ft_model(ecg)
                loss  = (
                    mse(preds["age"],    age)    +
                    ce( preds["gender"], gender) +
                    mse(preds["bmi"],    bmi)    +
                    mse(preds["sbp"],    sbp)    +
                    mse(preds["dbp"],    dbp)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ft_model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"  [Finetune] Epoch {epoch}/{args.finetune_epochs}  loss={avg:.4f}")

    return ft_model


# main
if __name__ == "__main__":

    args = get_args()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR = args.save_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"[INFO] Device   : {DEVICE}")
    print(f"[INFO] Save dir : {SAVE_DIR}")
    print(f"[INFO] Args     : {vars(args)}\n")

    # build dataset
    print(f"[1/4] Building synthetic ECG dataset ({args.n_samples} x {args.n_leads} x {args.seq_len})...")
    dataset = SyntheticECGDataset(n_samples=args.n_samples, n_leads=args.n_leads, seq_len=args.seq_len)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
    )
    print(f"      ECG tensor shape : {dataset.ecg.shape}")
    print(f"      Batches per epoch: {len(loader)}")

    # build model
    print("\n[2/4] Instantiating ECGFoundationModel...")
    foundation = ECGFoundationModel(
        n_leads        = args.n_leads,
        d_model        = args.d_model,
        nhead          = args.nhead,
        enc_layers     = args.enc_layers,
        dec_layers     = args.dec_layers,
        dim_ff         = args.dim_ff,
        dropout        = args.dropout,
        n_query_tokens = args.n_query_tokens,
        proj_dim       = args.proj_dim,
    )
    total_params = sum(p.numel() for p in foundation.parameters()) / 1e6
    print(f"      Parameters: {total_params:.2f} M")

    # sanity check output shape
    with torch.no_grad():
        z = foundation(torch.randn(2, args.n_leads, args.seq_len))
    print(f"      Projection output: {z.shape}  (expect [2, {args.proj_dim}])")

    # pre-train
    print(f"\n[3/4] Pre-training ({args.pretrain_epochs} epochs)...")
    foundation = pretrain(foundation, loader, args, DEVICE)
    pretrain_path = os.path.join(SAVE_DIR, "ecg_pretrained.pt")
    torch.save(foundation.state_dict(), pretrain_path)
    print(f"      Saved -> {pretrain_path}")

    # fine-tune
    print(f"\n[4/4] Fine-tuning ({args.finetune_epochs} epochs)...")
    ft_model = ECGFinetuneModel(foundation, freeze_backbone=args.freeze_backbone)
    ft_model = finetune(ft_model, loader, args, DEVICE)
    finetune_path = os.path.join(SAVE_DIR, "ecg_finetuned.pt")
    torch.save(ft_model.state_dict(), finetune_path)
    print(f"      Saved -> {finetune_path}")

    # inference on 4 samples
    print("\n-- Inference demo --")
    ft_model.eval()
    sample_ecg = dataset.ecg[:4].to(DEVICE)

    actual_age    = dataset.age[:4].numpy().round(1)
    actual_gender = dataset.gender[:4].numpy()
    actual_bmi    = dataset.bmi[:4].numpy().round(1)
    actual_sbp    = dataset.sbp[:4].numpy().round(1)
    actual_dbp    = dataset.dbp[:4].numpy().round(1)

    with torch.no_grad():
        out = ft_model(sample_ecg)

    pred_age    = out["age"].cpu().numpy().round(1)
    pred_gender = out["gender"].cpu().argmax(dim=1).numpy()
    pred_bmi    = out["bmi"].cpu().numpy().round(1)
    pred_sbp    = out["sbp"].cpu().numpy().round(1)
    pred_dbp    = out["dbp"].cpu().numpy().round(1)

    print(f"\n{'':12} {'Sample 1':>12} {'Sample 2':>12} {'Sample 3':>12} {'Sample 4':>12}")
    print("-" * 60)

    rows = [
        ("age   act",   actual_age),
        ("age   pred",  pred_age),
        ("gender act",  actual_gender),
        ("gender pred", pred_gender),
        ("bmi   act",   actual_bmi),
        ("bmi   pred",  pred_bmi),
        ("sbp   act",   actual_sbp),
        ("sbp   pred",  pred_sbp),
        ("dbp   act",   actual_dbp),
        ("dbp   pred",  pred_dbp),
    ]
    for i, (name, vals) in enumerate(rows):
        print(f"{name:12} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12}")
        if i in {1, 3, 5, 7}:
            print("-" * 60)