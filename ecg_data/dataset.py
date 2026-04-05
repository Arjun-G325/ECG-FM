"""
ecg_data/dataset.py — ECG Dataset  (v3)

Key fix from v2
───────────────
  v2 loaded the full 13 GB array into RAM at __init__ time → system hang.

  v3 uses mmap_mode='r' — the OS pages in only the rows being accessed.
  Per-sample normalisation moves to __getitem__ so no full array is ever
  resident in RAM.  RAM usage ≈ batch_size × 12 × seq_len × 4 bytes.
  For batch=64, seq_len=5000 that is ~15 MB per batch.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class IndianECGDataset(Dataset):
    """
    Memory-mapped loader for ecg_55k_X_clean.npy + ecg_55k_y_clean.npy.

    The signal array is never fully loaded into RAM.  Only pages for the
    current batch are resident at any time.
    """

    X_FILE = "ecg_55k_X_clean.npy"
    Y_FILE = "ecg_55k_y_clean.npy"

    def __init__(
        self,
        data_dir:  str  = "ecg_data/Indian_ECG_175k",
        seq_len:   int  = 5000,
        n_leads:   int  = 12,
        normalise: bool = True,
        split:     str  = "all",
        seed:      int  = 42,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.n_leads   = n_leads
        self.normalise = normalise

        x_path = os.path.join(data_dir, self.X_FILE)
        y_path = os.path.join(data_dir, self.Y_FILE)
        for p in (x_path, y_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Not found: {p}")

        # mmap_mode='r' — file stays on disk, OS loads pages on demand
        print(f"[Dataset] Memory-mapping {x_path} ...")
        X_mmap = np.load(x_path, mmap_mode='r')
        Y_raw  = np.load(y_path, allow_pickle=True)   # labels are small — fine to load

        print(f"[Dataset] X shape : {X_mmap.shape}  dtype={X_mmap.dtype}")
        print(f"[Dataset] Y shape : {Y_raw.shape}   dtype={Y_raw.dtype}")
        print(f"[Dataset] Disk    : {X_mmap.nbytes/1e9:.2f} GB  (NOT in RAM)")

        # detect layout from shape alone — no data read
        self._transpose, self._expand = self._layout_flags(X_mmap.shape, n_leads)

        # build split index — permutation of row indices only
        N   = X_mmap.shape[0]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
        if   split == "train": idx = idx[:int(0.8 * N)]
        elif split == "val":   idx = idx[int(0.8 * N):int(0.9 * N)]
        elif split == "test":  idx = idx[int(0.9 * N):]

        self._idx  = idx.astype(np.int64)
        self._mmap = X_mmap                  # keep mmap handle open

        # labels — small, safe to hold in RAM
        self.is_classification = self._detect_classification(Y_raw)

        if self.is_classification:
            Y_enc, self.class_names = self._encode_labels(Y_raw)
            self.num_classes = len(self.class_names)
            print(f"[Dataset] Task : classification  ({self.num_classes} classes)")
            for i, name in enumerate(self.class_names):
                c = int((Y_enc == i).sum())
                print(f"           {i:>3}  '{name}'  {c} ({100*c/N:.1f}%)")
            self.label = torch.from_numpy(Y_enc[idx]).long()
        else:
            Y_enc = Y_raw.astype(np.float32).ravel()
            self.num_classes = 1
            self.class_names = None
            print(f"[Dataset] Task : regression  "
                  f"min={Y_enc[idx].min():.2f}  max={Y_enc[idx].max():.2f}")
            self.target = torch.from_numpy(Y_enc[idx]).float()

        print(f"[Dataset] Split '{split}' → {len(self._idx)} samples")

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _layout_flags(shape, n_leads):
        if len(shape) == 2:
            return False, True       # (N, T) → expand lead dim
        _, a, b = shape
        if a == n_leads: return False, False   # (N, leads, T)
        if b == n_leads: return True,  False   # (N, T, leads)
        return (a > b), False                  # guess: smaller dim = leads

    @staticmethod
    def _detect_classification(Y):
        flat = Y.ravel()
        if np.issubdtype(flat.dtype, np.integer):  return True
        if flat.dtype.kind in ('U', 'S', 'O'):     return True
        if np.issubdtype(flat.dtype, np.floating):
            u = np.unique(flat.astype(float))
            if len(u) <= 50 and np.all(u == u.astype(int)): return True
        return False

    @staticmethod
    def _encode_labels(Y):
        flat    = Y.ravel().astype(str)
        classes = sorted(set(flat))
        mapping = {c: i for i, c in enumerate(classes)}
        encoded = np.array([mapping[v] for v in flat], dtype=np.int64)
        return encoded, classes

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self):
        return len(self._idx)

    def __getitem__(self, pos: int) -> dict:
        """Reads exactly one sample from disk (one OS page read)."""
        file_row = int(self._idx[pos])

        # np.array(...) materialises a single row and releases the mmap page
        x = np.array(self._mmap[file_row], dtype=np.float32)

        # fix layout → (n_leads, T)
        if self._expand:
            x = x[np.newaxis, :]
        elif self._transpose:
            x = x.T

        # truncate or zero-pad time axis
        T = x.shape[-1]
        if T > self.seq_len:
            x = x[:, :self.seq_len]
        elif T < self.seq_len:
            pad = np.zeros((x.shape[0], self.seq_len - T), dtype=np.float32)
            x   = np.concatenate([x, pad], axis=-1)

        # per-sample z-score (in-place, no extra allocation)
        if self.normalise:
            mu  = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1,  keepdims=True) + 1e-6
            x   = (x - mu) / std

        item = {"ecg": torch.from_numpy(x)}
        if self.is_classification:
            item["label"]  = self.label[pos]
        else:
            item["target"] = self.target[pos]
        return item


# ── Synthetic fallback ─────────────────────────────────────────────────────────

class ECGDataset(Dataset):
    """Synthetic dataset — used when no real data path is provided."""

    def __init__(self, n_samples=1000, n_leads=12, seq_len=5000, num_classes=5):
        super().__init__()
        signals = []
        for _ in range(n_samples):
            leads = []
            for _ in range(n_leads):
                base = 0.05 * np.random.randn(seq_len)
                for beat in np.arange(0.5, 10, np.random.uniform(0.7, 1.3)):
                    idx   = int(beat / 10 * seq_len)
                    width = np.random.randint(15, 35)
                    amp   = np.random.uniform(0.8, 1.5)
                    win   = np.arange(max(0, idx - width), min(seq_len, idx + width))
                    base[win] += amp * np.exp(-0.5 * ((win - idx) / (width / 3)) ** 2)
                leads.append(base)
            signals.append(leads)

        self.ecg         = torch.tensor(np.array(signals, dtype=np.float32))
        self.label       = torch.randint(0, num_classes, (n_samples,))
        self.is_classification = True
        self.num_classes = num_classes
        self.class_names = None

    def __len__(self):
        return len(self.ecg)

    def __getitem__(self, idx):
        return {"ecg": self.ecg[idx], "label": self.label[idx]}


# ── Augmentation stub (deprecated) ────────────────────────────────────────────

def augment_ecg(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Deprecated — not used by MAE pre-training."""
    import warnings
    warnings.warn("augment_ecg() is deprecated.", DeprecationWarning, stacklevel=2)
    B, L, T = x.shape
    def _aug(sig):
        sig      = sig + 0.02 * torch.randn_like(sig)
        scale    = torch.empty(B, 1, 1, device=sig.device).uniform_(0.8, 1.2)
        sig      = sig * scale
        crop_len = int(T * np.random.uniform(0.70, 0.95))
        start    = np.random.randint(0, T - crop_len)
        sig      = sig[:, :, start: start + crop_len]
        sig      = F.interpolate(sig, size=T, mode="linear", align_corners=False)
        mask_leads = np.random.choice(L, np.random.randint(1, 3), replace=False)
        sig[:, mask_leads, :] = 0.0
        return sig
    return _aug(x.clone()), _aug(x.clone())