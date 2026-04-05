"""
inspect_data.py — print shape, dtype, and sample statistics of the Indian ECG dataset

Run:
    python inspect_data.py
    python inspect_data.py --data_dir /path/to/Indian_ECG_175k
"""

import argparse
import numpy as np
import os

def inspect(path: str, name: str):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"  path  : {path}")
    arr = np.load(path, allow_pickle=True)
    print(f"  shape : {arr.shape}")
    print(f"  dtype : {arr.dtype}")
    print(f"  min   : {arr.min():.4f}")
    print(f"  max   : {arr.max():.4f}")
    print(f"  mean  : {arr.mean():.4f}")
    print(f"  std   : {arr.std():.4f}")
    print(f"  NaNs  : {np.isnan(arr.astype(float)).sum()}")

    if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] < 20):
        unique, counts = np.unique(arr, return_counts=True)
        if len(unique) <= 30:
            print(f"  unique classes ({len(unique)}):")
            for u, c in zip(unique, counts):
                print(f"    class {u:>6}  →  {c:>6} samples  ({100*c/len(arr):.1f}%)")
        else:
            print(f"  unique values  : {len(unique)}  (looks like regression target)")
            print(f"  first 5 values : {arr.flat[:5]}")

    print(f"  first sample shape: {arr[0].shape if arr.ndim > 1 else 'scalar'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=os.path.join(os.path.dirname(__file__), "ecg_data", "Indian_ECG_175k"),
    )
    args = parser.parse_args()

    x_path = os.path.join(args.data_dir, "ecg_55k_X_clean.npy")
    y_path = os.path.join(args.data_dir, "ecg_55k_y_clean.npy")

    inspect(x_path, "ecg_55k_X_clean.npy  (signals)")
    inspect(y_path, "ecg_55k_y_clean.npy  (labels)")
    print(f"\n{'─'*60}")
    print("  Paste the output above and share — then dataset.py will be written.")
    print(f"{'─'*60}\n")