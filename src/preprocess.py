"""
preprocess.py
--------------
Handles image preprocessing, stratified train/val/test splitting,
age normalization, and reproducibility setup for the UTKFace dataset.
"""

import os
import gc
import cv2
import numpy as np
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split

def preprocess_images_to_memmap(images, labels, target_size=(160, 160), save_X_path=None, save_y_path=None):
    """
    Resize images and save to memmap files on disk.
    - images: list of numpy arrays (RGB uint8)
    - labels: list of dicts, each must contain 'age' key
    - save_X_path, save_y_path: exact paths for memmap files (required)
    """
    if save_X_path is None or save_y_path is None:
        raise ValueError("Must provide save_X_path and save_y_path")

    os.makedirs(os.path.dirname(save_X_path), exist_ok=True)

    n = len(images)
    w, h = target_size
    print(f"Writing memmap files ({w}x{h}) ... (n={n})")

    if os.path.exists(save_X_path) and os.path.exists(save_y_path):
        print(f"Memmap files already exist:\n  {save_X_path}\n  {save_y_path}")
        return save_X_path, save_y_path

    # Create writable memmaps
    X_mm = np.memmap(save_X_path, dtype=np.float16, mode="w+", shape=(n, h, w, 3))
    y_mm = np.memmap(save_y_path, dtype=np.float32, mode="w+", shape=(n,))

    for i, img in enumerate(images):
        if img is None:
            raise ValueError(f"Image at index {i} is None")
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        X_mm[i] = (resized.astype(np.float32) / 255.0).astype(np.float16)
        y_mm[i] = float(labels[i]["age"])
        if (i + 1) % 1000 == 0 or (i + 1) == n:
            print(f"  Processed {i + 1}/{n} images", end="\r")

    del X_mm, y_mm
    gc.collect()
    print(f"\nMemmap creation finished for {w}x{h}.")


def load_and_split_from_memmap(X_path, y_path, n_samples, target_size=(160,160), normalize_01=True, random_seed=42,
                               age_bins=None, save_split_info=True,
                               split_info_path="./dataset_split/dataset_split_info.pkl"):
    """
    Load memmap files and perform stratified train/val/test splits.
    Returns the dictionary of objects (same interface you used before),
    but X_train/X_val/X_test are memmap slices (views) — no full-copy in RAM.
    """
    # Cleanup
    for name in ["X", "X_tmp", "X_train", "X_val", "X_test", "dbg_X", "dbg_y", "dbg_w"]:
        if name in globals():
            del globals()[name]
    gc.collect()

    rng = np.random.default_rng(random_seed)
    NORMALIZE_01 = normalize_01
    AGE_BINS = age_bins or [0, 5, 12, 18, 30, 45, 60, 80, 200]

    # Map memmaps read-only (float16 normalized)
    X_all = np.memmap(X_path, dtype=np.float16, mode="r", shape=(n_samples, target_size[1], target_size[0], 3))
    y_all = np.memmap(y_path, dtype=np.float32, mode="r", shape=(n_samples,))

    # Extract ages as numpy array (this does not copy the whole memmap; small view)
    ages = np.array(y_all, dtype=np.float32) # ages fits in RAM (one scalar per image)

    # Stratified splits using age bins
    bins_idx = np.digitize(ages, AGE_BINS, right=False) - 1
    idx_all = np.arange(len(ages))

    idx_tmp, idx_test, ages_tmp, ages_test, bins_tmp, bins_test = train_test_split(
        idx_all, ages, bins_idx,
        test_size=0.15, random_state=random_seed, stratify=bins_idx
    )

    val_ratio_of_remaining = 0.15 / (1.0 - 0.15)
    idx_train, idx_val, ages_train, ages_val, bins_train, bins_val = train_test_split(
        idx_tmp, ages_tmp, bins_tmp,
        test_size=val_ratio_of_remaining, random_state=random_seed, stratify=bins_tmp
    )

    # IMPORTANT: slice the memmap (these are views, not full copies)
    X_train = X_all[idx_train]
    X_val   = X_all[idx_val]
    X_test  = X_all[idx_test]

    # numeric label arrays
    y_train = ages_train
    y_val   = ages_val
    y_test  = ages_test

    # Standardize ages (small arrays in RAM)
    age_mean = y_train.mean()
    age_std = y_train.std()

    y_train_std = (y_train - age_mean) / age_std
    y_val_std = (y_val - age_mean) / age_std
    y_test_std = (y_test - age_mean) / age_std

    print(f"Age mean: {age_mean:.2f}, std: {age_std:.2f}")

    # NOTE: X_train etc are still float16 normalized [0,1] if preprocessed that way.
    # If caller wants float32 normalized arrays, they should convert per-batch (not whole arrays).
    print("Train/Val/Test memmap slice shapes:", X_train.shape, X_val.shape, X_test.shape)
    print("Bin counts (train):", Counter(bins_train))
    print("Bin counts (val):  ", Counter(bins_val))
    print("Bin counts (test): ", Counter(bins_test))

    # Save split info
    if save_split_info:
        split_info = {
            "idx_train": idx_train,
            "idx_val": idx_val,
            "idx_test": idx_test,
            "age_mean": float(age_mean),
            "age_std": float(age_std),
            "y_all": ages,
            "AGE_BINS": AGE_BINS,
            "RANDOM_SEED": random_seed
        }
        os.makedirs(os.path.dirname(split_info_path), exist_ok=True)
        with open(split_info_path, "wb") as f:
            pickle.dump(split_info, f)
        print(f"Split info saved to: {split_info_path}")

    return {
        "rng": rng,
        "NORMALIZE_01": NORMALIZE_01,
        "AGE_BINS": AGE_BINS,
        "bins_train": bins_train,
        "bins_val": bins_val,
        "bins_test": bins_test,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "age_mean": age_mean,
        "age_std": age_std,
        "y_train_std": y_train_std,
        "y_val_std": y_val_std,
        "y_test_std": y_test_std,
    }
