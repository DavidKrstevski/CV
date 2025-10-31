import os
import math
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------
# Configuration
# ---------------------------
IMAGE_DIR = "data/images"
MODEL_PATH = "best_model/best_model.keras"
OUTPUT_DIR = "evaluation"
RANDOM_SEED = 84
SAMPLE_FRACTION = 0.15        # ~15% of all images
TARGET_SIZE = (160, 160)
NORMALIZE_01 = True
BATCH_SIZE = 32
AGE_BINS = [0, 5, 12, 18, 30, 45, 60, 80, 200]
SHOW_WORST_N = 8

# ---------------------------
# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# For reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid")

# You may want to add a simple print statement here instead of the logger.info line
print("Starting evaluation script.")

# ---------------------------
# Utility: parse filename to metadata
# filename format expected: age_gender_race_date(...).ext
# ---------------------------
def parse_filename_meta(fname):
    base = os.path.splitext(fname)[0]
    parts = base.split("_")
    if len(parts) < 3:
        return None
    try:
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
    except Exception:
        return None
    return {"filename": fname, "age": age, "gender": gender, "race": race}

# ---------------------------
# 1) Load model
# ---------------------------
print("Loading model: %s" % MODEL_PATH) # Replaced logger.info with print
model = load_model(MODEL_PATH)

# ensure model is built (helps Sequential models)
_dummy = np.zeros((1, TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.float32)
if NORMALIZE_01:
    _dummy /= 255.0
try:
    _ = model.predict(_dummy, verbose=0)
except Exception:
    # ignore if already built
    pass

# ---------------------------
# 2) Load test set using memmap + saved split
# ---------------------------
# Load split info
with open("dataset_split/dataset_split_info.pkl", "rb") as f:
    split_info = pickle.load(f)

idx_test = split_info["idx_test"]
age_mean = split_info["age_mean"]
age_std = split_info["age_std"]

# Paths for memmap files
size_tag = f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}"
X_path = f"data/memmap/X_resized_{size_tag}.dat"
y_path = f"data/memmap/y_resized_{size_tag}.dat"

# Infer total number of samples (from split indices)
n_samples = len(split_info["idx_train"]) + len(split_info["idx_val"]) + len(split_info["idx_test"])

# Load memmap arrays (read-only)
X_all = np.memmap(X_path, dtype=np.float16, mode="r",
                  shape=(n_samples, TARGET_SIZE[1], TARGET_SIZE[0], 3))
y_all = np.memmap(y_path, dtype=np.float32, mode="r", shape=(n_samples,))

# Extract test set
X_test = X_all[idx_test].astype(np.float32)
y_real = np.array(y_all[idx_test], dtype=np.float32)

# Standardize target (for models trained on standardized ages)
y_std = (y_real - age_mean) / age_std

print("Loaded test set from memmap: %d images", X_test.shape[0])

# Reconstruct metadata dataframe
meta_list = []
for i in idx_test:
    fname = sorted(os.listdir(IMAGE_DIR))[i]
    meta = parse_filename_meta(fname)
    if meta is not None:
        meta_list.append(meta)
meta_df = pd.DataFrame(meta_list)

# ---------------------------
# 3) Model predictions (no augmentation)
# ---------------------------
print("Predicting on test set...")
y_pred_std = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1).flatten()

# Denormalize predictions back to real ages
y_pred = y_pred_std * age_std + age_mean

# Basic metrics (use real ages)
mae = mean_absolute_error(y_real, y_pred)
mse = mean_squared_error(y_real, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_real, y_pred)

print("=== Basic Metrics (Real Ages) ===")
print("MAE  : %.4f years", mae)
print("MSE  : %.4f", mse)
print("RMSE : %.4f", rmse)
print("R²   : %.4f", r2)

# Save CSV of predictions
results_df = meta_df.copy()
results_df["true_age_real"] = y_real
results_df["true_age_std"] = y_std
results_df["pred_age_real"] = y_pred
results_df["pred_age_std"] = y_pred_std
results_df["error_real"] = y_pred - y_real
results_df["abs_error_real"] = np.abs(results_df["error_real"])

csv_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
results_df.to_csv(csv_path, index=False)
print("Saved predictions to %s", csv_path)

# ---------------------------
# 4) Visualizations & Analyses (saved to OUTPUT_DIR)
# ---------------------------
def savefig(pth):
    try:
        plt.savefig(pth, dpi=150, bbox_inches="tight")
        print("Saved plot: %s", pth)
    except Exception as e:
        print("Failed to save plot %s: %s", pth, str(e))
    finally:
        plt.close()

# 4a: Scatter plot — Predicted vs True (real ages)
plt.figure(figsize=(7,7))
plt.scatter(y_real, y_pred, s=20, alpha=0.5, edgecolor="w", linewidth=0.7)
lims = [0, max(y_real.max(), y_pred.max()) + 5]
plt.plot(lims, lims, 'k--', alpha=0.7, label="Ideal = Perfect Prediction")
plt.xlim(lims)
plt.ylim(lims)
plt.xlabel("True Age (years)", fontsize=12)
plt.ylabel("Predicted Age (years)", fontsize=12)
plt.title("Predicted vs True Age (Test Set)", fontsize=14, weight="bold")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
savefig(os.path.join(OUTPUT_DIR, "pred_vs_true_scatter.png"))

# 4a (alt): Hexbin density
plt.figure(figsize=(7,7))
hb = plt.hexbin(y_real, y_pred, gridsize=50, cmap="Reds", mincnt=1)
cb = plt.colorbar(hb)
cb.set_label("Counts")
plt.plot(lims, lims, 'k--', alpha=0.7)
plt.xlim(lims)
plt.ylim(lims)
plt.xlabel("True Age (years)", fontsize=12)
plt.ylabel("Predicted Age (years)", fontsize=12)
plt.title("Predicted vs True Age (Hexbin Density)", fontsize=14, weight="bold")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
savefig(os.path.join(OUTPUT_DIR, "pred_vs_true_density.png"))

# 4b: Residuals vs true age (real age units)
residuals = y_pred - y_real
plt.figure(figsize=(8,5))
plt.scatter(y_real, residuals, alpha=0.5, s=20, edgecolor="w")
plt.axhline(0, color='k', linestyle='--', alpha=0.6)
plt.xlabel("True Age (years)", fontsize=12)
plt.ylabel("Prediction Error (Predicted - True) [years]", fontsize=12)
plt.title("Residuals vs True Age", fontsize=14, weight="bold")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
savefig(os.path.join(OUTPUT_DIR, "residuals_vs_age.png"))

# 4c: Residual histogram (real-age residuals)
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=40, kde=True, color="#FF8A65")
plt.xlabel("Residual (Predicted - True) [years]", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Distribution of Prediction Residuals (Errors in Years)", fontsize=14, weight="bold")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
savefig(os.path.join(OUTPUT_DIR, "residual_histogram.png"))

# 4d: Mean Absolute Error per real-age bin
bin_idx = np.digitize(y_real, AGE_BINS, right=False) - 1
bin_labels = []
bin_mae = []
for i in range(len(AGE_BINS)-1):
    lo, hi = AGE_BINS[i], AGE_BINS[i+1]
    mask = bin_idx == i
    if np.any(mask):
        bin_mae.append(np.mean(np.abs(y_pred[mask] - y_real[mask])))
    else:
        bin_mae.append(np.nan)
    if i < len(AGE_BINS)-2:
        bin_labels.append(f"{lo}–{hi-1}")
    else:
        bin_labels.append(f"{lo}+")
plt.figure(figsize=(10,5))
sns.barplot(x=bin_labels, y=bin_mae, palette="rocket")
plt.xlabel("Age Bin (years)", fontsize=12)
plt.ylabel("Mean Absolute Error (years)", fontsize=12)
plt.title("Mean Absolute Error per Age Bin", fontsize=14, weight="bold")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
savefig(os.path.join(OUTPUT_DIR, "mae_per_age_bin.png"))

# 4e: Hardest / worst predictions (real-age values)
worst_idx = np.argsort(-results_df["abs_error_real"].values)[:SHOW_WORST_N]
plt.figure(figsize=(12, 6))
for i, idx in enumerate(worst_idx):
    ax = plt.subplot(2, (SHOW_WORST_N+1)//2, i+1)
    img = X_test[idx]
    if NORMALIZE_01:
        img = np.clip(img, 0, 1)
    ax.imshow(img)
    true_age = results_df.loc[idx, "true_age_real"]
    pred_age = results_df.loc[idx, "pred_age_real"]
    abs_err = results_df.loc[idx, "abs_error_real"]
    ax.set_title(f"T:{true_age:.1f} / P:{pred_age:.1f}\nΔ={abs_err:.1f}", fontsize=10)
    ax.axis("off")
plt.suptitle("Hardest Predictions (Largest Absolute Errors in Years)", fontsize=14, weight="bold")
plt.tight_layout()
savefig(os.path.join(OUTPUT_DIR, "hardest_predictions.png"))

# 4f: Summary statistics file (real-age evaluation)
summary_path = os.path.join(OUTPUT_DIR, "results_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=== Test Set Evaluation Summary (Real Ages) ===\n")
    f.write(f"Total images evaluated: {len(y_real)}\n")
    f.write(f"MAE  : {mae:.6f} years\n")
    f.write(f"MSE  : {mse:.6f} (years²)\n")
    f.write(f"RMSE : {rmse:.6f} years\n")
    f.write(f"R²   : {r2:.6f}\n")

print("Saved summary to %s", summary_path)

# ---------------------------
# 5) Demographic Bias Analysis (Gender & Race)
# ---------------------------
gender_map = {0: "Male", 1: "Female"}
race_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}

demo_df = results_df.copy()

# Ensure gender and race metadata exist
if "gender" not in demo_df.columns or "race" not in demo_df.columns:
    if "gender" in meta_df.columns and "race" in meta_df.columns:
        demo_df["gender"] = meta_df["gender"].values
        demo_df["race"] = meta_df["race"].values
    else:
        print("Gender/race metadata missing — demographic bias plots may be incomplete.")

# Human-readable labels
demo_df["gender_str"] = demo_df["gender"].map(gender_map).fillna(demo_df["gender"].astype(str))
demo_df["race_str"] = demo_df["race"].map(race_map).fillna(demo_df["race"].astype(str))

# --- Gender-level statistics ---
gender_stats = demo_df.groupby("gender_str").agg(
    MAE=("abs_error_real", "mean"),
    RMSE=("abs_error_real", lambda x: math.sqrt(np.mean(x**2))),
    Count=("abs_error_real", "count")
).reset_index()

gender_stats_path = os.path.join(OUTPUT_DIR, "gender_error_stats.csv")
gender_stats.to_csv(gender_stats_path, index=False)
print("Saved gender error stats to %s", gender_stats_path)

plt.figure(figsize=(6,5))
sns.barplot(data=gender_stats, x="gender_str", y="MAE", palette=["#F9C5D5", "#F48FB1"])
plt.xlabel("Gender")
plt.ylabel("MAE (years)")
plt.title("Mean Absolute Error by Gender (Real Ages)")
plt.tight_layout()
savefig(os.path.join(OUTPUT_DIR, "mae_by_gender.png"))

# --- Race-level statistics ---
race_stats = demo_df.groupby("race_str").agg(
    MAE=("abs_error_real", "mean"),
    RMSE=("abs_error_real", lambda x: math.sqrt(np.mean(x**2))),
    Count=("abs_error_real", "count")
).reset_index()

race_stats_path = os.path.join(OUTPUT_DIR, "race_error_stats.csv")
race_stats.to_csv(race_stats_path, index=False)
print("Saved race error stats to %s", race_stats_path)

plt.figure(figsize=(8,5))
sns.barplot(data=race_stats, x="race_str", y="MAE", palette="mako")
plt.xlabel("Race")
plt.ylabel("MAE (years)")
plt.title("Mean Absolute Error by Race (Real Ages)")
plt.xticks(rotation=15)
plt.tight_layout()
savefig(os.path.join(OUTPUT_DIR, "mae_by_race.png"))