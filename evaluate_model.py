# evaluate_full_results_to_output.py
import os
import cv2
import sys
import atexit
import math
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------
# Configuration
# ---------------------------
IMAGE_DIR = "./images"
MODEL_PATH = "best_model.keras"
OUTPUT_DIR = "evaluate_model_output"
RANDOM_SEED = 84
SAMPLE_FRACTION = 0.15        # ~15% of all images
TARGET_SIZE = (224, 224)
NORMALIZE_01 = True
BATCH_SIZE = 32
AGE_BINS = [0, 5, 12, 18, 30, 45, 60, 80, 200]
SHOW_WORST_N = 8
USE_MCDO = True               # Monte Carlo Dropout uncertainty (set False to skip)
# ---------------------------

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up a "tee" for stdout/stderr so everything printed is also saved to a log file
log_path = os.path.join(OUTPUT_DIR, "run_log.txt")
_log_file = open(log_path, "w", encoding="utf-8")

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

# preserve originals
_original_stdout = sys.stdout
_original_stderr = sys.stderr

# replace with tee that writes to both console and log file
sys.stdout = Tee(_original_stdout, _log_file)
sys.stderr = Tee(_original_stderr, _log_file)

# register cleanup to restore streams and close file
def _cleanup():
    try:
        sys.stdout = _original_stdout
        sys.stderr = _original_stderr
    except Exception:
        pass
    try:
        _log_file.flush()
        _log_file.close()
    except Exception:
        pass

atexit.register(_cleanup)

# Also set up python logging (optional, logs to same file + console)
logger = logging.getLogger("evaluate_logger")
logger.setLevel(logging.INFO)
# Avoid adding multiple handlers if script re-imported
if not logger.handlers:
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(_original_stdout)  # still print to real console
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

# For reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid")

logger.info("Starting evaluation script. All console output will be saved to %s", log_path)

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
logger.info("Loading model: %s", MODEL_PATH)
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
# 2) Choose random subset (~15%) of available images and load them
# ---------------------------
all_files = sorted([f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))])
n_total = len(all_files)
subset_size = max(1, int(math.ceil(SAMPLE_FRACTION * n_total)))
logger.info("Found %d files, sampling %d (%.1f%%) for test subset.", n_total, subset_size, SAMPLE_FRACTION*100)

subset_files = np.random.choice(all_files, size=subset_size, replace=False)

X_list = []
y_list = []
meta_list = []
skipped = 0
for fname in subset_files:
    meta = parse_filename_meta(fname)
    if meta is None:
        skipped += 1
        continue
    img_path = os.path.join(IMAGE_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        skipped += 1
        continue
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        skipped += 1
        continue
    if img.ndim != 3 or img.shape[2] != 3:
        skipped += 1
        continue
    img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    X_list.append(img_resized)
    y_list.append(meta["age"])
    meta_list.append(meta)

if len(X_list) == 0:
    logger.error("No valid test images were loaded. Check IMAGE_DIR and filename format.")
    raise RuntimeError("No valid test images were loaded. Check IMAGE_DIR and filename format.")

X = np.stack(X_list, axis=0).astype(np.float32)
y = np.array(y_list, dtype=np.float32)
meta_df = pd.DataFrame(meta_list)

if NORMALIZE_01:
    X /= 255.0

logger.info("Loaded test subset: %d images (skipped %d)", X.shape[0], skipped)

# ---------------------------
# 3) Model predictions (no augmentation)
# ---------------------------
logger.info("Predicting on test subset...")
y_pred = model.predict(X, batch_size=BATCH_SIZE, verbose=1).flatten()

# Basic metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y, y_pred)

logger.info("=== Basic Metrics ===")
logger.info("MAE  : %.4f years", mae)
logger.info("MSE  : %.4f", mse)
logger.info("RMSE : %.4f", rmse)
logger.info("R²   : %.4f", r2)

# Save CSV of predictions
results_df = meta_df.copy()
results_df["true_age"] = y
results_df["pred_age"] = y_pred
results_df["error"] = y_pred - y
results_df["abs_error"] = np.abs(results_df["error"])
csv_path = os.path.join(OUTPUT_DIR, "test_predictions_subset.csv")
results_df.to_csv(csv_path, index=False)
logger.info("Saved predictions to %s", csv_path)

# ---------------------------
# 4) Visualizations & Analyses (saved to OUTPUT_DIR)
# ---------------------------
def savefig(pth):
    try:
        plt.savefig(pth, dpi=150)
        logger.info("Saved plot: %s", pth)
    except Exception as e:
        logger.exception("Failed to save plot %s: %s", pth, str(e))
    finally:
        plt.close()

# 4a: Scatter plot: predicted vs. true (with density)
plt.figure(figsize=(7,7))
plt.scatter(y, y_pred, s=20, alpha=0.5, edgecolor="w", linewidth=0.7)
lims = [0, max(y.max(), y_pred.max()) + 5]
plt.plot(lims, lims, 'k--', alpha=0.7, label="Ideal = Perfect Prediction")
plt.xlim(lims)
plt.ylim(lims)
plt.xlabel("True Age")
plt.ylabel("Predicted Age")
plt.title("Predicted vs True Age (Test Subset)", fontsize=14, weight="bold")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
savefig(os.path.join(OUTPUT_DIR, "pred_vs_true_scatter.png"))

plt.figure(figsize=(7,7))
hb = plt.hexbin(y, y_pred, gridsize=50, cmap="Reds", mincnt=1)
cb = plt.colorbar(hb)
cb.set_label("Counts")
plt.plot(lims, lims, 'k--', alpha=0.7)
plt.xlim(lims)
plt.ylim(lims)
plt.xlabel("True Age")
plt.ylabel("Predicted Age")
plt.title("Predicted vs True Age (Hexbin density)")
plt.tight_layout()
savefig(os.path.join(OUTPUT_DIR, "pred_vs_true_density.png"))

# 4b: Residuals vs true age
residuals = y_pred - y
plt.figure(figsize=(8,5))
plt.scatter(y, residuals, alpha=0.5, s=20, edgecolor="w")
plt.axhline(0, color='k', linestyle='--', alpha=0.6)
plt.xlabel("True Age")
plt.ylabel("Prediction Error (Residual) (Pred - True)")
plt.title("Residuals vs True Age")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
savefig(os.path.join(OUTPUT_DIR, "residuals_vs_age.png"))

# 4c: Residual histogram
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=40, kde=True)
plt.xlabel("Residual (years)")
plt.title("Distribution of Prediction Residuals (Errors)")
plt.tight_layout()
savefig(os.path.join(OUTPUT_DIR, "residual_histogram.png"))

# 4d: Mean Absolute Error per age bin
bin_idx = np.digitize(y, AGE_BINS, right=False) - 1
bin_labels = []
bin_mae = []
for i in range(len(AGE_BINS)-1):
    lo, hi = AGE_BINS[i], AGE_BINS[i+1]
    mask = bin_idx == i
    if np.any(mask):
        bin_mae.append(np.mean(np.abs(y_pred[mask] - y[mask])))
    else:
        bin_mae.append(np.nan)
    if i < len(AGE_BINS)-2:
        bin_labels.append(f"{lo}–{hi-1}")
    else:
        bin_labels.append(f"{lo}+")
plt.figure(figsize=(10,5))
sns.barplot(x=bin_labels, y=bin_mae, palette="rocket")
plt.xlabel("Age Bin")
plt.ylabel("MAE (years)")
plt.title("Mean Absolute Error per Age Bin")
plt.tight_layout()
savefig(os.path.join(OUTPUT_DIR, "mae_per_age_bin.png"))

# 4e: Hardest / worst predictions (images)
worst_idx = np.argsort(-results_df["abs_error"].values)[:SHOW_WORST_N]
plt.figure(figsize=(12, 6))
for i, idx in enumerate(worst_idx):
    ax = plt.subplot(2, (SHOW_WORST_N+1)//2, i+1)
    img = X[idx]
    if NORMALIZE_01:
        img = np.clip(img, 0, 1)
    ax.imshow(img)
    t = int(results_df.loc[idx, "true_age"])
    p = results_df.loc[idx, "pred_age"]
    ae = results_df.loc[idx, "abs_error"]
    ax.set_title(f"T:{t} / P:{p:.1f}\nΔ={ae:.1f}", fontsize=10)
    ax.axis("off")
plt.suptitle("Hardest Predictions (Largest Absolute Errors)")
plt.tight_layout()
savefig(os.path.join(OUTPUT_DIR, "hardest_predictions.png"))

# 4f: Summary statistics file (also saved)
summary_path = os.path.join(OUTPUT_DIR, "results_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=== Test subset evaluation summary ===\n")
    f.write(f"Total images evaluated: {len(y)}\n")
    f.write(f"MAE  : {mae:.6f}\n")
    f.write(f"MSE  : {mse:.6f}\n")
    f.write(f"RMSE : {rmse:.6f}\n")
    f.write(f"R2   : {r2:.6f}\n")
logger.info("Saved summary to %s", summary_path)

# ---------------------------
# 5) Demographic bias analysis (gender & race)
# ---------------------------
gender_map = {0: "Male", 1: "Female"}
race_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}

demo_df = results_df.copy()
if "gender" not in demo_df.columns or "race" not in demo_df.columns:
    demo_df["gender"] = meta_df["gender"].values
    demo_df["race"] = meta_df["race"].values

demo_df["gender_str"] = demo_df["gender"].map(gender_map).fillna(demo_df["gender"].astype(str))
demo_df["race_str"] = demo_df["race"].map(race_map).fillna(demo_df["race"].astype(str))

gender_stats = demo_df.groupby("gender_str").agg(
    MAE=("abs_error", "mean"),
    RMSE=("abs_error", lambda x: math.sqrt(np.mean(x**2))),
    Count=("abs_error", "count")
).reset_index()
gender_stats_path = os.path.join(OUTPUT_DIR, "gender_error_stats.csv")
gender_stats.to_csv(gender_stats_path, index=False)
logger.info("Saved gender error stats to %s", gender_stats_path)

plt.figure(figsize=(6,5))
sns.barplot(data=gender_stats, x="gender_str", y="MAE", palette=["#F9C5D5", "#F48FB1"])
plt.xlabel("Gender")
plt.ylabel("MAE (years)")
plt.title("Mean Absolute Error by Gender")
plt.tight_layout()
savefig(os.path.join(OUTPUT_DIR, "mae_by_gender.png"))

race_stats = demo_df.groupby("race_str").agg(
    MAE=("abs_error", "mean"),
    RMSE=("abs_error", lambda x: math.sqrt(np.mean(x**2))),
    Count=("abs_error", "count")
).reset_index()
race_stats_path = os.path.join(OUTPUT_DIR, "race_error_stats.csv")
race_stats.to_csv(race_stats_path, index=False)
logger.info("Saved race error stats to %s", race_stats_path)

plt.figure(figsize=(8,5))
sns.barplot(data=race_stats, x="race_str", y="MAE", palette="mako")
plt.xlabel("Race")
plt.ylabel("MAE (years)")
plt.title("Mean Absolute Error by Race")
plt.xticks(rotation=15)
plt.tight_layout()
savefig(os.path.join(OUTPUT_DIR, "mae_by_race.png"))

# ---------------------------
# 6) Calibration per age bin (mean predicted vs mean true)
# ---------------------------
demo_df["age_bin"] = pd.cut(
    demo_df["true_age"],
    bins=AGE_BINS,
    include_lowest=True,
    right=False,
    labels=[f"{AGE_BINS[i]}–{AGE_BINS[i+1]-1}" if i < len(AGE_BINS)-2 else f"{AGE_BINS[i]}+" for i in range(len(AGE_BINS)-1)]
)
calib = demo_df.groupby("age_bin", observed=False).agg(
    true_mean=("true_age", "mean"),
    pred_mean=("pred_age", "mean"),
    count=("true_age", "count")
).reset_index()

plt.figure(figsize=(7,6))
plt.scatter(calib["true_mean"], calib["pred_mean"], s=calib["count"]*2, c=calib["count"], cmap="Reds")
plt.plot(lims, lims, "k--", alpha=0.6)
plt.xlabel("Mean True Age (bin)")
plt.ylabel("Mean Predicted Age (bin)")
plt.title("Calibration: Predicted vs True (per age bin)")
plt.colorbar(label="Count")
plt.tight_layout()
savefig(os.path.join(OUTPUT_DIR, "calibration_age_bin.png"))
