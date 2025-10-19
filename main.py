# This file contains the newest and best code. The file called "best_model.keras" is the
# best model for this main.py file.
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#%% 1. Data Setup
image_dir = "./images"
images, labels = [], []

# Limit loading for speed (e.g., 50 random files)
all_files = os.listdir(image_dir)
random.shuffle(all_files)
max_images = 24108

for file_name in all_files[:max_images]:
    name = os.path.splitext(file_name)[0].replace(".chip", "")
    parts = name.split("_")

    # Expecting 4 parts: [age, gender, race, date]
    if len(parts) != 4 or any(p == "" for p in parts):
        continue

    try:
        age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
        date = parts[3]
    except ValueError:
        continue

    # Validate values
    if not (0 <= age <= 116 and gender in (0, 1) and 0 <= race <= 4 and len(date) == 17):
        continue

    img_path = os.path.join(image_dir, file_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"[Warning] Failed to read: {file_name}")
        continue

    # Convert from BGR (OpenCV) → RGB (matplotlib expects this)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Check image properties
    if img.ndim != 3 or img.shape[2] != 3:
        print(f"[Warning] Unexpected format for: {file_name}, shape={img.shape}")
        continue

    images.append(img)
    labels.append({
        "age": age,
        "gender": gender,
        "race": race,
        "datetime": date
    })

print(f"Loaded {len(images)} valid images out of {len(all_files)} files.")

#%% 1a. Dataset Summary Statistics
df = pd.DataFrame(labels)

print("\n=== Dataset Summary ===")
print(f"Total images: {len(df)}")
print(f"Age range: {df['age'].min()} - {df['age'].max()} (mean: {df['age'].mean():.1f})")
print(f"Gender distribution:\n{df['gender'].value_counts()}")
print(f"Race distribution:\n{df['race'].value_counts()}")
print(f"Average image size (HxW): {np.mean([img.shape[:2] for img in images], axis=0).astype(int)}")

#%% 1b. Label Replacement for Interpretability
gender_map = {0: "Male", 1: "Female"}
race_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}

df["gender"] = df["gender"].map(gender_map)
df["race"] = df["race"].map(race_map)

#%% 2. Visualization
# Create lists for easier reference
age_list = df["age"]
genders = df["gender"]
races = df["race"]

# Set theme and gentle palette
sns.set_theme(style="whitegrid")
rosa_palette = ["#F9C5D5", "#F7A1C4", "#F48FB1", "#F06292", "#EC407A"]

# --- Random sample visualization ---
plt.figure(figsize=(10, 6))
for i, idx in enumerate(random.sample(range(len(images)), min(9, len(images)))):
    img = images[idx]
    label = labels[idx]
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        f'Age: {label["age"]}\nGender: {gender_map[label["gender"]] if label["gender"] in gender_map else label["gender"]} '
        f'Race: {race_map[label["race"]] if label["race"] in race_map else label["race"]}',
        fontsize=10,
        color="#4A4A4A"
    )
plt.suptitle("Random Sample of UTKFace Images", fontsize=14, color="#333333", weight="bold")
plt.tight_layout()
plt.show()

# --- Age distribution histogram ---
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], color="#F48FB1", kde=True)
plt.title('Age Distribution', fontsize=14, color="#333333", weight="bold")
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# --- Gender balance ---
plt.figure(figsize=(6, 5))
ax = sns.countplot(data=df, x="gender", hue="gender",
                   palette=["#F9C5D5", "#F48FB1"], legend=False)
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', fontsize=10)
plt.title("Gender Balance", fontsize=14, color="#333333", weight="bold")
plt.xlabel("Gender", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# --- Race balance ---
plt.figure(figsize=(7, 5))
ax = sns.countplot(data=df, x="race", hue="race", palette=rosa_palette, legend=False)
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', fontsize=9)
plt.title("Race Balance", fontsize=14, color="#333333", weight="bold")
plt.xlabel("Race", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# --- Age distribution by gender ---
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="gender", y="age", hue="gender",
            palette=["#F9C5D5", "#F48FB1"], legend=False, showfliers=True)
plt.title("Age Distribution by Gender", fontsize=14, color="#333333", weight="bold")
plt.xlabel("Gender", fontsize=12)
plt.ylabel("Age", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# --- Age distribution by race ---
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="race", y="age", hue="race",
            palette=rosa_palette, legend=False, showfliers=True)
plt.title("Age Distribution by Race", fontsize=14, color="#333333", weight="bold")
plt.xlabel("Race", fontsize=12)
plt.ylabel("Age", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

#%% 3. Data Preprocessing (RAM-based, ready for augmentation)
import gc
from sklearn.model_selection import train_test_split
from collections import Counter

# --- Free potential leftovers ---
for name in ["X", "X_tmp", "X_train", "X_val", "X_test", "dbg_X", "dbg_y", "dbg_w"]:
    if name in globals():
        del globals()[name]
gc.collect()

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
TARGET_SIZE = (224, 224)
NORMALIZE_01 = True  # normalize images to [0,1] float32

# --- Resize all images into RAM ---
print("Resizing all images into RAM...")
X_all = np.zeros((len(images), TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.uint8)

for i, img in enumerate(images):
    X_all[i] = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

# Optional: clear original images to free RAM
images = None
gc.collect()

# --- Extract numeric labels ---
ages = np.array([d["age"] for d in labels], dtype=np.float32)

# --- Stratified splits based on age bins ---
AGE_BINS = [0, 5, 12, 18, 30, 45, 60, 80, 200]
bins_idx = np.digitize(ages, AGE_BINS, right=False) - 1
idx_all = np.arange(len(ages))

idx_tmp, idx_test, ages_tmp, ages_test, bins_tmp, bins_test = train_test_split(
    idx_all, ages, bins_idx,
    test_size=0.15, random_state=RANDOM_SEED, stratify=bins_idx
)

val_ratio_of_remaining = 0.15 / (1.0 - 0.15)
idx_train, idx_val, ages_train, ages_val, bins_train, bins_val = train_test_split(
    idx_tmp, ages_tmp, bins_tmp,
    test_size=val_ratio_of_remaining, random_state=RANDOM_SEED, stratify=bins_tmp
)

X_train, X_val, X_test = X_all[idx_train], X_all[idx_val], X_all[idx_test]
y_train, y_val, y_test = ages_train, ages_val, ages_test

# --- Optional normalization to [0,1] ---
if NORMALIZE_01:
    X_train = X_train.astype(np.float32) / 255.0
    X_val = X_val.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

print("Train/Val/Test shapes:", X_train.shape, X_val.shape, X_test.shape)
print("Bin counts (train):", Counter(bins_train))
print("Bin counts (val):  ", Counter(bins_val))
print("Bin counts (test): ", Counter(bins_test))

#%%
# --- Visualize stratified splits by age bins ---
K = len(AGE_BINS) - 1
bin_labels = []
for i in range(K):
    lo, hi = AGE_BINS[i], AGE_BINS[i + 1]
    if i < K - 1:
        bin_labels.append(f"{lo}–{hi-1}")
    else:
        bin_labels.append(f"{lo}+")

def counts_in_order(counter, K):
    return np.array([counter.get(i, 0) for i in range(K)], dtype=np.int32)

train_counts = counts_in_order(Counter(bins_train), K)
val_counts   = counts_in_order(Counter(bins_val),   K)
test_counts  = counts_in_order(Counter(bins_test),  K)

# --- Raw counts ---
x = np.arange(K)
w = 0.25

# --- Age-bin distribution: Train vs Val vs Test (pretty version) ---
plt.figure(figsize=(12, 5))
sns.set_theme(style="whitegrid")

# Define rosa-inspired palette
rosa_palette = ["#F9C5D5", "#F48FB1", "#EC407A"]

# Plot bars
plt.bar(x - w, train_counts, width=w, label="Train", color=rosa_palette[0])
plt.bar(x, val_counts, width=w, label="Val", color=rosa_palette[1])
plt.bar(x + w, test_counts, width=w, label="Test", color=rosa_palette[2])

# Labels & titles
plt.xticks(x, bin_labels, rotation=0, fontsize=11, color="#333333")
plt.xlabel("Age Bins", fontsize=12, color="#333333")
plt.ylabel("Count", fontsize=12, color="#333333")
plt.title("Age-bin Distribution: Train vs Val vs Test (Counts)",
          fontsize=14, color="#333333", weight="bold")

# Grid, legend, and layout
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(title="", fontsize=11, loc="upper right", frameon=False)
plt.tight_layout()
plt.show()

# --- Normalized percentages ---
train_pct = (train_counts / train_counts.sum()) * 100.0
val_pct   = (val_counts / val_counts.sum()) * 100.0
test_pct  = (test_counts / test_counts.sum()) * 100.0

plt.figure(figsize=(12, 5))
sns.set_theme(style="whitegrid")

# Rosa-inspired palette
rosa_palette = ["#F9C5D5", "#F48FB1", "#EC407A"]

# Plot bars
plt.bar(x - w, train_pct, width=w, label="Train", color=rosa_palette[0])
plt.bar(x, val_pct, width=w, label="Val", color=rosa_palette[1])
plt.bar(x + w, test_pct, width=w, label="Test", color=rosa_palette[2])

# Labels & titles
plt.xticks(x, bin_labels, rotation=0, fontsize=11, color="#333333")
plt.xlabel("Age Bins", fontsize=12, color="#333333")
plt.ylabel("Share (%)", fontsize=12, color="#333333")
plt.title("Age-bin Distribution: Train vs Val vs Test (Percent)",
          fontsize=14, color="#333333", weight="bold")

# Add grid and value labels
plt.grid(True, linestyle='--', alpha=0.3)
for i, (tp, vp, sp) in enumerate(zip(train_pct, val_pct, test_pct)):
    plt.text(i - w, tp + 0.5, f"{tp:.1f}%", ha='center', va='bottom', fontsize=9, color="#4A4A4A")
    plt.text(i,     vp + 0.5, f"{vp:.1f}%", ha='center', va='bottom', fontsize=9, color="#4A4A4A")
    plt.text(i + w, sp + 0.5, f"{sp:.1f}%", ha='center', va='bottom', fontsize=9, color="#4A4A4A")

# Legend and layout
plt.legend(title="", fontsize=11, loc="upper right", frameon=False)
plt.tight_layout()
plt.show()

#%% 3b. Data Augmentation Utilities (RAM-based, on-the-fly)
from collections import Counter

# --- Augmentation functions ---
def random_hflip(img, p=0.5):
    """Random horizontal flip."""
    if rng.random() < p:
        return np.ascontiguousarray(img[:, ::-1, :])
    return img

def random_rotate(img, max_deg=10):
    """Random rotation by ±max_deg."""
    deg = float(rng.uniform(-max_deg, max_deg))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def random_crop_and_resize(img, scale=(0.88, 1.0)):
    """Random crop (scale factor) and resize back to original size."""
    h, w = img.shape[:2]
    s = float(rng.uniform(scale[0], scale[1]))
    new_h, new_w = int(h * s), int(w * s)
    max_y = max(h - new_h, 0)
    max_x = max(w - new_w, 0)
    y0 = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    x0 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
    crop = img[y0:y0 + new_h, x0:x0 + new_w, :]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)

def random_brightness_contrast(img, b_lim=0.15, c_lim=0.15, p=0.8):
    """Brightness/contrast jitter (expects float32 image in [0,1])."""
    if rng.random() > p:
        return img
    brightness = float(rng.uniform(-b_lim, b_lim))
    contrast = 1.0 + float(rng.uniform(-c_lim, c_lim))
    out = img * contrast + brightness
    return np.clip(out, 0.0, 1.0)

def add_gaussian_noise(img, sigma=0.02, p=0.3):
    """Add Gaussian noise to image (expects float32 image in [0,1])."""
    if rng.random() > p:
        return img
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    out = img + noise
    return np.clip(out, 0.0, 1.0)

def augment_once(img_uint8):
    """
    Apply random augmentations sequentially.
    Input: uint8 RGB image.
    Output: float32 RGB image, normalized to [0,1].
    """
    y = random_hflip(img_uint8, p=0.5)
    y = random_rotate(y, max_deg=10)
    y = random_crop_and_resize(y, scale=(0.88, 1.0))
    y = y.astype(np.float32) / 255.0 if NORMALIZE_01 else y.astype(np.float32)
    y = random_brightness_contrast(y, b_lim=0.15, c_lim=0.15, p=0.8)
    y = add_gaussian_noise(y, sigma=0.02, p=0.3)
    return y

# Example usage (on-the-fly)
# aug_img = augment_once(X_train[0])

#%%
# --- Compute inverse-frequency sample weights (per age bin) ---
def make_sample_weights(bins):
    """
    Compute per-sample weights inversely proportional to bin frequency.
    This helps rebalance the training process across age groups.
    """
    counts = Counter(bins)
    n = len(bins)
    K = len(counts)
    return np.array([n / (K * counts[b]) for b in bins], dtype=np.float32)

train_weights = make_sample_weights(bins_train)
print("Example weights:", train_weights[:10])

#%%
# --- Visualize average sample weight per age bin ---
unique_bins = sorted(set(bins_train))
avg_weights = [train_weights[np.array(bins_train) == b].mean() for b in unique_bins]

bin_labels = []
for i in unique_bins:
    lo, hi = AGE_BINS[i], AGE_BINS[i + 1]
    if i < len(AGE_BINS) - 2:
        bin_labels.append(f"{lo}–{hi-1}")
    else:
        bin_labels.append(f"{lo}+")

# --- Inverse-frequency sample weights per age bin ---
plt.figure(figsize=(10, 5))
sns.set_theme(style="whitegrid")

bar_color = "#EC407A"  # from the rosa palette

plt.bar(bin_labels, avg_weights, color=bar_color, alpha=0.8, edgecolor="white", linewidth=1.2)

# Labels & title
plt.xlabel("Age Bins", fontsize=12, color="#333333")
plt.ylabel("Average Sample Weight", fontsize=12, color="#333333")
plt.title("Inverse-Frequency Sample Weights per Age Bin",
          fontsize=14, color="#333333", weight="bold")

# Add gentle gridlines
plt.grid(axis="y", linestyle="--", alpha=0.3)

# Add numeric labels above bars
for i, v in enumerate(avg_weights):
    plt.text(i, v + (max(avg_weights) * 0.02), f"{v:.2f}",
             ha='center', va='bottom', fontsize=9, color="#4A4A4A")

plt.tight_layout()
plt.show()

#%% 3c. Batch Generators (RAM-based, on-the-fly augmentation)
import torch

def train_batch_generator(X, y, weights=None, batch_size=32, shuffle=True):
    """
    RAM-based batch generator with on-the-fly augmentation.
    Yields (bx, by[, bw]) batches indefinitely.
    """
    n = len(y)
    order = np.arange(n)
    while True:
        if shuffle:
            rng.shuffle(order)
        for start in range(0, n, batch_size):
            sel = order[start:start + batch_size]
            bx = np.empty((len(sel),) + X.shape[1:], dtype=np.float32)
            for i, j in enumerate(sel):
                bx[i] = augment_once((X[j] * 255).astype(np.uint8) if X[j].dtype != np.uint8 else X[j])
            by = y[sel]
            if weights is not None:
                bw = weights[sel]
                yield bx, by, bw
            else:
                yield bx, by


def val_batch_iterator(X, y, batch_size=32):
    """
    RAM-based validation iterator (no augmentation, deterministic order).
    """
    n = len(y)
    for start in range(0, n, batch_size):
        sel = slice(start, start + batch_size)
        bx = (X[sel].astype(np.float32) / 255.0) if NORMALIZE_01 else X[sel].astype(np.float32)
        by = y[sel]
        yield bx, by


# --- Debug one batch ---
dbg_gen = train_batch_generator(X_train, y_train, weights=train_weights, batch_size=8)
dbg_batch = next(dbg_gen)

if len(dbg_batch) == 3:
    dbg_X, dbg_y, dbg_w = dbg_batch
else:
    dbg_X, dbg_y = dbg_batch
    dbg_w = np.ones_like(dbg_y, dtype=np.float32)

print("Debug batch:", dbg_X.shape, dbg_y.shape, dbg_w.shape, dbg_X.dtype)

#%% Torch conversion check
X_t = torch.from_numpy(dbg_X).permute(0, 3, 1, 2)  # NHWC -> NCHW
y_t = torch.from_numpy(dbg_y).float()
w_t = torch.from_numpy(dbg_w).float()
print(X_t.shape, y_t.shape, w_t.shape)  # -> torch.Size([8, 3, 224, 224]) torch.Size([8]) torch.Size([8])

#%% Visualize a few augmented samples (fixed title positioning)
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 4, figsize=(10, 5), constrained_layout=True)  # uses constrained layout
fig.suptitle("Augmented Sample Previews", fontsize=14, color="#333333", weight="bold", y=1.02)

for ax, img, age in zip(axes.ravel(), dbg_X, dbg_y):
    ax.imshow(np.clip(img, 0, 1))  # float32 RGB in [0,1]
    ax.set_title(f"Age: {int(age)}", fontsize=10, color="#4A4A4A", pad=4)
    ax.axis("off")

plt.show()

#%%
# --- PyTorch adapter for (numpy -> torch) conversion ---
def torchify_batch(batch):
    """
    Convert numpy batch tuples to PyTorch tensors.
    Supports (X, y) or (X, y, w) tuples.
    """
    if len(batch) == 3:
        X, y, w = batch
        return (
            torch.from_numpy(X).permute(0, 3, 1, 2),  # NHWC -> NCHW
            torch.from_numpy(y).float(),
            torch.from_numpy(w).float()
        )
    else:
        X, y = batch
        return (
            torch.from_numpy(X).permute(0, 3, 1, 2),
            torch.from_numpy(y).float()
        )

# --- Example: one PyTorch batch ---
dbg_gen = train_batch_generator(X_train, y_train, weights=train_weights, batch_size=16)
dbg_batch = next(dbg_gen)
X_np, y_np, w_np = dbg_batch if len(dbg_batch) == 3 else (*dbg_batch, np.ones_like(dbg_batch[1], dtype=np.float32))
X_t, y_t, w_t = torchify_batch((X_np, y_np, w_np))

print("Torch batch shapes:", X_t.shape, y_t.shape, w_t.shape)

#%% 4. Model Design & Implementation (RAM-based training ready)
from tensorflow.keras import layers, models, regularizers, optimizers

# --- Define CNN model ---
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # Dense head
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    # Output
    layers.Dense(1, activation='linear', dtype='float32')  # regression output
])

# --- Compile model ---
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='mae',           # mean absolute error for age regression
    metrics=['mae', 'mse']
)

# --- Print model summary ---
model.summary()

#%% 5. Training (RAM-based, on-the-fly augmentation)
from tensorflow.keras import callbacks, optimizers

# Set seaborn theme for plots
sns.set_theme(style="whitegrid")

# --- Batch generators ---
batch_size = 32
train_gen = train_batch_generator(X_train, y_train, weights=train_weights, batch_size=batch_size)
val_gen   = val_batch_iterator(X_val, y_val, batch_size=batch_size)

steps_per_epoch = len(X_train) // batch_size
validation_steps = len(X_val) // batch_size

print(f"Training samples: {len(y_train)} | Validation samples: {len(y_val)}")
print(f"Steps per epoch: {steps_per_epoch} | Validation steps: {validation_steps}")
print(f"Batch size: {batch_size}")

# --- Callbacks ---
checkpoint_cb = callbacks.ModelCheckpoint(
    "best_model.keras",
    save_best_only=True,
    monitor="val_mae",
    mode="min"
)

reduce_lr_cb = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

early_stop_cb = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# --- Compile model ---
optimizer = optimizers.Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer,
    loss='mae',
    metrics=['mae', 'mse']
)

# --- Train the model ---
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=80,
    callbacks=[checkpoint_cb, reduce_lr_cb, early_stop_cb],
    verbose=1
)

# --- Plot training history ---
plt.figure(figsize=(14, 6))

train_color = "#F48FB1"
val_color = "#F9C5D5"

# Loss Curve
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color=train_color, linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color=val_color, linewidth=2)
plt.title('Loss Curve (MAE)', fontsize=14, weight='bold', color="#4A4A4A")
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MAE Loss', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

# MAE Curve
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE', color=train_color, linewidth=2)
plt.plot(history.history['val_mae'], label='Validation MAE', color=val_color, linewidth=2)
plt.title('MAE Curve', fontsize=14, weight='bold', color="#4A4A4A")
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MAE', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

# --- Evaluate on Test Set ---
X_test_array = X_test.astype(np.float32)  # already normalized if NORMALIZE_01=True
y_test_array = y_test

test_loss, test_mae, test_mse = model.evaluate(X_test_array, y_test_array, verbose=1)
print(f"\n--- Test Set Evaluation ---")
print(f"MAE: {test_mae:.2f} years")
print(f"MSE: {test_mse:.2f}")
print(f"Loss (MAE): {test_loss:.2f}")

# --- Optional: quick summary stats ---
print(f"Training epochs: {len(history.history['loss'])}")
print(f"Min validation MAE: {min(history.history['val_mae']):.2f}")
print(f"Max training MAE: {max(history.history['mae']):.2f}")