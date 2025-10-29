# This file contains the newest and best code. The file called "best_model.keras" is the
# best model for this main.py file.
import os

from src.data_loader import load_utkface_dataset
from src.visualize import (plot_random_samples, plot_distribution_charts, plot_age_bin_distribution,
                           plot_avg_sample_weight_per_bin, plot_augmented_samples, plot_training_history)
from src.preprocess import preprocess_images_to_memmap, load_and_split_from_memmap
from src.augment import set_augment_seed
from src.utils import make_sample_weights, train_batch_generator, val_batch_iterator, evaluate_model_on_test
from src.build_model import build_model
from src.train_model import train_model

#%% --- Step 1: Load data ---
images, labels, df = load_utkface_dataset("./data/images")

#%% --- Step 2: Visualize ---
plot_random_samples(images, labels, gender_map={0:"Male",1:"Female"},
                    race_map={0:"White",1:"Black",2:"Asian",3:"Indian",4:"Other"})
plot_distribution_charts(df)

#%% --- 3. Data Preprocessing (disk-based via memmap) ---
target_size = (192, 192)  # or (160, 160) or (224, 224)
memmap_dir = "./data/memmap"
size_tag = f"{target_size[0]}x{target_size[1]}"
X_path = os.path.join(memmap_dir, f"X_resized_{size_tag}.dat")
y_path = os.path.join(memmap_dir, f"y_resized_{size_tag}.dat")

# Create memmap only if it doesn’t exist yet
if not (os.path.exists(X_path) and os.path.exists(y_path)):
    preprocess_images_to_memmap(images, labels, target_size=target_size, save_X_path=X_path, save_y_path=y_path)
else:
    print("Memmap files already exist — skipping preprocessing.")

n_samples = len(images)

preproc = load_and_split_from_memmap(
    X_path=X_path,
    y_path=y_path,
    n_samples=n_samples,
    target_size=target_size,
    normalize_01=True
)

# Unpack all returned objects
rng = preproc["rng"]
NORMALIZE_01 = preproc["NORMALIZE_01"]
AGE_BINS = preproc["AGE_BINS"]

bins_train = preproc["bins_train"]
bins_val = preproc["bins_val"]
bins_test = preproc["bins_test"]

X_train = preproc["X_train"]
X_val = preproc["X_val"]
X_test = preproc["X_test"]

y_train = preproc["y_train"]
y_val = preproc["y_val"]
y_test = preproc["y_test"]

y_train_std = preproc["y_train_std"]
y_val_std = preproc["y_val_std"]
y_test_std = preproc["y_test_std"]

age_mean = preproc["age_mean"]
age_std = preproc["age_std"]

# --- Visualize stratified splits by age bins ---
plot_age_bin_distribution(bins_train=bins_train, bins_val=bins_val, bins_test=bins_test, AGE_BINS=AGE_BINS)

#%% --- Data Augmentation Utilities (RAM-based, on-the-fly) ---
# Set RNG and normalization from preprocessing
set_augment_seed(rng, NORMALIZE_01)

#%% --- Compute inverse-frequency sample weights (per age bin) ---
train_weights = make_sample_weights(bins_train)
print("Example weights:", train_weights[:10])

#%% --- Visualize average sample weight per age bin ---
plot_avg_sample_weight_per_bin(train_weights=train_weights, bins_train=bins_train, AGE_BINS=AGE_BINS)

#%% --- Batch Generators (RAM-based, on-the-fly augmentation) ---
# --- Batch generators using standardized ages ---
batch_size = 16
train_gen = train_batch_generator(X_train, y_train_std, batch_size=batch_size, weights=train_weights)
val_gen = val_batch_iterator(X_val, y_val_std, batch_size=batch_size)

# Create a small batch for visualization
dbg_gen = train_batch_generator(X_train, y_train_std, batch_size=batch_size, weights=train_weights)
dbg_batch = next(dbg_gen)

if len(dbg_batch) == 3:
    dbg_X, dbg_y_std, _ = dbg_batch
else:
    dbg_X, dbg_y_std = dbg_batch

# Map to raw ages
dbg_y_raw = y_train[:len(dbg_y_std)]

# Plot augmented samples
plot_augmented_samples(dbg_X, dbg_y_raw)

#%% 4. Model Design & Implementation (RAM-based training ready)
model = build_model(input_shape=target_size + (3,))
model.summary()

#%% 5. Training (RAM-based, on-the-fly augmentation)
# Train the model
history = train_model(
    model=model,
    train_gen=train_gen,
    val_gen=val_gen,
    y_train_std=y_train_std,
    y_val_std=y_val_std,
    batch_size=batch_size,
    epochs=50
)

# Plot training curves
plot_training_history(history)

evaluate_model_on_test(
    model=model,
    X_test=X_test,
    y_test_std=y_test_std,
    age_mean=age_mean,
    age_std=age_std,
    history=history,
    normalize=True,
    verbose=True
)