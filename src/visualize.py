import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.container import BarContainer
import numpy as np
from collections import Counter

def plot_random_samples(images, labels, gender_map, race_map):
    # --- Random sample visualization ---
    plt.figure(figsize=(10, 6))
    for i, idx in enumerate(random.sample(range(len(images)), min(9, len(images)))):
        img = images[idx]
        label = labels[idx]
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")
        gender_value = label.get("gender")
        race_value = label.get("race")
        plt.title(
            f'Age: {label["age"]}\n'
            f'Gender: {gender_map.get(gender_value, gender_value)} '
            f'Race: {race_map.get(race_value, race_value)}',
            fontsize=10,
            color="#4A4A4A"
        )
    plt.suptitle("Random Sample of UTKFace Images", fontsize=14, color="#333333", weight="bold")
    plt.tight_layout()
    plt.show()

def plot_distribution_charts(df):
    # Set theme and gentle palette
    sns.set_theme(style="whitegrid")
    rosa_palette = ["#F9C5D5", "#F7A1C4", "#F48FB1", "#F06292", "#EC407A"]

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
        if isinstance(container, BarContainer):
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
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt='%d', label_type='edge', fontsize=10)
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

def plot_age_bin_distribution(bins_train, bins_val, bins_test, AGE_BINS, rosa_palette=None):
    """
    Visualize stratified splits by age bins for train, val, and test sets.

    Parameters
    ----------
    bins_train, bins_val, bins_test : array-like
        Arrays of bin indices corresponding to train, val, test splits.
    AGE_BINS : list[int]
        List of age bin edges.
    rosa_palette : list[str], optional
        Custom colors for the train/val/test bars. Defaults to pink-inspired palette.

    Returns
    -------
    bin_labels : list[str]
        Human-readable labels for each age bin.
    """

    # --- Default rosa-inspired palette ---
    if rosa_palette is None:
        rosa_palette = ["#F9C5D5", "#F48FB1", "#EC407A"]

    K = len(AGE_BINS) - 1

    # --- Create bin labels ---
    bin_labels = []
    for i in range(K):
        lo, hi = AGE_BINS[i], AGE_BINS[i + 1]
        if i < K - 1:
            bin_labels.append(f"{lo}–{hi-1}")
        else:
            bin_labels.append(f"{lo}+")

    # --- Helper function to count occurrences ---
    def counts_in_order(counter, K):
        return np.array([counter.get(i, 0) for i in range(K)], dtype=np.int32)

    train_counts = counts_in_order(Counter(bins_train), K)
    val_counts   = counts_in_order(Counter(bins_val), K)
    test_counts  = counts_in_order(Counter(bins_test), K)

    x = np.arange(K)
    w = 0.25

    # --- Raw counts plot ---
    plt.figure(figsize=(12, 5))
    sns.set_theme(style="whitegrid")

    plt.bar(x - w, train_counts, width=w, label="Train", color=rosa_palette[0])
    plt.bar(x, val_counts, width=w, label="Val", color=rosa_palette[1])
    plt.bar(x + w, test_counts, width=w, label="Test", color=rosa_palette[2])

    plt.xticks(x, bin_labels, rotation=0, fontsize=11, color="#333333")
    plt.xlabel("Age Bins", fontsize=12, color="#333333")
    plt.ylabel("Count", fontsize=12, color="#333333")
    plt.title("Age-bin Distribution: Train vs Val vs Test (Counts)",
              fontsize=14, color="#333333", weight="bold")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(title="", fontsize=11, loc="upper right", frameon=False)
    plt.tight_layout()
    plt.show()

    # --- Normalized percentages plot ---
    train_pct = (train_counts / train_counts.sum()) * 100.0
    val_pct   = (val_counts / val_counts.sum()) * 100.0
    test_pct  = (test_counts / test_counts.sum()) * 100.0

    plt.figure(figsize=(12, 5))
    sns.set_theme(style="whitegrid")

    plt.bar(x - w, train_pct, width=w, label="Train", color=rosa_palette[0])
    plt.bar(x, val_pct, width=w, label="Val", color=rosa_palette[1])
    plt.bar(x + w, test_pct, width=w, label="Test", color=rosa_palette[2])

    plt.xticks(x, bin_labels, rotation=0, fontsize=11, color="#333333")
    plt.xlabel("Age Bins", fontsize=12, color="#333333")
    plt.ylabel("Share (%)", fontsize=12, color="#333333")
    plt.title("Age-bin Distribution: Train vs Val vs Test (Percent)",
              fontsize=14, color="#333333", weight="bold")

    plt.grid(True, linestyle='--', alpha=0.3)
    for i, (tp, vp, sp) in enumerate(zip(train_pct, val_pct, test_pct)):
        plt.text(i - w, tp + 0.5, f"{tp:.1f}%", ha='center', va='bottom', fontsize=9, color="#4A4A4A")
        plt.text(i, vp + 0.5, f"{vp:.1f}%", ha='center', va='bottom', fontsize=9, color="#4A4A4A")
        plt.text(i + w, sp + 0.5, f"{sp:.1f}%", ha='center', va='bottom', fontsize=9, color="#4A4A4A")

    plt.legend(title="", fontsize=11, loc="upper right", frameon=False)
    plt.tight_layout()
    plt.show()

def plot_avg_sample_weight_per_bin(train_weights, bins_train, AGE_BINS, bar_color="#EC407A"):
    """
    Plot average inverse-frequency sample weight per age bin.

    Parameters
    ----------
    train_weights : array-like
        Per-sample weights computed via make_sample_weights().
    bins_train : array-like
        Age bin index for each sample in the training set.
    AGE_BINS : list[int]
        List of age bin edges.
    bar_color : str, optional
        Color of the bars. Default is a rosa-inspired pink.

    Returns
    -------
    bin_labels : list[str]
        Human-readable labels for each age bin.
    """
    unique_bins = sorted(set(bins_train))
    avg_weights = [train_weights[np.array(bins_train) == b].mean() for b in unique_bins]

    # Create human-readable bin labels
    bin_labels = []
    for i in unique_bins:
        lo, hi = AGE_BINS[i], AGE_BINS[i + 1]
        if i < len(AGE_BINS) - 2:
            bin_labels.append(f"{lo}–{hi - 1}")
        else:
            bin_labels.append(f"{lo}+")

    # Plot
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid")

    plt.bar(bin_labels, avg_weights, color=bar_color, alpha=0.8, edgecolor="white", linewidth=1.2)
    plt.xlabel("Age Bins", fontsize=12, color="#333333")
    plt.ylabel("Average Sample Weight", fontsize=12, color="#333333")
    plt.title("Inverse-Frequency Sample Weights per Age Bin",
              fontsize=14, color="#333333", weight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    # Add numeric labels above bars
    for i, v in enumerate(avg_weights):
        plt.text(i, v + (max(avg_weights) * 0.02), f"{v:.2f}",
                 ha='center', va='bottom', fontsize=9, color="#4A4A4A")

    plt.tight_layout()
    plt.show()

def plot_augmented_samples(batch_X, batch_y_raw, n_rows=2, n_cols=4, title="Augmented Sample Previews"):
    """
    Visualize a grid of augmented images from a batch.

    Parameters
    ----------
    batch_X : np.ndarray
        Batch of images (float32, normalized to [0,1]).
    batch_y_raw : array-like
        Corresponding raw labels (e.g., ages).
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.
    title : str
        Figure title.
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5), constrained_layout=True)
    fig.suptitle(title, fontsize=14, color="#333333", weight="bold", y=1.02)

    for ax, img, label in zip(axes.ravel(), batch_X, batch_y_raw):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f"Age: {int(label)}", fontsize=10, color="#4A4A4A", pad=4)
        ax.axis("off")

    plt.show()

def plot_training_history(history, title_suffix=""):
    """
    Plot loss and MAE curves from model training history.
    """
    plt.figure(figsize=(14, 6))

    train_color = "#F48FB1"
    val_color = "#F9C5D5"

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', color=train_color, linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color=val_color, linewidth=2)
    plt.title(f'Loss Curve {title_suffix}', fontsize=14, weight='bold', color="#4A4A4A")
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    # MAE Curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE', color=train_color, linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', color=val_color, linewidth=2)
    plt.title(f'MAE Curve {title_suffix}', fontsize=14, weight='bold', color="#4A4A4A")
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()