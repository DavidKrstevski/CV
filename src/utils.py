import numpy as np
from collections import Counter
from src.augment import augment_once

def make_sample_weights(bins):
    """
    Compute per-sample weights inversely proportional to bin frequency.
    This helps rebalance the training process across age groups.

    Parameters
    ----------
    bins : array-like
        Age bin index for each sample.

    Returns
    -------
    np.ndarray
        Per-sample weights (float32) of same length as `bins`.
    """
    counts = Counter(bins)
    n = len(bins)
    K = len(counts)
    return np.array([n / (K * counts[b]) for b in bins], dtype=np.float32)

def train_batch_generator(X, y, batch_size, weights=None, shuffle=True):
    """
    Disk/memmap-safe batch generator with on-the-fly augmentation.
    Converts each sample individually to float32 for augmentation.
    """
    n = len(y)
    order = np.arange(n)

    while True:
        if shuffle:
            np.random.shuffle(order)
        for start in range(0, n, batch_size):
            sel = order[start:start + batch_size]
            bx = np.empty((len(sel),) + X.shape[1:], dtype=np.float32)

            for i, j in enumerate(sel):
                # X[j] is a memmap row (float16, normalized [0,1])
                img = (X[j] * 255.0).astype(np.uint8)
                aug = augment_once(img) # augment_once returns float32 normalized [0,1]
                bx[i] = aug

            by = y[sel]

            if weights is not None:
                bw = weights[sel]
                yield bx, by, bw
            else:
                yield bx, by

def val_batch_iterator(X, y, batch_size, normalize=True):
    """
    Disk/memmap-safe validation iterator (no augmentation, deterministic order).
    """
    n = len(y)
    for start in range(0, n, batch_size):
        sel = slice(start, start + batch_size)
        bx = X[sel].astype(np.float32)
        # X is already normalized [0,1] if loaded from memmap
        if not normalize:
            bx = (bx * 255.0)
        by = y[sel]
        yield bx, by


def evaluate_model_on_test(model, X_test, y_test_std, age_mean, age_std, history=None, normalize=True, verbose=True):
    """
    Evaluate the model on the test set with standardized labels,
    and compute raw (interpretable) MAE and MSE.

    Args:
        model: Trained Keras model.
        X_test: Test images (numpy array).
        y_test_std: Standardized target labels (numpy array).
        age_mean: Mean used for standardization.
        age_std: Std used for standardization.
        history: Optional Keras History object for printing summary stats.
        normalize: Whether X_test is already normalized to [0,1].
        verbose: Print results if True.

    Returns:
        dict with standardized and raw metrics:
            {
                'test_loss': float,
                'test_mae': float,
                'test_mse': float,
                'raw_mae': float,
                'raw_mse': float
            }
    """
    # Ensure float32 input
    X_test_array = X_test.astype(np.float32)
    y_test_array = y_test_std

    # Evaluate standardized metrics
    test_loss, test_mae, test_mse = model.evaluate(X_test_array, y_test_array, verbose=verbose)

    # Convert predictions back to raw ages
    y_pred_std = model.predict(X_test_array, verbose=0)
    y_pred_raw = y_pred_std * age_std + age_mean
    y_test_raw  = y_test_array * age_std + age_mean

    # Raw MAE/MSE
    raw_mae = np.mean(np.abs(y_pred_raw.flatten() - y_test_raw.flatten()))
    raw_mse = np.mean((y_pred_raw.flatten() - y_test_raw.flatten())**2)

    if verbose:
        print(f"\n--- Test Set Evaluation ---")
        print(f"Standardized MAE (model output): {test_mae:.4f}")
        print(f"Standardized MSE: {test_mse:.4f}")
        print(f"Raw MAE: {raw_mae:.2f} years")
        print(f"Raw MSE: {raw_mse:.2f}")
        if history is not None:
            print(f"\nTraining epochs: {len(history.history['loss'])}")
            print(f"Min validation MAE: {min(history.history['val_mae']):.4f} (standardized)")
