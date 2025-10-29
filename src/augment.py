"""
augment.py
-----------
RAM-based, on-the-fly image augmentation utilities for UTKFace.
"""

import numpy as np
import cv2

# The RNG and normalization flag will be injected from the main preprocessing context
rng = None
NORMALIZE_01 = True


def set_augment_seed(random_generator, normalize=True):
    """
    Set the RNG object and normalization flag for augmentations.

    Parameters
    ----------
    random_generator : np.random.Generator
        RNG instance from preprocessing.
    normalize : bool
        Whether to normalize output images to [0,1] float32.
    """
    global rng, NORMALIZE_01
    rng = random_generator
    NORMALIZE_01 = normalize


# --- Augmentation functions ---
def random_hflip(img, p=0.5):
    if rng.random() < p:
        return np.ascontiguousarray(img[:, ::-1, :])
    return img


def random_rotate(img, max_deg=10):
    deg = float(rng.uniform(-max_deg, max_deg))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def random_crop_and_resize(img, scale=(0.88, 1.0)):
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
    if rng.random() > p:
        return img
    brightness = float(rng.uniform(-b_lim, b_lim))
    contrast = 1.0 + float(rng.uniform(-c_lim, c_lim))
    out = img * contrast + brightness
    return np.clip(out, 0.0, 1.0)


def add_gaussian_noise(img, sigma=0.02, p=0.3):
    if rng.random() > p:
        return img
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    out = img + noise
    return np.clip(out, 0.0, 1.0)


def augment_once(img_uint8):
    """
    Apply random augmentations sequentially.
    Input: uint8 RGB image.
    Output: float32 RGB image, normalized to [0,1] if NORMALIZE_01=True.
    """
    y = random_hflip(img_uint8, p=0.5)
    y = random_rotate(y, max_deg=10)
    y = random_crop_and_resize(y, scale=(0.88, 1.0))
    y = y.astype(np.float32) / 255.0 if NORMALIZE_01 else y.astype(np.float32)
    y = random_brightness_contrast(y, b_lim=0.15, c_lim=0.15, p=0.8)
    y = add_gaussian_noise(y, sigma=0.02, p=0.3)
    return y
