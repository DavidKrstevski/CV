import os
import random
import cv2
import numpy as np
import pandas as pd

def load_utkface_dataset(image_dir="./data/images"):
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

    # %% 1a. Dataset Summary Statistics
    df = pd.DataFrame(labels)

    print("\n=== Dataset Summary ===")
    print(f"Total images: {len(df)}")
    print(f"Age range: {df['age'].min()} - {df['age'].max()} (mean: {df['age'].mean():.1f})")
    print(f"Gender distribution:\n{df['gender'].value_counts()}")
    print(f"Race distribution:\n{df['race'].value_counts()}")
    print(f"Average image size (HxW): {np.mean([img.shape[:2] for img in images], axis=0).astype(int)}")

    return images, labels, df