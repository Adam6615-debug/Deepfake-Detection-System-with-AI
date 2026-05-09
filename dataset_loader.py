"""
dataset_loader.py
-----------------
Loads real and fake images from the data/ folder.
Folder structure expected:
    data/
        real/   <-- put real face images here (.jpg, .png)
        fake/   <-- put deepfake images here (.jpg, .png)
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
IMAGE_SIZE = (224, 224)  # Standard input size for CNN models


def load_images_from_folder(folder_path: str, label: int, image_size=IMAGE_SIZE):
    """
    Loads all images from a folder and assigns them a label.

    Args:
        folder_path: Path to the folder containing images.
        label: 0 = real, 1 = fake
        image_size: Tuple (width, height) to resize images to.

    Returns:
        images: List of numpy arrays (H, W, 3)
        labels: List of ints (all same value = label)
    """
    images = []
    labels = []

    if not os.path.exists(folder_path):
        print(f"[WARNING] Folder not found: {folder_path}")
        return images, labels

    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if len(files) == 0:
        print(f"[WARNING] No images found in: {folder_path}")
        return images, labels

    print(f"[INFO] Loading {len(files)} images from: {folder_path}")

    for filename in tqdm(files, desc=f"  {'Real' if label == 0 else 'Fake'} images"):
        filepath = os.path.join(folder_path, filename)
        try:
            img = Image.open(filepath).convert("RGB")
            img = img.resize(image_size)
            img_array = np.array(img, dtype=np.float32)
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"[WARNING] Could not load {filename}: {e}")

    return images, labels


def load_dataset(data_dir: str = "data", image_size=IMAGE_SIZE):
    """
    Loads the full dataset from data/real/ and data/fake/.

    Args:
        data_dir: Root directory containing real/ and fake/ subfolders.
        image_size: Tuple (width, height) to resize images to.

    Returns:
        X: numpy array of shape (N, H, W, 3), pixel values in [0, 1]
        y: numpy array of shape (N,), labels (0=real, 1=fake)
    """
    real_path = os.path.join(data_dir, "real")
    fake_path = os.path.join(data_dir, "fake")

    real_images, real_labels = load_images_from_folder(real_path, label=0, image_size=image_size)
    fake_images, fake_labels = load_images_from_folder(fake_path, label=1, image_size=image_size)

    all_images = real_images + fake_images
    all_labels = real_labels + fake_labels

    if len(all_images) == 0:
        print("[ERROR] No images loaded. Check your data/ folder.")
        return None, None

    X = np.array(all_images, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    y = np.array(all_labels, dtype=np.int32)

    print(f"\n[INFO] Dataset loaded:")
    print(f"       Total images : {len(X)}")
    print(f"       Real (0)     : {np.sum(y == 0)}")
    print(f"       Fake (1)     : {np.sum(y == 1)}")
    print(f"       Image shape  : {X[0].shape}")

    return X, y


def generate_dummy_dataset(n_real=20, n_fake=20, image_size=IMAGE_SIZE):
    """
    Generates random dummy images for testing without real data.
    Use this to verify your pipeline works before adding real images.

    Args:
        n_real: Number of dummy real images.
        n_fake: Number of dummy fake images.
        image_size: Tuple (width, height).

    Returns:
        X: numpy array of shape (N, H, W, 3), values in [0, 1]
        y: numpy array of shape (N,)
    """
    print("[INFO] Generating dummy dataset for testing...")
    H, W = image_size

    real = np.random.rand(n_real, H, W, 3).astype(np.float32)
    fake = np.random.rand(n_fake, H, W, 3).astype(np.float32)

    X = np.concatenate([real, fake], axis=0)
    y = np.array([0] * n_real + [1] * n_fake, dtype=np.int32)

    print(f"[INFO] Dummy dataset: {len(X)} images ({n_real} real, {n_fake} fake)")
    return X, y


if __name__ == "__main__":
    # Quick test — runs dummy data if no real images present
    X, y = load_dataset("data")
    if X is None:
        print("[INFO] Falling back to dummy dataset...")
        X, y = generate_dummy_dataset()
    print(f"[OK] X shape: {X.shape}, y shape: {y.shape}")
