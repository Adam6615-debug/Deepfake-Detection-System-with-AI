"""
preprocessor.py
---------------
Image preprocessing pipeline for deepfake detection.
"""

import io
import numpy as np
import cv2
from PIL import Image


IMAGE_SIZE  = (224, 224)
ELA_QUALITY = 90   # JPEG re-save quality for ELA
ELA_SCALE   = 15   # Amplification factor


# ─── ELA (Error Level Analysis) ──────────────────────────────────────────────

def apply_ela(image_array: np.ndarray, quality: int = ELA_QUALITY, scale: int = ELA_SCALE) -> np.ndarray:
    """
    Applies Error Level Analysis to a single image.

    Args:
        image_array: numpy array (H, W, 3), values in [0, 255] uint8
        quality: JPEG re-save quality (default 90)
        scale: multiplier to amplify pixel differences (default 15)

    Returns:
        ela_image: numpy array (H, W, 3), values in [0, 255] uint8
    """
    pil_img = Image.fromarray(image_array.astype(np.uint8))
    buffer  = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)

    original   = np.array(pil_img,    dtype=np.float32)
    compressed = np.array(compressed, dtype=np.float32)

    ela = np.abs(original - compressed) * scale
    ela = np.clip(ela, 0, 255).astype(np.uint8)
    return ela


def apply_ela_batch(images: np.ndarray, quality: int = ELA_QUALITY, scale: int = ELA_SCALE) -> np.ndarray:
    """
    Applies ELA to a batch of images.

    Args:
        images: numpy array (N, H, W, 3), values in [0, 1] float
    Returns:
        ela_batch: numpy array (N, H, W, 3), values in [0, 1] float
    """
    results = []
    for img in images:
        img_uint8 = (img * 255).astype(np.uint8)
        ela = apply_ela(img_uint8, quality=quality, scale=scale)
        results.append(ela.astype(np.float32) / 255.0)
    return np.array(results, dtype=np.float32)


# ─── Face Detection ──────────────────────────────────────────────────────────

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_and_crop_face(image_array: np.ndarray, target_size=IMAGE_SIZE) -> np.ndarray:
    """
    Detects the largest face in an image and crops to it.
    Falls back to center-crop if no face is detected.

    Args:
        image_array: numpy array (H, W, 3), values in [0, 255] uint8
    Returns:
        face_array: numpy array (H, W, 3), values in [0, 255] uint8
    """
    gray  = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        areas       = [w * h for (x, y, w, h) in faces]
        x, y, w, h = faces[np.argmax(areas)]
        pad = int(min(w, h) * 0.1)
        x   = max(0, x - pad)
        y   = max(0, y - pad)
        w   = min(image_array.shape[1] - x, w + 2 * pad)
        h   = min(image_array.shape[0] - y, h + 2 * pad)
        cropped = image_array[y:y+h, x:x+w]
    else:
        H, W    = image_array.shape[:2]
        cropped = image_array[H//4 : H*3//4, W//4 : W*3//4]

    return cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)


# ─── Main Preprocessing Pipeline ─────────────────────────────────────────────

def preprocess_single(
    image_array: np.ndarray,
    use_ela: bool = False,      # FIX: default False — raw RGB trains better
    detect_face: bool = True
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single image.

    Args:
        image_array: (H, W, 3) — accepts EITHER:
                       • uint8  [0, 255]  (from PIL / disk)
                       • float32 [0, 1]  (from dataset_loader)
                     Both handled correctly — no double-normalization.
        use_ela:     Apply ELA transform (default False for stable training)
        detect_face: Crop to largest face region (default True)

    Returns:
        float32 (224, 224, 3) in [0, 1]
    """
    # FIX: safely convert to uint8 regardless of input range
    if image_array.dtype != np.uint8:
        if image_array.max() <= 1.0:
            img = (image_array * 255).astype(np.uint8)   # float [0,1] → uint8
        else:
            img = image_array.astype(np.uint8)            # float [0,255] → uint8
    else:
        img = image_array.copy()

    # Face detection / resize
    if detect_face:
        img = detect_and_crop_face(img, target_size=IMAGE_SIZE)
    else:
        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # Optional ELA
    if use_ela:
        img = apply_ela(img)

    return img.astype(np.float32) / 255.0


def preprocess_batch(
    images: np.ndarray,
    use_ela: bool = False,      # FIX: default False to match preprocess_single
    detect_face: bool = True
) -> np.ndarray:
    """
    Preprocesses a batch of images.

    Args:
        images: numpy array (N, H, W, 3), values in [0, 1] float
    Returns:
        processed: numpy array (N, 224, 224, 3), values in [0, 1] float
    """
    return np.array(
        [preprocess_single(img, use_ela=use_ela, detect_face=detect_face) for img in images],
        dtype=np.float32
    )


# ─── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[TEST] Running preprocessor on dummy images...")

    dummy_uint8 = (np.random.rand(300, 300, 3) * 255).astype(np.uint8)
    dummy_float = np.random.rand(300, 300, 3).astype(np.float32)

    for label, img in [("uint8 input", dummy_uint8), ("float input", dummy_float)]:
        out = preprocess_single(img, use_ela=False, detect_face=False)
        print(f"[OK] {label} -> shape {out.shape}, range [{out.min():.3f}, {out.max():.3f}]")

    ela_out = apply_ela(dummy_uint8)
    print(f"[OK] ELA output shape: {ela_out.shape}, dtype: {ela_out.dtype}")

    batch     = np.random.rand(4, 224, 224, 3).astype(np.float32)
    batch_out = preprocess_batch(batch, use_ela=False, detect_face=False)
    print(f"[OK] Batch output: {batch_out.shape}")
    print("[DONE] Preprocessor working correctly.")
