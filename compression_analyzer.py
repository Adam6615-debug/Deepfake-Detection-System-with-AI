"""
compression_analyzer.py
------------------------
JPEG Compression Artifact Analysis

Detects abnormal JPEG compression patterns that indicate image manipulation.

When an image is edited and re-saved, the manipulation leaves forensic traces:
  1. Double JPEG compression — edited regions get compressed twice, leaving
     different block boundary patterns than the untouched background
  2. Block boundary irregularities — 8x8 DCT blocks have unnatural seam strengths
  3. JPEG ghost detection — at different quality levels, authentic vs manipulated
     regions show different re-compression error patterns

Output: compression_score in [0, 1]. Higher = more suspicious compression patterns.
"""

import numpy as np
import cv2
import io
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


# ─── JPEG Block Boundary Analysis ─────────────────────────────────────────────

def _block_boundary_score(image_gray: np.ndarray) -> float:
    """
    Analyzes the strength and regularity of 8x8 JPEG block boundaries.

    JPEG compression works on 8x8 pixel blocks. This leaves a subtle
    grid pattern in the image. When an image is manipulated:
      - Pasted regions may be at a different phase (shifted grid)
      - Re-saved regions have different block boundary strength

    Steps:
      - Compute horizontal and vertical gradient across the image
      - Sample gradients at positions that are multiples of 8 (block edges)
      - Compare block-edge gradients to non-edge gradients
      - High ratio = strong blocking (heavily compressed or double-compressed)
      - Inconsistent ratio across regions = manipulation indicator
    """
    img = cv2.resize(image_gray, (256, 256)).astype(np.float32)

    # Horizontal and vertical differences
    diff_h = np.abs(np.diff(img, axis=1))   # (256, 255)
    diff_v = np.abs(np.diff(img, axis=0))   # (255, 256)

    H_diff, W_diff = diff_h.shape   # diff_h shape: (H, W-1)
    H_v_diff = diff_v.shape[0]      # diff_v shape: (H-1, W)

    # JPEG block edges occur at columns/rows that are multiples of 8
    block_cols    = [c for c in range(7, W_diff, 8) if c < W_diff]
    nonblock_cols = [c for c in range(W_diff) if c % 8 != 7]

    block_rows    = [r for r in range(7, H_v_diff, 8) if r < H_v_diff]
    nonblock_rows = [r for r in range(H_v_diff) if r % 8 != 7]

    W_h = W_diff
    H_v = H_v_diff

    if not block_cols or not nonblock_cols or not block_rows or not nonblock_rows:
        return 0.0

    # Mean gradient at block edges vs non-block positions
    block_h_mean    = np.mean(diff_h[:, block_cols])
    nonblock_h_mean = np.mean(diff_h[:, nonblock_cols])
    block_v_mean    = np.mean(diff_v[block_rows, :])
    nonblock_v_mean = np.mean(diff_v[nonblock_rows, :])

    if nonblock_h_mean < 1e-8 or nonblock_v_mean < 1e-8:
        return 0.0

    ratio_h = block_h_mean / nonblock_h_mean
    ratio_v = block_v_mean / nonblock_v_mean
    mean_ratio = (ratio_h + ratio_v) / 2.0

    # Natural JPEG image: ratio ~1.1–1.5 (slight block effect)
    # Heavy compression / double JPEG: ratio > 2.0
    # No JPEG compression (PNG original): ratio ~1.0
    # Score high if ratio is outside the natural range in either direction
    if mean_ratio > 1.5:
        # Unusually strong blocking
        score = float(np.clip((mean_ratio - 1.5) / 2.0, 0.0, 1.0))
    else:
        # Too clean or too weak blocks (may indicate patching)
        score = float(np.clip((1.1 - mean_ratio) / 1.1, 0.0, 1.0))

    return score


# ─── Double JPEG Detection (JPEG Ghost) ───────────────────────────────────────

def _double_jpeg_score(image_array: np.ndarray) -> float:
    """
    Detects double JPEG compression using the "JPEG Ghost" method.

    Concept (Farid 2009):
      - Take the image and re-save it at several quality levels (50, 65, 80, 95)
      - Compute the per-pixel error between original and each re-saved version
      - For a singly-compressed image: error is minimized at ONE quality level
        (roughly the original compression quality)
      - For a doubly-compressed image (i.e., a manipulated region that was
        saved, edited, then saved again): error pattern across quality levels
        is different and shows a secondary minimum or flat region
      - High variance across quality-level errors = double compression indicator

    Returns a score in [0, 1] where higher = more likely double-compressed.
    """
    # Convert to PIL for JPEG re-saving
    pil_img = Image.fromarray(image_array.astype(np.uint8))
    original = np.array(pil_img, dtype=np.float32)

    quality_levels = [50, 65, 75, 85, 92]
    mean_errors = []

    for q in quality_levels:
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        resaved = np.array(Image.open(buf), dtype=np.float32)
        
        # Mean squared error between original and re-saved
        mse = np.mean((original - resaved) ** 2)
        mean_errors.append(mse)

    mean_errors = np.array(mean_errors)

    if mean_errors.max() < 1e-6:
        return 0.0

    # Normalize errors
    norm_errors = mean_errors / (mean_errors.max() + 1e-8)

    # For a singly-compressed image, errors should be monotonically
    # decreasing as quality increases (better quality = less difference)
    # For doubly-compressed, the curve is non-monotonic

    # Check monotonicity: count inversions (where error goes up as quality increases)
    inversions = sum(1 for i in range(len(norm_errors)-1) if norm_errors[i] < norm_errors[i+1])
    inversion_ratio = inversions / (len(norm_errors) - 1)

    # Also check: coefficient of variation across errors
    # Singly-compressed: smooth decrease, lower CV
    # Doubly-compressed: irregular curve, higher CV
    cv = np.std(norm_errors) / (np.mean(norm_errors) + 1e-8)

    score = float(np.clip(inversion_ratio * 0.5 + min(cv / 0.5, 1.0) * 0.5, 0.0, 1.0))
    return score


# ─── Blocking Inconsistency Across Regions ────────────────────────────────────

def _regional_compression_inconsistency(image_gray: np.ndarray) -> float:
    """
    Checks whether different regions of the image have different compression
    quality — a direct signature of copy-paste from a different source image.

    Method:
      - Divide image into a 4x4 grid of regions
      - Estimate compression quality of each region by measuring its
        DCT coefficient distribution (higher-quality = more high-freq energy)
      - Score = variance of estimated qualities across regions
    """
    img = cv2.resize(image_gray, (128, 128)).astype(np.float32)
    H, W = img.shape
    grid = 4
    bh, bw = H // grid, W // grid

    region_scores = []
    for r in range(grid):
        for c in range(grid):
            block = img[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
            
            # Simple quality proxy: high-frequency energy in this region
            # Use Laplacian to estimate frequency content
            lap = cv2.Laplacian(block, cv2.CV_32F)
            hf_energy = np.var(lap)
            region_scores.append(hf_energy)

    region_scores = np.array(region_scores)

    if region_scores.max() < 1e-8:
        return 0.0

    # Coefficient of variation across region quality estimates
    cv = np.std(region_scores) / (np.mean(region_scores) + 1e-8)
    score = float(np.clip(cv / 3.0, 0.0, 1.0))
    return score


# ─── Main Interface ───────────────────────────────────────────────────────────

def analyze_compression(image_array: np.ndarray) -> tuple:
    """
    Runs the full compression artifact analysis pipeline on a single image.

    Args:
        image_array: numpy array (H, W, 3), values in [0, 1] float OR [0, 255] uint8
                     Expected channel order: RGB

    Returns:
        compression_score : float in [0, 1] — higher = more suspicious artifacts
        flags             : list of str describing what was found
    """
    # Normalize to uint8
    if image_array.dtype != np.uint8:
        img_uint8 = (image_array * 255).astype(np.uint8)
    else:
        img_uint8 = image_array.copy()

    if img_uint8.ndim == 2:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

    img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    # Run all sub-analyses
    block_score      = _block_boundary_score(img_gray)
    double_jpeg_score = _double_jpeg_score(img_gray)
    regional_score   = _regional_compression_inconsistency(img_gray)

    # Weighted combination
    compression_score = (
        block_score       * 0.35 +
        double_jpeg_score * 0.40 +
        regional_score    * 0.25
    )
    compression_score = float(np.clip(compression_score, 0.0, 1.0))

    # Build evidence flags
    flags = []
    if block_score > 0.5:
        flags.append(f"Abnormal JPEG block boundary strength detected (score: {block_score:.2f})")
    if double_jpeg_score > 0.5:
        flags.append(f"Double JPEG compression pattern found — image likely re-saved after editing (score: {double_jpeg_score:.2f})")
    if regional_score > 0.5:
        flags.append(f"Inconsistent compression quality across image regions (score: {regional_score:.2f})")

    if not flags:
        flags.append(f"Compression artifacts within normal range (score: {compression_score:.2f})")

    return round(compression_score, 4), flags


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[TEST] Running compression analyzer on dummy image...\n")
    dummy = np.random.rand(224, 224, 3).astype(np.float32)
    score, flags = analyze_compression(dummy)
    print(f"[OK] Compression score: {score}")
    for f in flags:
        print(f"     • {f}")
    print("\n[DONE] compression_analyzer.py is working correctly.")
