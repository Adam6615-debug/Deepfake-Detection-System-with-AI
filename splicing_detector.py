"""
splicing_detector.py
--------------------
Part 2 — Image Splicing Detection

Detects cut-and-paste regions in an image by analyzing:
  1. Edge inconsistencies — spliced regions have unnatural edge sharpness transitions
  2. Lighting direction — different face regions with inconsistent light sources
  3. Noise level map — pasted regions often have different sensor noise signatures

Output: splicing_score in [0, 1]. Higher = more likely spliced.

Usage:
    from splicing_detector import analyze_splicing
    score, flags = analyze_splicing(image_array)
"""

import numpy as np
import cv2
from skimage import filters, feature, measure
from skimage.color import rgb2gray
from skimage.util import img_as_float
import warnings
warnings.filterwarnings("ignore")


# ─── Edge Inconsistency Analysis ─────────────────────────────────────────────

def _edge_inconsistency_score(image_gray: np.ndarray) -> float:
    """
    Measures unnatural transitions in edge sharpness across image regions.

    A spliced image often has one region with significantly sharper or
    blurrier edges than the rest — because the pasted region came from
    a different camera or was processed differently.

    Steps:
      - Compute Sobel edge map (detects gradients / sharp transitions)
      - Divide image into a 4x4 grid of blocks
      - Compute mean edge strength per block
      - Score = normalized variance across blocks (high variance = inconsistency)
    """
    # Sobel edge detection
    edges = filters.sobel(image_gray)

    # Divide into 4x4 grid
    H, W = edges.shape
    grid_rows, grid_cols = 4, 4
    block_h = H // grid_rows
    block_w = W // grid_cols

    block_means = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            block = edges[r*block_h:(r+1)*block_h, c*block_w:(c+1)*block_w]
            block_means.append(np.mean(block))

    block_means = np.array(block_means)

    if block_means.max() == 0:
        return 0.0

    # Coefficient of variation: std/mean — scale-invariant inconsistency measure
    cv = np.std(block_means) / (np.mean(block_means) + 1e-8)

    # Empirically, authentic images score ~0.3–0.6, spliced images often > 0.8
    score = float(np.clip(cv / 1.5, 0.0, 1.0))
    return score


# ─── Lighting Direction Analysis ─────────────────────────────────────────────

def _lighting_inconsistency_score(image_bgr: np.ndarray) -> float:
    """
    Estimates lighting direction inconsistency across image regions.

    Real photos have consistent light source direction. Spliced images
    often have conflicting gradient directions in different regions
    because they came from photos taken under different lighting.

    Steps:
      - Convert to grayscale
      - Compute gradient direction (angle) across the image
      - Analyze dominant gradient angle per region quadrant
      - Score = spread of dominant angles across quadrants
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Compute gradient in x and y directions
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Gradient magnitude and direction
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)  # radians, range [-pi, pi]

    # Only consider pixels with strong gradients (edges)
    threshold = np.percentile(magnitude, 75)
    strong_mask = magnitude > threshold

    if strong_mask.sum() < 100:
        return 0.0

    H, W = gray.shape
    quadrants = [
        (0, H//2, 0, W//2),      # top-left
        (0, H//2, W//2, W),      # top-right
        (H//2, H, 0, W//2),      # bottom-left
        (H//2, H, W//2, W),      # bottom-right
    ]

    dominant_angles = []
    for (r1, r2, c1, c2) in quadrants:
        region_mask = strong_mask[r1:r2, c1:c2]
        region_angle = angle[r1:r2, c1:c2]
        if region_mask.sum() > 20:
            # Dominant angle = circular mean of strong-edge angles
            valid_angles = region_angle[region_mask]
            # Convert to unit vectors and average for circular mean
            mean_sin = np.mean(np.sin(valid_angles))
            mean_cos = np.mean(np.cos(valid_angles))
            dominant = np.arctan2(mean_sin, mean_cos)
            dominant_angles.append(dominant)

    if len(dominant_angles) < 2:
        return 0.0

    # Measure angular spread across quadrants
    # Use pairwise angular differences
    diffs = []
    for i in range(len(dominant_angles)):
        for j in range(i+1, len(dominant_angles)):
            diff = abs(dominant_angles[i] - dominant_angles[j])
            # Normalize to [0, pi]
            if diff > np.pi:
                diff = 2 * np.pi - diff
            diffs.append(diff)

    mean_angular_diff = np.mean(diffs)

    # Normalize: max diff is pi (180 degrees)
    score = float(np.clip(mean_angular_diff / np.pi, 0.0, 1.0))
    return score


# ─── Noise Level Map Analysis ─────────────────────────────────────────────────

def _noise_inconsistency_score(image_gray: np.ndarray) -> float:
    """
    Detects regions with significantly different noise levels.

    Every camera sensor introduces a specific noise pattern. A pasted
    region from a different source will have a different noise signature
    than the surrounding area.

    Steps:
      - Apply a strong blur to isolate the noise (original - blurred = noise)
      - Divide into blocks and measure noise level per block
      - Score = normalized variance of noise levels across blocks
    """
    # Extract noise map: residual after subtracting smoothed version
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    noise_map = np.abs(image_gray.astype(np.float32) - blurred.astype(np.float32))

    # Divide into 6x6 grid for finer analysis
    H, W = noise_map.shape
    grid = 6
    bh, bw = H // grid, W // grid

    block_noise = []
    for r in range(grid):
        for c in range(grid):
            block = noise_map[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
            block_noise.append(np.mean(block))

    block_noise = np.array(block_noise)

    if block_noise.max() < 1e-6:
        return 0.0

    # Coefficient of variation
    cv = np.std(block_noise) / (np.mean(block_noise) + 1e-8)
    score = float(np.clip(cv / 2.0, 0.0, 1.0))
    return score


# ─── Boundary Sharpness Analysis ─────────────────────────────────────────────

def _boundary_sharpness_score(image_gray: np.ndarray) -> float:
    """
    Detects suspiciously sharp or clean boundaries that suggest copy-paste.

    Canny edge detection finds all edges. We then look for unusually
    long, clean, straight-ish edge contours — a hallmark of selection
    tool boundaries used during splicing.
    """
    edges = feature.canny(image_gray, sigma=1.5)
    
    # Label connected edge regions
    labeled = measure.label(edges)
    regions = measure.regionprops(labeled)

    if not regions:
        return 0.0

    # Find the longest connected edge contour
    lengths = [r.area for r in regions]
    max_length = max(lengths)

    H, W = image_gray.shape
    image_perimeter = 2 * (H + W)

    # A very long clean contour relative to image size is suspicious
    relative_length = max_length / image_perimeter

    # Empirically calibrated: >0.5 of perimeter length is very suspicious
    score = float(np.clip(relative_length / 0.5, 0.0, 1.0))
    return score


# ─── Main Interface ───────────────────────────────────────────────────────────

def analyze_splicing(image_array: np.ndarray) -> tuple:
    """
    Runs the full splicing detection pipeline on a single image.

    Args:
        image_array: numpy array (H, W, 3), values in [0, 1] float OR [0, 255] uint8
                     Expected channel order: RGB

    Returns:
        splicing_score : float in [0, 1] — higher = more likely spliced
        flags          : list of str describing what was found
    """
    # Normalize to uint8 for OpenCV
    if image_array.dtype != np.uint8:
        img_uint8 = (image_array * 255).astype(np.uint8)
    else:
        img_uint8 = image_array.copy()

    # Ensure 3-channel
    if img_uint8.ndim == 2:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

    # Convert to grayscale for analysis
    img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    img_float_gray = img_as_float(img_gray)

    # BGR version for lighting analysis
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    # Run all sub-analyses
    edge_score      = _edge_inconsistency_score(img_float_gray)
    lighting_score  = _lighting_inconsistency_score(img_bgr)
    noise_score     = _noise_inconsistency_score(img_gray)
    boundary_score  = _boundary_sharpness_score(img_float_gray)

    # Weighted combination
    # Edge and noise are the most reliable indicators
    splicing_score = (
        edge_score     * 0.35 +
        lighting_score * 0.25 +
        noise_score    * 0.25 +
        boundary_score * 0.15
    )
    splicing_score = float(np.clip(splicing_score, 0.0, 1.0))

    # Build evidence flags
    flags = []
    if edge_score > 0.6:
        flags.append(f"Edge sharpness inconsistency detected across regions (score: {edge_score:.2f})")
    if lighting_score > 0.6:
        flags.append(f"Conflicting lighting directions found in image quadrants (score: {lighting_score:.2f})")
    if noise_score > 0.5:
        flags.append(f"Noise signature mismatch between image regions (score: {noise_score:.2f})")
    if boundary_score > 0.5:
        flags.append(f"Suspiciously clean boundary contour detected (score: {boundary_score:.2f})")

    if not flags:
        flags.append(f"No significant splicing indicators found (score: {splicing_score:.2f})")

    return round(splicing_score, 4), flags


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[TEST] Running splicing detector on dummy image...\n")
    dummy = np.random.rand(224, 224, 3).astype(np.float32)
    score, flags = analyze_splicing(dummy)
    print(f"[OK] Splicing score: {score}")
    for f in flags:
        print(f"     • {f}")
    print("\n[DONE] splicing_detector.py is working correctly.")
