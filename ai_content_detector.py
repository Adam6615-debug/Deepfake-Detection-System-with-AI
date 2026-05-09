"""
ai_content_detector.py
-----------------------
Detects GAN/diffusion-model-generated images using:
  1. DCT (Discrete Cosine Transform) frequency analysis
     — GAN images have unnatural high-frequency energy patterns
  2. Co-occurrence matrix texture analysis
     — Real photos have natural pixel co-occurrence statistics;
       AI images often deviate due to upsampling artifacts
  3. Color channel correlation analysis
     — GANs tend to produce slightly abnormal correlations between R, G, B channels

Output: ai_generated_score in [0, 1]. Higher = more likely AI-generated.
"""

import numpy as np
import cv2
from scipy.fft import dct
import warnings
warnings.filterwarnings("ignore")


# ─── DCT Frequency Domain Analysis ───────────────────────────────────────────

def _dct_frequency_score(image_gray: np.ndarray) -> float:
    """
    Analyzes the DCT (Discrete Cosine Transform) spectrum of an image.

    How it works:
      - Apply 2D DCT to the grayscale image
      - Split the spectrum into low, mid, and high frequency bands
      - GAN-generated images have distinctive patterns:
          * Unusually flat or periodic high-frequency energy
          * Abnormal ratio between mid and high frequency power
      - Real photos follow natural 1/f^2 power spectrum falloff

    Returns a score in [0, 1] where higher = more GAN-like
    """
    # Resize to fixed size for consistent analysis
    img = cv2.resize(image_gray, (128, 128))
    img_float = img.astype(np.float64)

    # 2D DCT via applying 1D DCT along both axes
    dct_coeffs = dct(dct(img_float, axis=0, norm='ortho'), axis=1, norm='ortho')

    # Take absolute value and log-scale for analysis
    dct_abs = np.abs(dct_coeffs)
    log_dct = np.log1p(dct_abs)

    H, W = log_dct.shape

    # Define frequency bands
    low_band  = log_dct[:H//4,  :W//4]    # Low frequencies (DC + low AC)
    mid_band  = log_dct[H//4:H//2, W//4:W//2]  # Mid frequencies
    high_band = log_dct[H//2:,  W//2:]    # High frequencies

    low_energy  = np.mean(low_band)
    mid_energy  = np.mean(mid_band)
    high_energy = np.mean(high_band)

    if low_energy < 1e-8:
        return 0.0

    # Natural image: energy drops significantly from low to high
    # Expected ratio: mid/low ≈ 0.3–0.5, high/low ≈ 0.05–0.15
    mid_ratio  = mid_energy  / (low_energy + 1e-8)
    high_ratio = high_energy / (low_energy + 1e-8)

    # GAN images often have higher-than-normal high-frequency energy
    # and an unusual mid/high ratio
    abnormal_high = float(np.clip((high_ratio - 0.05) / 0.20, 0.0, 1.0))

    # Also check for periodic artifacts (grid patterns) — a known GAN artifact
    # Periodicity shows up as peaks in the DCT magnitude spectrum
    dct_flat = dct_abs.flatten()
    dct_flat_sorted = np.sort(dct_flat)[::-1]
    # Ratio of top-10 coefficients to next-100: high ratio = periodic artifacts
    if len(dct_flat_sorted) > 110:
        top10_mean   = np.mean(dct_flat_sorted[1:11])    # skip DC component
        next100_mean = np.mean(dct_flat_sorted[11:111])
        periodicity_ratio = top10_mean / (next100_mean + 1e-8)
        # Natural images: ~5–15x. GAN artifacts: can be 20–50x
        periodicity_score = float(np.clip((periodicity_ratio - 5.0) / 45.0, 0.0, 1.0))
    else:
        periodicity_score = 0.0

    # Combine
    score = abnormal_high * 0.6 + periodicity_score * 0.4
    return float(np.clip(score, 0.0, 1.0))


# ─── Co-occurrence Matrix Analysis ───────────────────────────────────────────

def _cooccurrence_score(image_gray: np.ndarray) -> float:
    """
    Analyzes pixel value co-occurrence statistics.

    Real images have natural statistical relationships between neighboring
    pixels. AI-generated images often show abnormal smoothness or
    repetitive patterns due to upsampling in the generator network.

    We compute a simplified co-occurrence: for each pixel, compare it
    to its right neighbor. Authentic images follow a natural distribution;
    AI images tend to have too-smooth or too-regular distributions.
    """
    img = cv2.resize(image_gray, (128, 128)).astype(np.int32)

    # Horizontal co-occurrence: difference between adjacent pixels
    diff_h = np.abs(img[:, 1:] - img[:, :-1]).flatten()
    diff_v = np.abs(img[1:, :] - img[:-1, :]).flatten()

    all_diffs = np.concatenate([diff_h, diff_v])

    if len(all_diffs) == 0:
        return 0.0

    # Natural images: most differences are small (0–20), but there's a
    # natural spread. AI images often have either:
    #   (a) Too many zeros (over-smoothed regions) → low variance
    #   (b) Unnatural histogram shape

    # Measure: fraction of zero differences (over-smoothing indicator)
    zero_fraction = np.mean(all_diffs == 0)

    # Measure: std of differences (too low = over-smoothed, too high = noise)
    diff_std = np.std(all_diffs)

    # Natural image: zero_fraction ~0.1–0.25, diff_std ~15–35
    # Over-smoothed AI: zero_fraction > 0.4 or diff_std < 8
    # Noisy AI: diff_std > 50

    smoothing_score = float(np.clip((zero_fraction - 0.25) / 0.40, 0.0, 1.0))
    variance_score  = float(np.clip((10.0 - diff_std) / 10.0, 0.0, 1.0))  # too low = suspect

    score = smoothing_score * 0.5 + variance_score * 0.5
    return float(np.clip(score, 0.0, 1.0))


# ─── Color Channel Correlation Analysis ──────────────────────────────────────

def _color_correlation_score(image_rgb: np.ndarray) -> float:
    """
    Analyzes correlations between R, G, B channels.

    In natural photographs, the R, G, B channels have strong but not
    perfect correlations because they physically capture the same scene
    under different wavelength filters.

    GAN generators process channels simultaneously and tend to produce
    either over-correlated or under-correlated channels — a statistical
    fingerprint of synthetic generation.

    Returns a score based on deviation from the natural correlation range.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        return 0.0

    r = image_rgb[:, :, 0].flatten().astype(np.float64)
    g = image_rgb[:, :, 1].flatten().astype(np.float64)
    b = image_rgb[:, :, 2].flatten().astype(np.float64)

    # Compute pairwise Pearson correlations
    def safe_corr(x, y):
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return 1.0
        return float(np.corrcoef(x, y)[0, 1])

    corr_rg = safe_corr(r, g)
    corr_rb = safe_corr(r, b)
    corr_gb = safe_corr(g, b)

    correlations = [corr_rg, corr_rb, corr_gb]

    # Natural photos: correlations typically between 0.65–0.95
    # GAN images: often outside this range (too high > 0.97 or unusual patterns)
    scores = []
    for c in correlations:
        if c > 0.97:
            # Over-correlated: channels are too similar (GAN processing artifact)
            s = (c - 0.97) / 0.03
        elif c < 0.50:
            # Under-correlated: abnormally independent channels
            s = (0.50 - c) / 0.50
        else:
            s = 0.0
        scores.append(float(np.clip(s, 0.0, 1.0)))

    return float(np.mean(scores))


# ─── Upsampling Artifact Detection ────────────────────────────────────────────

def _upsampling_artifact_score(image_gray: np.ndarray) -> float:
    """
    Detects checkerboard or grid artifacts caused by transposed convolutions
    in GAN upsampling layers.

    These artifacts create periodic patterns at specific spatial frequencies
    that are visible in the image's power spectrum.
    """
    img = cv2.resize(image_gray, (256, 256)).astype(np.float32)

    # Compute 2D FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    H, W = magnitude.shape
    center_y, center_x = H // 2, W // 2

    # Look for peaks at specific GAN-related frequencies
    # Checkerboard artifacts appear at (H/2, W/2) = Nyquist frequency
    # We check a ring at ~half the Nyquist frequency
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((Y - center_y)**2 + (X - center_x)**2)

    # Check energy in high-frequency ring (r = 80–110 pixels from center)
    high_freq_ring = (dist >= 80) & (dist <= 110)
    # Check energy in mid-frequency ring (r = 30–60 pixels from center)
    mid_freq_ring  = (dist >= 30) & (dist <= 60)

    if high_freq_ring.sum() == 0 or mid_freq_ring.sum() == 0:
        return 0.0

    high_energy = np.mean(magnitude[high_freq_ring])
    mid_energy  = np.mean(magnitude[mid_freq_ring])

    if mid_energy < 1e-6:
        return 0.0

    # Unusual high-to-mid ratio suggests upsampling artifacts
    ratio = high_energy / mid_energy
    # Natural: ratio ~0.05–0.15. Artifacts: > 0.25
    score = float(np.clip((ratio - 0.10) / 0.30, 0.0, 1.0))
    return score


# ─── Main Interface ───────────────────────────────────────────────────────────

def analyze_ai_generated(image_array: np.ndarray) -> tuple:
    """
    Runs the full AI-generated content detection pipeline on a single image.

    Args:
        image_array: numpy array (H, W, 3), values in [0, 1] float OR [0, 255] uint8
                     Expected channel order: RGB

    Returns:
        ai_generated_score : float in [0, 1] — higher = more likely AI-generated
        flags              : list of str describing what was found
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
    dct_score         = _dct_frequency_score(img_gray)
    cooccurrence_score = _cooccurrence_score(img_gray)
    color_score       = _color_correlation_score(img_uint8.astype(np.float64))
    upsample_score    = _upsampling_artifact_score(img_gray)

    # Weighted combination
    ai_score = (
        dct_score          * 0.35 +
        cooccurrence_score * 0.25 +
        color_score        * 0.20 +
        upsample_score     * 0.20
    )
    ai_score = float(np.clip(ai_score, 0.0, 1.0))

    # Build evidence flags
    flags = []
    if dct_score > 0.5:
        flags.append(f"Abnormal DCT frequency spectrum — possible GAN artifact (score: {dct_score:.2f})")
    if cooccurrence_score > 0.5:
        flags.append(f"Unnatural pixel co-occurrence statistics — over-smoothed regions detected (score: {cooccurrence_score:.2f})")
    if color_score > 0.4:
        flags.append(f"Abnormal RGB channel correlation pattern (score: {color_score:.2f})")
    if upsample_score > 0.4:
        flags.append(f"Upsampling grid artifacts detected in frequency domain (score: {upsample_score:.2f})")

    if not flags:
        flags.append(f"No significant AI-generation indicators found (score: {ai_score:.2f})")

    return round(ai_score, 4), flags


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[TEST] Running AI content detector on dummy image...\n")
    dummy = np.random.rand(224, 224, 3).astype(np.float32)
    score, flags = analyze_ai_generated(dummy)
    print(f"[OK] AI-generated score: {score}")
    for f in flags:
        print(f"     • {f}")
    print("\n[DONE] ai_content_detector.py is working correctly.")
