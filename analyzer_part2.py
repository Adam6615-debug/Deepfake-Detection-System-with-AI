"""
analyzer_part2.py
-----------------
  - Imports preprocessor and report_stub
  - Runs splicing detection, AI-content detection, and compression analysis
"""
import argparse
import os
import sys
import numpy as np
from splicing_detector   import analyze_splicing
from ai_content_detector import analyze_ai_generated
from compression_analyzer import analyze_compression
from PIL import Image

try:
    from preprocessor import preprocess_single
    from report_stub  import (create_report, update_report, add_evidence_flag,
                               compute_final_verdict, print_report_summary, save_report)
    PART1_AVAILABLE = True
except ImportError:
    print("[WARNING] (preprocessor.py, report_stub.py) not found.")
    PART1_AVAILABLE = False

def run_part2_analysis(image_path: str, report: dict = None) -> dict:
    """
    Runs all analyses on a single image and fills the report.

    Args:
        image_path : path to the image file (JPG or PNG)
        report     : existing report dict (optional).
                     If None, creates a new empty report.

    Returns:
        report: dict with splicing_score, ai_generated_score,
                compression_score, and evidence_flags filled in.
    """
    print("\n" + "=" * 55)
    print("  PART 2 ANALYSIS — Image Forensics")
    print("=" * 55)

    # ── Load image ────────────────────────────────────────────────────────
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return report or {}

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0

    # ── Preprocess () ───────────────────
    if PART1_AVAILABLE:
        print("[INFO] Preprocessing (ELA + face detection)...")
        processed = preprocess_single(img_array, use_ela=False, detect_face=True)
        # Note: use_ela=False here because Part 2 analyses need the original
        # pixel structure, not the ELA-transformed version
    else:
        processed = img_array

    # ── Create report if not provided ─────────────────────────────────────
    if report is None:
        if PART1_AVAILABLE:
            report = create_report(image_path)
        else:
            report = {"image_path": image_path, "filename": os.path.basename(image_path), "evidence_flags": []}

    # ── Run analyses ───────────────────────────────────────────────

    # 1. Image Splicing Detection
    print("\n[1/3] Running splicing detection...")
    splicing_score, splicing_flags = analyze_splicing(processed)
    print(f"      Splicing score: {splicing_score:.4f}")

    # 2. AI-Generated Content Detection
    print("[2/3] Running AI-content detection (DCT analysis)...")
    ai_score, ai_flags = analyze_ai_generated(processed)
    print(f"      AI-generated score: {ai_score:.4f}")

    # 3. Compression Artifact Analysis
    print("[3/3] Running compression artifact analysis...")
    compression_score, compression_flags = analyze_compression(processed)
    print(f"      Compression score: {compression_score:.4f}")

    # ── Fill report ───────────────────────────────────────────────────────
    if PART1_AVAILABLE:
        report = update_report(report,
            splicing_score      = splicing_score,
            ai_generated_score  = ai_score,
            compression_score   = compression_score
        )
        for flag in splicing_flags + ai_flags + compression_flags:
            report = add_evidence_flag(report, flag)
    else:
        # Fallback: fill dict directly
        report["splicing_score"]     = splicing_score
        report["ai_generated_score"] = ai_score
        report["compression_score"]  = compression_score
        report["evidence_flags"].extend(splicing_flags + ai_flags + compression_flags)

    print("\n[INFO] Part 2 analysis complete.")
    return report

def run_test():
    """Sanity check with a random dummy image — no real image needed."""
    print("\n[TEST] Running Part 2 pipeline sanity check with dummy data...\n")

    dummy_array = np.random.rand(224, 224, 3).astype(np.float32)

    splicing_score, s_flags     = analyze_splicing(dummy_array)
    ai_score,       ai_flags    = analyze_ai_generated(dummy_array)
    comp_score,     comp_flags  = analyze_compression(dummy_array)

    print(f"[OK] Splicing score    : {splicing_score}")
    print(f"[OK] AI-generated score: {ai_score}")
    print(f"[OK] Compression score : {comp_score}")
    print("\n[DONE] All Part 2 modules working correctly.")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Detection — Part 2 (Image Forensics)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image file to analyze")
    parser.add_argument("--test",  action="store_true",
                        help="Run sanity check with dummy data (no image needed)")
    args = parser.parse_args()

    if args.test:
        run_test()
    else:
        print("Usage:")
        print("  python analyzer_part2.py --test")
        print("  python analyzer_part2.py --image path/to/image.jpg")
