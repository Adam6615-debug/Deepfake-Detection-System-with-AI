"""
report_stub.py
--------------
Defines the unified report structure for the deepfake detection system.

THIS FILE IS THE CONTRACT BETWEEN ALL 3 TEAM MEMBERS.

- Adam (Part 1) fills in:     deepfake_score
- Amr (Part 2) fills:  splicing_score, ai_generated_score, compression_score
- Mazen (Part 3) fills:  shap_explanation, blink_score, final_verdict
  and generates the final HTML/PDF report.
"""

import json
import os
from datetime import datetime


REPORTS_DIR = "outputs/reports"


def create_report(image_path: str) -> dict:
    """
    Creates an empty report template for a given image.
    All analysis scores start as None — teammates fill them in.

    Args:
        image_path: path to the image being analyzed

    Returns:
        report: dict following the project's standard report format
    """
    report = {
        # ── Metadata ──────────────────────────────────────────────────────────
        "image_path":    image_path,
        "filename":      os.path.basename(image_path),
        "analyzed_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version":       "1.0",

        # ── Part 1: Adam — Deepfake Detection (CNN model) ─────────────────────
        # Score in [0, 1]. Higher = more likely fake.
        "deepfake_score": None,   # float, e.g. 0.93
        "deepfake_label": None,   # str, "REAL" or "FAKE"
        "deepfake_confidence": None,  # float, e.g. 93.0 (percentage)

        # ── Part 2: Teammate 1 — Image Analysis ───────────────────────────────
        # Image splicing: detects cut-and-paste regions
        "splicing_score": None,   # float [0, 1]

        # AI-generated content: statistical frequency domain analysis
        "ai_generated_score": None,   # float [0, 1]

        # JPEG compression artifacts: block boundary irregularities
        "compression_score": None,    # float [0, 1]

        # ── Part 3: Teammate 2 — Explainability + Final Output ─────────────────
        # SHAP explanation: path to heatmap image showing suspicious regions
        "shap_heatmap_path": None,    # str, path to PNG file

        # Blink analysis (BONUS): checks for unnatural blink patterns in video
        "blink_score": None,          # float [0, 1] or None if input is image (not video)
        "blink_frames_analyzed": None,  # int, number of video frames checked

        # ── Final Verdict (computed by Teammate 2 from all scores) ────────────
        "final_verdict":     None,    # str, "AUTHENTIC" or "MANIPULATED"
        "final_confidence":  None,    # float, overall confidence percentage
        "manipulation_type": None,    # str, e.g. "Deepfake", "Spliced", "AI-Generated", "Mixed"
        "risk_level":        None,    # str, "LOW", "MEDIUM", "HIGH", "CRITICAL"

        # ── Evidence Summary (filled by Teammate 2) ───────────────────────────
        "evidence_flags": [],         # list of str describing what was found
        "report_path":    None,       # str, path to final HTML/PDF report
    }

    return report


def update_report(report: dict, **kwargs) -> dict:
    """
    Updates a report dict with new values.
    Each team member calls this to add their analysis results.

    Args:
        report: existing report dict
        **kwargs: any report fields to update

    Returns:
        Updated report dict

    Example:
        report = update_report(report, deepfake_score=0.87, deepfake_label="FAKE")
        report = update_report(report, splicing_score=0.34, compression_score=0.21)
    """
    for key, value in kwargs.items():
        if key not in report:
            print(f"[WARNING] Unknown report field: '{key}' — adding anyway.")
        report[key] = value
    return report


def add_evidence_flag(report: dict, flag: str) -> dict:
    """
    Appends an evidence flag string to the report.

    Args:
        report: existing report dict
        flag: descriptive string, e.g. "High deepfake probability (0.93)"

    Returns:
        Updated report dict
    """
    report["evidence_flags"].append(flag)
    return report


def compute_final_verdict(report: dict) -> dict:
    """
    Computes the final verdict from all available sub-scores.
    Teammate 2 calls this after all scores are filled in.

    Weights:
        deepfake_score   : 40%
        splicing_score   : 25%
        ai_generated_score: 20%
        compression_score: 15%

    Args:
        report: report dict with scores filled in

    Returns:
        Report with final_verdict, final_confidence, risk_level filled in
    """
    scores   = []
    weights  = []

    score_weights = [
        ("deepfake_score",      0.40),
        ("splicing_score",      0.25),
        ("ai_generated_score",  0.20),
        ("compression_score",   0.15),
    ]

    for field, weight in score_weights:
        val = report.get(field)
        if val is not None:
            scores.append(val)
            weights.append(weight)

    if not scores:
        report["final_verdict"] = "UNKNOWN"
        report["risk_level"]    = "UNKNOWN"
        return report

    # Normalize weights in case some scores are missing
    total_weight = sum(weights)
    normalized   = [w / total_weight for w in weights]
    final_score  = sum(s * w for s, w in zip(scores, normalized))

    # Determine verdict
    if final_score >= 0.7:
        verdict = "MANIPULATED"
    elif final_score >= 0.4:
        verdict = "SUSPICIOUS"
    else:
        verdict = "AUTHENTIC"

    # Risk level
    if final_score >= 0.85:
        risk = "CRITICAL"
    elif final_score >= 0.7:
        risk = "HIGH"
    elif final_score >= 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    report["final_verdict"]    = verdict
    report["final_confidence"] = round(final_score * 100, 1)
    report["risk_level"]       = risk

    return report


def save_report(report: dict, output_dir: str = REPORTS_DIR) -> str:
    """
    Saves the report as a JSON file.

    Args:
        report: report dict
        output_dir: directory to save into

    Returns:
        Path to saved JSON file
    """
    os.makedirs(output_dir, exist_ok=True)

    safe_name = report["filename"].replace(".", "_").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"report_{safe_name}_{timestamp}.json"
    filepath  = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[INFO] Report saved to: {filepath}")
    report["report_path"] = filepath
    return filepath


def print_report_summary(report: dict):
    """
    Prints a human-readable summary of the report to console.

    Args:
        report: report dict
    """
    print("\n" + "=" * 55)
    print("  DEEPFAKE DETECTION REPORT")
    print("=" * 55)
    print(f"  File       : {report['filename']}")
    print(f"  Analyzed   : {report['analyzed_at']}")
    print("-" * 55)
    print(f"  Deepfake score      : {_fmt(report['deepfake_score'])}")
    print(f"  Splicing score      : {_fmt(report['splicing_score'])}")
    print(f"  AI-generated score  : {_fmt(report['ai_generated_score'])}")
    print(f"  Compression score   : {_fmt(report['compression_score'])}")
    print(f"  Blink score (bonus) : {_fmt(report['blink_score'])}")
    print("-" * 55)
    print(f"  FINAL VERDICT  : {report.get('final_verdict', 'N/A')}")
    print(f"  RISK LEVEL     : {report.get('risk_level', 'N/A')}")
    print(f"  CONFIDENCE     : {report.get('final_confidence', 'N/A')}%")
    if report["evidence_flags"]:
        print("\n  Evidence flags:")
        for flag in report["evidence_flags"]:
            print(f"    • {flag}")
    print("=" * 55 + "\n")


def _fmt(value):
    if value is None:
        return "pending..."
    return f"{value:.4f}"


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[TEST] Creating and updating a sample report...\n")

    report = create_report("data/fake/sample_001.jpg")
    report = update_report(report, deepfake_score=0.91, deepfake_label="FAKE", deepfake_confidence=91.0)
    report = add_evidence_flag(report, "High deepfake probability detected (0.91)")
    report = add_evidence_flag(report, "Face inconsistencies found in eye region")

    # Simulate teammate 1 filling in their scores
    report = update_report(report, splicing_score=0.44, ai_generated_score=0.67, compression_score=0.38)
    report = add_evidence_flag(report, "Compression artifacts detected at image boundaries")

    # Compute final verdict
    report = compute_final_verdict(report)
    print_report_summary(report)

    filepath = save_report(report)
    print(f"[DONE] Report saved: {filepath}")
