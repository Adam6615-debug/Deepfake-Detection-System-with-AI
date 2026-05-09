"""
main.py
-------
Entry point for the deepfake detection system.

Usage:
    python main.py --mode train
    python main.py --mode test
    python main.py --mode predict --image path/to/image.jpg
"""

import argparse
import numpy as np
import os

# ── Part 1 ────────────────────────────────────────────────────────────────────
from dataset_loader import load_dataset, generate_dummy_dataset
from preprocessor   import preprocess_batch, preprocess_single
from model          import build_model, train_model, load_model, predict_single, plot_training_history
from report_stub    import create_report, update_report, add_evidence_flag, compute_final_verdict, print_report_summary, save_report

# ── Part 2 ────────────────────────────────────────────────────────────────────
from analyzer_part2 import run_part2_analysis

# ── Part 3 ────────────────────────────────────────────────────────────────────
from simple_explainer     import analyze_with_simple_explainer
from reverse_image_search import analyze_with_reverse_search
from report_generator     import create_final_report


# ── ONE config used by both train and predict ──────────────────────────────────
# Changing these here updates BOTH paths — they must always match.
USE_ELA     = False   # ELA hurts on standard JPEG datasets; keep False
DETECT_FACE = True    # Always crop to face region


def run_training(use_dummy: bool = False):
    """Full training pipeline."""
    print("\n" + "=" * 55)
    print("  DEEPFAKE DETECTOR — TRAINING MODE")
    print("=" * 55 + "\n")

    if use_dummy:
        X, y = generate_dummy_dataset(n_real=40, n_fake=40)
    else:
        X, y = load_dataset("data")
        if X is None or len(X) < 10:
            print("[INFO] Not enough real data found. Using dummy dataset for demo.")
            X, y = generate_dummy_dataset(n_real=40, n_fake=40)

    print(f"\n[INFO] Preprocessing images (ELA={USE_ELA}, face_detect={DETECT_FACE})...")
    X = preprocess_batch(X, use_ela=USE_ELA, detect_face=DETECT_FACE)

    # FIX: shuffle before split so batches aren't all-real then all-fake
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    model, history = train_model(X, y, epochs=15, batch_size=16)
    plot_training_history(history)

    print("\n[DONE] Training complete. Model saved to models/deepfake_detector.keras")
    return model


def run_predict(image_path: str):
    """
    Full prediction pipeline — Parts 1, 2, and 3 on a single image.
    """
    print("\n" + "=" * 55)
    print("  DEEPFAKE DETECTOR — FULL ANALYSIS (Parts 1, 2, 3)")
    print("=" * 55 + "\n")

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    # ── Part 3: Reverse Image Search ──────────────────────────────────────
    print("[PART 3] Running reverse image search...")
    report = create_report(image_path)
    report = analyze_with_reverse_search(image_path, report)

    # ── Part 1: CNN deepfake score ─────────────────────────────────────────
    print("\n[PART 1] Running deepfake detection (CNN)...")

    try:
        model = load_model()
    except FileNotFoundError:
        print("[INFO] No trained model found — building untrained model for demo.")
        print("       Run 'python main.py --mode train' first for real results.")
        model = build_model()

    from PIL import Image
    # FIX: load as uint8 — do NOT divide by 255 here.
    # preprocess_single handles normalization internally.
    img       = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.uint8)

    # Uses the same USE_ELA / DETECT_FACE flags as training
    processed  = preprocess_single(img_array, use_ela=USE_ELA, detect_face=DETECT_FACE)
    prediction = predict_single(model, processed)

    print(f"       Deepfake score : {prediction['deepfake_score']}")
    print(f"       Label          : {prediction['label']}")
    print(f"       Confidence     : {prediction['confidence']}%")

    report = update_report(report,
        deepfake_score      = prediction["deepfake_score"],
        deepfake_label      = prediction["label"],
        deepfake_confidence = prediction["confidence"]
    )

    # Pass the preprocessed array to the explainer so it uses the exact same
    # input that produced the score above — popped before JSON save.
    report["_preprocessed_image"] = processed

    if prediction["deepfake_score"] >= 0.7:
        report = add_evidence_flag(report, f"High deepfake probability ({prediction['deepfake_score']:.2f})")
    elif prediction["deepfake_score"] >= 0.5:
        report = add_evidence_flag(report, f"Moderate deepfake probability ({prediction['deepfake_score']:.2f})")
    else:
        report = add_evidence_flag(report, f"Low deepfake probability ({prediction['deepfake_score']:.2f}) — likely authentic")

    # ── Part 2: Image forensics ────────────────────────────────────────────
    print("\n[PART 2] Running image forensics analysis...")
    report = run_part2_analysis(image_path, report)

    # ── Part 3: Explainability heatmap ────────────────────────────────────
    print("\n[PART 3] Generating explainability heatmap...")
    report = analyze_with_simple_explainer(image_path, report)

    # ── Final verdict ──────────────────────────────────────────────────────
    report = compute_final_verdict(report)
    print_report_summary(report)

    print("\n[PART 3] Generating final report...")
    report = create_final_report(report, format="both")
    save_report(report)


def run_test():
    """Quick sanity check — Parts 1, 2, and 3 with dummy data."""
    print("\n[TEST] Running full pipeline sanity check (Parts 1, 2, 3)...\n")

    # ── Part 1 ────────────────────────────────────────────────────────────
    print("--- Part 1 (Adam: CNN deepfake detector) ---")
    X, y   = generate_dummy_dataset(n_real=10, n_fake=10)
    X      = preprocess_batch(X, use_ela=USE_ELA, detect_face=False)
    model  = build_model()
    result = predict_single(model, X[0])
    print(f"[OK] Deepfake score: {result['deepfake_score']}  ->  {result['label']}")

    # ── Part 2 ────────────────────────────────────────────────────────────
    print("\n--- Part 2 (Amr: image forensics) ---")
    from splicing_detector    import analyze_splicing
    from ai_content_detector  import analyze_ai_generated
    from compression_analyzer import analyze_compression

    dummy_img          = np.random.rand(224, 224, 3).astype(np.float32)
    splicing_score,    _ = analyze_splicing(dummy_img)
    ai_score,          _ = analyze_ai_generated(dummy_img)
    compression_score, _ = analyze_compression(dummy_img)
    print(f"[OK] Splicing score    : {splicing_score}")
    print(f"[OK] AI-generated score: {ai_score}")
    print(f"[OK] Compression score : {compression_score}")

    # ── Part 3 ────────────────────────────────────────────────────────────
    print("\n--- Part 3 (Mazen: explainability + reports) ---")
    from reverse_image_search import ReverseImageSearch
    from simple_explainer     import create_simple_heatmap
    from report_generator     import generate_html_report

    reverse_search = ReverseImageSearch()
    print("[OK] Reverse image search initialized")

    explainer_result = create_simple_heatmap("dummy_test_image.jpg")
    print(f"[OK] Simple explainability: {'completed' if explainer_result else 'failed (expected without model)'}")

    test_report = create_report("dummy_test_image.jpg")
    test_report = update_report(test_report,
        deepfake_score      = result["deepfake_score"],
        deepfake_label      = result["label"],
        deepfake_confidence = result["confidence"],
        splicing_score      = splicing_score,
        ai_generated_score  = ai_score,
        compression_score   = compression_score
    )
    test_report = compute_final_verdict(test_report)
    html_path   = generate_html_report(test_report)
    print(f"[OK] HTML report generated: {html_path}")

    # ── Combined report ────────────────────────────────────────────────────
    print("\n--- Combined Report ---")
    report = create_report("dummy_test_image.jpg")
    report = update_report(report,
        deepfake_score      = result["deepfake_score"],
        deepfake_label      = result["label"],
        deepfake_confidence = result["confidence"],
        splicing_score      = splicing_score,
        ai_generated_score  = ai_score,
        compression_score   = compression_score
    )
    report = compute_final_verdict(report)
    print_report_summary(report)

    print("[DONE] All components (Part 1 + Part 2 + Part 3) working correctly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Detection System — Parts 1, 2, 3")
    parser.add_argument("--mode",  type=str, default="test",
                        choices=["train", "predict", "test"],
                        help="Run mode: train | predict | test")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image file (required for predict mode)")
    parser.add_argument("--dummy", action="store_true",
                        help="Force dummy data even if real data exists")

    args = parser.parse_args()

    if args.mode == "train":
        run_training(use_dummy=args.dummy)
    elif args.mode == "predict":
        if not args.image:
            print("[ERROR] --image is required for predict mode.")
            print("  Example: python main.py --mode predict --image data/fake/test.jpg")
        else:
            run_predict(args.image)
    elif args.mode == "test":
        run_test()
