"""
simple_explainer.py
-------------------
Simple explainability for deepfake detection model.
Uses gradient-based visualization that works with any model architecture.
No complex dependencies or nested model issues.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from model import load_model, build_model


def create_simple_heatmap(
    image_path: str,
    model_path: str = "models/deepfake_detector.keras",
    known_score: float = None,
    preprocessed_image: np.ndarray = None
) -> str:
    """
    Generate simple gradient-based heatmap for deepfake model prediction.

    Args:
        image_path:         Path to the input image (used for display + output filename)
        model_path:         Path to trained model
        known_score:        The deepfake_score already computed by main.py.
                            If supplied, the label panel shows this score instead
                            of running a second independent prediction.
        preprocessed_image: The float32 (224,224,3) array already fed to
                            predict_single in main.py. If supplied, gradients are
                            computed on this exact tensor so the heatmap is
                            consistent with the report score.

    Returns:
        str: Path to saved heatmap image
    """
    try:
        # Load model
        if os.path.exists(model_path):
            model = load_model()
        else:
            print("[WARNING] No trained model found for explainability analysis")
            return None

        # ── Choose input array ─────────────────────────────────────────────
        if preprocessed_image is not None:
            # Use exactly what main.py fed to predict_single
            img_array = preprocessed_image.astype(np.float32)
        else:
            # Fallback: simple load — only used when called standalone
            img = Image.open(image_path).convert("RGB").resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension and convert to tensor
        img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0))

        # ── Gradient computation ───────────────────────────────────────────
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = model(img_tensor)
            output_node = predictions[0][1] if predictions.shape[1] > 1 else predictions[0][0]

        gradients = tape.gradient(output_node, img_tensor)

        # ── Score to display: prefer the authoritative one from main.py ────
        display_score = known_score if known_score is not None else float(output_node)

        if gradients is not None:
            grad_array = gradients[0].numpy()
            return _create_gradient_visualization(img_array, grad_array, image_path, display_score)
        else:
            print("[EXPLAINER] Gradient computation failed")
            return None

    except Exception as e:
        print(f"[ERROR] Simple explainability failed: {str(e)}")
        return None


def _create_gradient_visualization(img_array, grad_array, image_path, fake_score):
    """Create visualization from gradients."""
    try:
        # Compute gradient magnitude
        grad_magnitude = np.abs(grad_array).mean(axis=-1)

        # Normalize for visualization
        grad_normalized = (grad_magnitude - grad_magnitude.min()) / (grad_magnitude.max() - grad_magnitude.min() + 1e-8)

        # Determine label and confidence correctly
        prediction_label = "FAKE" if fake_score >= 0.5 else "REAL"
        confidence = fake_score if fake_score >= 0.5 else 1.0 - fake_score
        label_color = "red" if prediction_label == "FAKE" else "green"

        # Load the ORIGINAL image (full photo, no crop) purely for display in Panel 1.
        # img_array is the face-cropped preprocessed version used for gradients —
        # showing that as "Original Image" made it look zoomed in.
        try:
            original_display = np.array(
                Image.open(image_path).convert("RGB").resize((224, 224)),
                dtype=np.float32
            ) / 255.0
        except Exception:
            original_display = img_array  # fallback if path unavailable

        # Create visualization
        plt.figure(figsize=(15, 5))

        # Original image — full photo, not the face crop
        plt.subplot(1, 4, 1)
        plt.imshow(original_display)
        plt.title("Original Image")
        plt.axis('off')

        # Prediction info panel
        plt.subplot(1, 4, 2)
        plt.text(0.5, 0.75, f"Deepfake Score: {fake_score:.3f}",
                ha='center', va='center', fontsize=13, fontweight='bold')
        plt.text(0.5, 0.55, f"Prediction: {prediction_label}",
                ha='center', va='center', fontsize=14, fontweight='bold',
                color=label_color)
        plt.text(0.5, 0.35, f"Confidence: {confidence * 100:.1f}%",
                ha='center', va='center', fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')

        # Gradient magnitude heatmap
        plt.subplot(1, 4, 3)
        plt.imshow(grad_magnitude, cmap='hot')
        plt.title("Gradient Magnitude\n(Yellow = High Influence)")
        plt.axis('off')

        # Overlay — on the face-cropped img_array (same spatial space as gradients)
        plt.subplot(1, 4, 4)
        overlay = img_array.copy()
        high_influence = grad_normalized > 0.7
        overlay[high_influence] = [1, 0.2, 0.2]
        plt.imshow(overlay)
        plt.title("Important Regions\n(Red = High Influence)")
        plt.axis('off')

        # Save the heatmap — always overwrite so no stale file persists
        os.makedirs("outputs/heatmaps", exist_ok=True)
        heatmap_path = f"outputs/heatmaps/gradient_{os.path.basename(image_path)}"
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[EXPLAINER] Heatmap saved to: {heatmap_path}")
        return heatmap_path

    except Exception as e:
        print(f"[ERROR] Visualization failed: {str(e)}")
        return None


def analyze_with_simple_explainer(image_path: str, report: dict) -> dict:
    """
    Run simple explainability analysis and update report with heatmap path.

    Reads `deepfake_score` and `_preprocessed_image` from the report so the
    heatmap label always matches what main.py computed — not an independent
    second prediction on a differently-preprocessed image.

    Args:
        image_path: Path to input image
        report: Report dictionary (must already contain deepfake_score)

    Returns:
        dict: Updated report dictionary
    """
    print("[EXPLAINER] Generating explainability heatmap...")

    known_score        = report.get("deepfake_score", None)
    preprocessed_image = report.get("_preprocessed_image", None)

    heatmap_path = create_simple_heatmap(
        image_path,
        known_score=known_score,
        preprocessed_image=preprocessed_image
    )

    if heatmap_path:
        report["shap_heatmap_path"] = heatmap_path
        report["explainability_method"] = "Gradient Analysis"
        print("[EXPLAINER] Explainability analysis complete")
    else:
        report["shap_heatmap_path"] = None
        report["explainability_method"] = "Failed"
        print("[EXPLAINER] Explainability analysis failed")

    # Remove internal key — don't serialise a numpy array to JSON
    report.pop("_preprocessed_image", None)

    return report


def create_feature_importance_map(image_path: str, model_path: str = "models/deepfake_detector.keras") -> str:
    """
    Create a feature importance map by occluding different parts of the image.
    This is a model-agnostic approach that always works.

    Args:
        image_path: Path to the input image
        model_path: Path to trained model

    Returns:
        str: Path to saved heatmap image
    """
    try:
        if os.path.exists(model_path):
            model = load_model()
        else:
            print("[WARNING] No trained model found for feature importance analysis")
            return None

        img = Image.open(image_path).convert("RGB").resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0

        original_pred = model.predict(np.expand_dims(img_array, axis=0), verbose=0)
        original_score = float(original_pred[0][1] if original_pred.shape[1] > 1 else original_pred[0][0])

        patch_size = 32
        importance_map = np.zeros((224, 224))

        for i in range(0, 224, patch_size):
            for j in range(0, 224, patch_size):
                occluded_img = img_array.copy()
                occluded_img[i:i+patch_size, j:j+patch_size] = 0
                occluded_pred = model.predict(np.expand_dims(occluded_img, axis=0), verbose=0)
                occluded_score = float(occluded_pred[0][1] if occluded_pred.shape[1] > 1 else occluded_pred[0][0])
                importance_map[i:i+patch_size, j:j+patch_size] = abs(original_score - occluded_score)

        return _create_occlusion_visualization(img_array, importance_map, image_path, original_score)

    except Exception as e:
        print(f"[ERROR] Feature importance analysis failed: {str(e)}")
        return None


def _create_occlusion_visualization(img_array, importance_map, image_path, fake_score):
    """Create visualization from occlusion importance map."""
    try:
        importance_normalized = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-8)

        prediction_label = "FAKE" if fake_score >= 0.5 else "REAL"
        confidence = fake_score if fake_score >= 0.5 else 1.0 - fake_score
        label_color = "red" if prediction_label == "FAKE" else "green"

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 4, 1)
        plt.imshow(img_array)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.text(0.5, 0.75, f"Deepfake Score: {fake_score:.3f}",
                ha='center', va='center', fontsize=13, fontweight='bold')
        plt.text(0.5, 0.55, f"Prediction: {prediction_label}",
                ha='center', va='center', fontsize=14, fontweight='bold',
                color=label_color)
        plt.text(0.5, 0.35, f"Confidence: {confidence * 100:.1f}%",
                ha='center', va='center', fontsize=12)
        plt.text(0.5, 0.15, "Method: Occlusion",
                ha='center', va='center', fontsize=11, color='gray')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(importance_map, cmap='hot')
        plt.title("Feature Importance\n(Yellow = Critical Regions)")
        plt.axis('off')

        plt.subplot(1, 4, 4)
        overlay = img_array.copy()
        important = importance_normalized > 0.6
        overlay[important] = [1, 0.2, 0.2]
        plt.imshow(overlay)
        plt.title("Critical Regions\n(Red = Most Important)")
        plt.axis('off')

        os.makedirs("outputs/heatmaps", exist_ok=True)
        heatmap_path = f"outputs/heatmaps/occlusion_{os.path.basename(image_path)}"
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[OCCLUSION] Heatmap saved to: {heatmap_path}")
        return heatmap_path

    except Exception as e:
        print(f"[ERROR] Occlusion visualization failed: {str(e)}")
        return None
