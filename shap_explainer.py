"""
shap_explainer.py
-----------------
SHAP explainability for deepfake detection model.
Generates heatmaps showing which pixels triggered fake detection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shap
import tensorflow as tf
from model import load_model, build_model


def create_shap_heatmap(image_path: str, model_path: str = "models/deepfake_detector.keras") -> str:
    """
    Generate SHAP heatmap for deepfake model prediction.
    
    Args:
        image_path: Path to the input image
        model_path: Path to trained model
        
    Returns:
        str: Path to saved heatmap image
    """
    try:
        # Load model
        if os.path.exists(model_path):
            model = load_model()
        else:
            print("[WARNING] No trained model found for SHAP analysis")
            return None
            
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Try SHAP with different explainers
        try:
            # Create background dataset (use a few random images as baseline)
            background = np.random.rand(5, 224, 224, 3) * 0.5  # Simple background
            
            # Try DeepExplainer first
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(np.expand_dims(img_array, axis=0))
            
            # For a single-sigmoid binary model (shape: batch x 1),
            # shap_values is a list with one element: shap_values[0]
            # (NOT shap_values[1] — there is no index 1 for a 1-output model)
            if isinstance(shap_values, list):
                fake_class_shap = shap_values[0][0]  # FIX: index 0, not 1
            else:
                fake_class_shap = shap_values[0]
                
            return _create_heatmap_visualization(img_array, fake_class_shap, image_path, "SHAP")
            
        except Exception as shap_error:
            print(f"[SHAP] DeepExplainer failed: {str(shap_error)}")
            
            # Fallback: Create a simple gradient-based heatmap
            return _create_gradient_heatmap(model, img_array, image_path)
        
    except Exception as e:
        print(f"[ERROR] SHAP analysis failed: {str(e)}")
        return None


def _create_heatmap_visualization(img_array, shap_values, image_path, method_name):
    """Create visualization from SHAP values."""
    try:
        # Create heatmap visualization
        plt.figure(figsize=(12, 4))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(img_array)
        plt.title("Original Image")
        plt.axis('off')
        
        # SHAP values (positive = contributes to "fake" prediction)
        plt.subplot(1, 3, 2)
        shap_image = shap_values.mean(axis=-1)  # Average across RGB channels
        plt.imshow(shap_image, cmap='RdBu_r', vmin=-np.abs(shap_image).max(), vmax=np.abs(shap_image).max())
        plt.title(f"{method_name} Values\n(Red = Fake, Blue = Real)")
        plt.axis('off')
        
        # Overlay on original image
        plt.subplot(1, 3, 3)
        # Normalize SHAP values for overlay
        shap_normalized = (shap_image - shap_image.min()) / (shap_image.max() - shap_image.min())
        # Create red channel for fake areas
        overlay = img_array.copy()
        fake_mask = shap_normalized > 0.5  # Threshold for significant fake indicators
        overlay[fake_mask] = [1, 0, 0]  # Red overlay for fake regions
        
        plt.imshow(overlay)
        plt.title("Fake Regions Highlighted")
        plt.axis('off')
        
        # Save the heatmap
        os.makedirs("outputs/heatmaps", exist_ok=True)
        heatmap_path = f"outputs/heatmaps/{method_name.lower()}_{os.path.basename(image_path)}"
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[{method_name}] Heatmap saved to: {heatmap_path}")
        return heatmap_path
        
    except Exception as e:
        print(f"[ERROR] {method_name} visualization failed: {str(e)}")
        return None


def _create_gradient_heatmap(model, img_array, image_path):
    """Create a simple gradient-based heatmap as fallback."""
    try:
        import tensorflow as tf
        
        # Convert to tensor
        img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0))
        
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            prediction = model(img_tensor)
            fake_score = prediction[0][1] if prediction.shape[1] > 1 else prediction[0][0]
        
        # Compute gradients
        gradients = tape.gradient(fake_score, img_tensor)
        if gradients is not None:
            # Convert to numpy and process
            grad_array = gradients[0].numpy()
            grad_magnitude = np.abs(grad_array).mean(axis=-1)
            
            return _create_heatmap_visualization(img_array, grad_array, image_path, "Gradient")
        else:
            print("[SHAP] Gradient computation failed")
            return None
            
    except Exception as e:
        print(f"[ERROR] Gradient heatmap failed: {str(e)}")
        return None


def analyze_with_shap(image_path: str, report: dict) -> dict:
    """
    Run SHAP analysis and update report with heatmap path.
    
    Args:
        image_path: Path to input image
        report: Report dictionary to update
        
    Returns:
        dict: Updated report dictionary
    """
    print("[SHAP] Generating explainability heatmap...")
    
    heatmap_path = create_shap_heatmap(image_path)
    
    if heatmap_path:
        report["shap_heatmap_path"] = heatmap_path
        print(f"[SHAP] Explainability analysis complete")
    else:
        report["shap_heatmap_path"] = None
        print("[SHAP] Explainability analysis failed")
        
    return report
