"""
gradcam_explainer.py
-------------------
Grad-CAM explainability for deepfake detection model.
Generates heatmaps showing which regions influenced the fake detection.
Works seamlessly with MobileNetV2 and modern TensorFlow.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tensorflow as tf
from model import load_model, build_model


class GradCAM:
    """Grad-CAM implementation for CNN explainability."""
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained TensorFlow model
            layer_name: Target layer for Grad-CAM (auto-detected if None)
        """
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        
    def _find_target_layer(self):
        """Find the last convolutional layer in the model."""
        # FIX: model.layers order is:
        #   [0] InputLayer
        #   [1] TFOpLambda (preprocess_input * 255)  <-- NOT the nested model
        #   [2] mobilenetv2_1.00_224 (Functional)    <-- this is the nested model
        # We scan ALL layers for the first nested Functional model, not just layers[1]
        nested_model = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model):
                nested_model = layer
                break

        target_model = nested_model if nested_model is not None else self.model

        for layer in reversed(target_model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D,
                                  tf.keras.layers.DepthwiseConv2D,
                                  tf.keras.layers.SeparableConv2D)):
                return layer.name

        raise ValueError("No convolutional layer found in model")

    def _get_target_layer(self):
        """Get the actual target layer object."""
        # FIX: scan ALL layers for nested Functional model, not just index 1
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model):
                try:
                    return layer.get_layer(self.layer_name)
                except ValueError:
                    pass
        # Fall back to searching the top-level model
        return self.model.get_layer(self.layer_name)
    
    def compute_heatmap(self, image, class_idx=None, eps=1e-8):
        """
        Compute Grad-CAM heatmap.
        
        Args:
            image: Input image tensor
            class_idx: Target class index (auto-detected if None)
            eps: Small value to prevent division by zero
            
        Returns:
            numpy array: Grad-CAM heatmap
        """
        # Get the target layer (handle nested models)
        target_layer = self._get_target_layer()
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[target_layer.output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            # Watch the target layer
            conv_outputs, predictions = grad_model(image)
            
            # Get the predicted class if not specified
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Get the loss for the target class
            loss = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Pool gradients (global average pooling)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the convolutional outputs
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU to keep only positive influences
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize the heatmap
        heatmap = heatmap - tf.reduce_min(heatmap)
        heatmap = heatmap / (tf.reduce_max(heatmap) + eps)
        
        return heatmap.numpy()


def create_gradcam_heatmap(image_path: str, model_path: str = "models/deepfake_detector.keras") -> str:
    """
    Generate Grad-CAM heatmap for deepfake model prediction.
    
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
            print("[WARNING] No trained model found for Grad-CAM analysis")
            return None
            
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_tensor = np.expand_dims(img_array, axis=0)
        
        # Get prediction
        prediction = model.predict(img_tensor, verbose=0)
        fake_score = prediction[0][1] if prediction.shape[1] > 1 else prediction[0][0]
        predicted_class = 1 if fake_score > 0.5 else 0  # 1=fake, 0=real
        
        # Create Grad-CAM
        gradcam = GradCAM(model)
        heatmap = gradcam.compute_heatmap(img_tensor, class_idx=predicted_class)
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (224, 224))
        
        # Create visualization
        return _create_gradcam_visualization(img_array, heatmap, image_path, fake_score)
        
    except Exception as e:
        print(f"[ERROR] Grad-CAM analysis failed: {str(e)}")
        return None


def _create_gradcam_visualization(img_array, heatmap, image_path, fake_score):
    """Create Grad-CAM visualization with overlay."""
    try:
        import cv2
        
        # Convert heatmap to colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Convert image to 0-255 range
        img_uint8 = np.uint8(255 * img_array)
        
        # Create overlay
        overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_colored, 0.4, 0)
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 4, 1)
        plt.imshow(img_array)
        plt.title("Original Image")
        plt.axis('off')
        
        # Prediction info
        plt.subplot(1, 4, 2)
        plt.text(0.5, 0.7, f"Fake Score: {fake_score:.3f}", 
                ha='center', va='center', fontsize=14, fontweight='bold')
        plt.text(0.5, 0.5, f"Prediction: {'FAKE' if fake_score > 0.5 else 'REAL'}", 
                ha='center', va='center', fontsize=12)
        plt.text(0.5, 0.3, f"Confidence: {max(fake_score, 1-fake_score)*100:.1f}%", 
                ha='center', va='center', fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        
        # Heatmap
        plt.subplot(1, 4, 3)
        plt.imshow(heatmap, cmap='jet')
        plt.title("Grad-CAM Heatmap\n(Red = High Influence)")
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 4, 4)
        plt.imshow(overlay / 255.0)
        plt.title("Regions of Interest\n(Overlay)")
        plt.axis('off')
        
        # Save the heatmap
        os.makedirs("outputs/heatmaps", exist_ok=True)
        heatmap_path = f"outputs/heatmaps/gradcam_{os.path.basename(image_path)}"
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Grad-CAM] Heatmap saved to: {heatmap_path}")
        return heatmap_path
        
    except Exception as e:
        print(f"[ERROR] Grad-CAM visualization failed: {str(e)}")
        return None


def analyze_with_gradcam(image_path: str, report: dict) -> dict:
    """
    Run Grad-CAM analysis and update report with heatmap path.
    
    Args:
        image_path: Path to input image
        report: Report dictionary to update
        
    Returns:
        dict: Updated report dictionary
    """
    print("[Grad-CAM] Generating explainability heatmap...")
    
    heatmap_path = create_gradcam_heatmap(image_path)
    
    if heatmap_path:
        report["shap_heatmap_path"] = heatmap_path  # Keep same key for compatibility
        report["explainability_method"] = "Grad-CAM"
        print(f"[Grad-CAM] Explainability analysis complete")
    else:
        report["shap_heatmap_path"] = None
        report["explainability_method"] = "Failed"
        print("[Grad-CAM] Explainability analysis failed")
        
    return report