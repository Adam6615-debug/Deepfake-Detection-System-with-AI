"""
model.py
--------
Deepfake detection model using transfer learning with MobileNetV2.

Architecture:
  MobileNetV2 (frozen) → GlobalAveragePooling → Dense(256) → Dropout
  → Dense(128) → Dropout → BatchNorm → Dense(1, sigmoid)

Output: probability score in [0, 1]
  - Close to 0 → likely REAL
  - Close to 1 → likely FAKE
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


IMAGE_SIZE  = (224, 224)
INPUT_SHAPE = (224, 224, 3)
MODEL_PATH  = os.path.join("models", "deepfake_detector.keras")


# ─── Model Definition ─────────────────────────────────────────────────────────

def build_model(input_shape=INPUT_SHAPE, learning_rate=1e-4) -> tf.keras.Model:
    """
    Builds the deepfake detection model using MobileNetV2 as a feature extractor.

    Args:
        input_shape: (H, W, C) — default (224, 224, 3)
        learning_rate: Adam optimizer learning rate

    Returns:
        Compiled Keras model
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="deepfake_detector")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(X: np.ndarray, y: np.ndarray, epochs: int = 15, batch_size: int = 16):
    """
    Trains the model on preprocessed images.

    Args:
        X: numpy array (N, 224, 224, 3), values in [0, 1]
        y: numpy array (N,), labels (0=real, 1=fake)
        epochs: number of training epochs
        batch_size: images per training step

    Returns:
        model: trained Keras model
        history: training history
    """
    if len(X) < 10:
        print("[WARNING] Very small dataset. Results will not be meaningful.")
        print("          Add more images to data/real/ and data/fake/ for real results.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    print(f"\n[INFO] Training set  : {len(X_train)} images")
    print(f"[INFO] Validation set: {len(X_val)} images")

    # FIX: compute class weights so the model can't cheat by always
    # predicting the majority class (REAL) to minimise loss.
    n_total = len(y_train)
    n_real  = int(np.sum(y_train == 0))
    n_fake  = int(np.sum(y_train == 1))
    weight_real = n_total / (2.0 * n_real) if n_real > 0 else 1.0
    weight_fake = n_total / (2.0 * n_fake) if n_fake > 0 else 1.0
    class_weight = {0: weight_real, 1: weight_fake}
    print(f"\n[INFO] Class weights → real: {weight_real:.3f}, fake: {weight_fake:.3f}")

    model = build_model(learning_rate=1e-5)
    model.summary()

    os.makedirs("models", exist_ok=True)
    callback_list = [
        callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_auc",          
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_auc",       
            patience=5,
            mode="max",
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,      
        callbacks=callback_list,
        verbose=1
    )

    print(f"\n[INFO] Best model saved to: {MODEL_PATH}")
    return model, history


# ─── Inference ────────────────────────────────────────────────────────────────

def load_model() -> tf.keras.Model:
    """Loads the saved model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No saved model found at {MODEL_PATH}. "
            "Train the model first: python main.py --mode train"
        )
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


def predict_single(model: tf.keras.Model, image: np.ndarray) -> dict:
    """
    Predicts whether a single preprocessed image is real or fake.

    Args:
        model: trained Keras model
        image: numpy array (224, 224, 3), values in [0, 1]

    Returns:
        dict with keys: deepfake_score, label, confidence
    """
    img_batch  = np.expand_dims(image, axis=0)
    score      = float(model.predict(img_batch, verbose=0)[0][0])
    label      = "FAKE" if score >= 0.5 else "REAL"
    confidence = score if score >= 0.5 else 1 - score
    return {
        "deepfake_score": round(score, 4),
        "label":          label,
        "confidence":     round(confidence * 100, 1)
    }


def predict_batch(model: tf.keras.Model, images: np.ndarray) -> list:
    """
    Runs prediction on a batch of preprocessed images.

    Args:
        model: trained Keras model
        images: numpy array (N, 224, 224, 3), values in [0, 1]

    Returns:
        List of prediction dicts
    """
    scores  = model.predict(images, verbose=0).flatten()
    results = []
    for score in scores:
        score      = float(score)
        label      = "FAKE" if score >= 0.5 else "REAL"
        confidence = score if score >= 0.5 else 1 - score
        results.append({
            "deepfake_score": round(score, 4),
            "label":          label,
            "confidence":     round(confidence * 100, 1)
        })
    return results


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_training_history(history, save_path: str = "outputs/training_history.png"):
    """
    Plots accuracy, loss, and AUC curves from training history.

    Args:
        history: Keras History object returned by model.fit()
        save_path: where to save the plot image
    """
    os.makedirs("outputs", exist_ok=True)

    has_auc = "auc" in history.history

    ncols = 3 if has_auc else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    # AUC (FIX: added so you can spot a collapsed model immediately)
    if has_auc:
        axes[2].plot(history.history["auc"],     label="Train")
        axes[2].plot(history.history["val_auc"], label="Val")
        axes[2].set_title("AUC")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Training plot saved to: {save_path}")


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[TEST] Building model with dummy data...")
    model = build_model()
    model.summary()

    dummy_img = np.random.rand(224, 224, 3).astype(np.float32)
    result    = predict_single(model, dummy_img)
    print(f"\n[TEST] Prediction on random image: {result}")
    print("[DONE] model.py is working correctly.")
