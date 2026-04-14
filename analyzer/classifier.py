"""
AI/ML Classifier for Cat Urine Detection

Implements a TensorFlow/Keras convolutional neural network that distinguishes
between three classes of UV-fluorescent regions:

  0 - no_urine          : No urine present
  1 - urine             : Cat urine detected
  2 - other_fluorescent : False positive (detergent, brightener, fabric, etc.)

The classifier works on cropped bounding-box patches produced by the
UVFluorescenceDetector, reducing the false-positive rate caused by
household cleaning products and laundry detergents that also fluoresce
under UV light.

Training guidance:
- Collect ~500+ labelled patch images per class.
- Use transfer learning from MobileNetV2 (pre-trained on ImageNet) for
  best accuracy with limited training data.
- Augment with random brightness/contrast changes to simulate different
  UV intensities and surface materials.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

# TensorFlow import is deferred to allow the module to load even when TF is
# not installed (useful for testing with mock classifiers).
try:
    import tensorflow as tf
    from tensorflow import keras

    _TF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TF_AVAILABLE = False

CLASS_NAMES = ["no_urine", "urine", "other_fluorescent"]
DEFAULT_INPUT_SIZE = 224


def build_model(
    num_classes: int = 3,
    input_size: int = DEFAULT_INPUT_SIZE,
    learning_rate: float = 1e-4,
) -> "keras.Model":
    """
    Build a transfer-learning classifier based on MobileNetV2.

    The MobileNetV2 base is frozen; only the custom classification head
    is trained initially, which converges faster with small datasets.

    Args:
        num_classes: Number of output classes.
        input_size: Square input image size (pixels).
        learning_rate: Adam optimiser learning rate.

    Returns:
        Compiled Keras model ready for training.
    """
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required to build the model.")

    base = keras.applications.MobileNetV2(
        input_shape=(input_size, input_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = keras.Input(shape=(input_size, input_size, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class UrineClassifier:
    """
    Wraps the Keras model for inference on UV-fluorescence image patches.

    The classifier receives cropped region-of-interest patches from the
    UVFluorescenceDetector and returns a class label plus confidence score.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: int = DEFAULT_INPUT_SIZE,
        confidence_threshold: float = 0.7,
        num_classes: int = 3,
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model_path: Path to a saved Keras model directory.  If None or the
                path does not exist, a fresh untrained model is created (useful
                for development / testing).
            input_size: Input image size expected by the model.
            confidence_threshold: Minimum probability to report a positive.
            num_classes: Number of output classes.
            class_names: Human-readable class names.
        """
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names or CLASS_NAMES
        self._model: Optional["keras.Model"] = None

        if model_path and os.path.exists(model_path):
            self._model = self._load_model(model_path)
        else:
            if _TF_AVAILABLE:
                self._model = build_model(
                    num_classes=num_classes,
                    input_size=input_size,
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, patch: np.ndarray) -> Tuple[str, float]:
        """
        Classify a single image patch.

        Args:
            patch: BGR image patch (any size; will be resized internally).

        Returns:
            Tuple of (class_name, confidence) where confidence is 0.0–1.0.
        """
        if self._model is None:
            raise RuntimeError(
                "No model available. Install TensorFlow or load a saved model."
            )
        preprocessed = self._preprocess(patch)
        probabilities = self._model.predict(preprocessed, verbose=0)[0]
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
        class_name = self.class_names[class_index]
        return class_name, confidence

    def predict_batch(
        self, patches: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        """
        Classify multiple patches in a single forward pass for efficiency.

        Args:
            patches: List of BGR image patches.

        Returns:
            List of (class_name, confidence) tuples.
        """
        if not patches:
            return []
        if self._model is None:
            raise RuntimeError(
                "No model available. Install TensorFlow or load a saved model."
            )
        batch = np.concatenate(
            [self._preprocess(p) for p in patches], axis=0
        )
        probabilities = self._model.predict(batch, verbose=0)
        results = []
        for probs in probabilities:
            idx = int(np.argmax(probs))
            results.append((self.class_names[idx], float(probs[idx])))
        return results

    def save(self, model_path: str) -> None:
        """Save the model to disk in SavedModel format."""
        if self._model is None:
            raise RuntimeError("No model to save.")
        os.makedirs(model_path, exist_ok=True)
        self._model.save(model_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(self, patch: np.ndarray) -> np.ndarray:
        """Resize, convert BGR→RGB, and add batch dimension."""
        import cv2

        resized = cv2.resize(patch, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.expand_dims(rgb.astype(np.float32), axis=0)

    @staticmethod
    def _load_model(model_path: str) -> "keras.Model":
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow is required to load the model.")
        return keras.models.load_model(model_path)
