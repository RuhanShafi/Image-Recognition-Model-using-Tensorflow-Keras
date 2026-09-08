import numpy as np
import tensorflow as tf

CLASS_NAMES = ['Boys', 'Girls', 'Man', 'Woman'] # label-index complient with the model's output layer
IMG_HEIGHT = 180
IMG_WIDTH = 180


class Predictor:
    """Loads the TFLite model once and exposes a fast predict() call,
    suitable for repeated calls in a live webcam loop."""

    def __init__(self, model_path, class_names=None):
        self.class_names = class_names or CLASS_NAMES
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
        except Exception as e:
            raise RuntimeError(f"Failed to load TFLite model from '{model_path}': {e}") from e

        self._input_detail = self.interpreter.get_input_details()[0]
        self._output_detail = self.interpreter.get_output_details()[0]

    def predict(self, image: np.ndarray):
        """
        image: RGB uint8 numpy array, any size, shape (H, W, 3).
        Returns (label: str, confidence: float 0-1).
        """
        batch = self._preprocess(image)

        self.interpreter.set_tensor(self._input_detail["index"], batch)
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(self._output_detail["index"])[0]

        probs = tf.nn.softmax(logits).numpy()
        idx = int(np.argmax(probs))

        return self.class_names[idx], float(probs[idx])

    def predict_all(self, image: np.ndarray):
        batch = self._preprocess(image)
        self.interpreter.set_tensor(self._input_detail["index"], batch)
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(self._output_detail["index"])[0]
        probs = tf.nn.softmax(logits).numpy()
        return {name: float(p) for name, p in zip(self.class_names, probs)}

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an RGB image (H, W, 3), got shape {image.shape}")

        resized = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        batch = tf.expand_dims(resized, axis=0)

        # Match the TFLite input tensor's expected dtype (float32 unless you quantized).
        batch = tf.cast(batch, self._input_detail["dtype"]).numpy()
        return batch