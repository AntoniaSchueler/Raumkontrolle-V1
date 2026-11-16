from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf


class TeachableMachineClassifier:
    """Lädt ein mit Teachable Machine exportiertes Keras-Bildklassifikationsmodell."""

    def __init__(self, model_path: Path, labels_path: Path):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Modell nicht gefunden: {self.model_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels nicht gefunden: {self.labels_path}")

        self.labels = self._load_labels()
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

    def _load_labels(self):
        labels = []
        with self.labels_path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels.append(parts[1].strip())
                else:
                    labels.append(line)
        if not labels:
            raise ValueError(f"Keine Labels in {self.labels_path} gefunden.")
        return labels

    def predict(self, image: Image.Image):
        image = image.convert("RGB").resize((224, 224))
        array = np.asarray(image, dtype=np.float32)
        # Standard-Vorverarbeitung der Teachable-Machine-Keras-Exporte.
        array = (array / 127.5) - 1.0
        batch = np.expand_dims(array, axis=0)

        prediction = self.model.predict(batch, verbose=0)[0]
        if len(prediction) != len(self.labels):
            raise ValueError(
                f"Das Modell liefert {len(prediction)} Werte, "
                f"aber die Labeldatei enthält {len(self.labels)} Klassen."
            )

        probabilities = {
            label: float(prediction[index])
            for index, label in enumerate(self.labels)
        }
        best_index = int(np.argmax(prediction))
        return {
            "label": self.labels[best_index],
            "confidence": float(prediction[best_index]),
            "probabilities": probabilities,
        }
