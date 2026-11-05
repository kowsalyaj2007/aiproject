from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "emotion_cnn.keras"
CLASSES = ROOT / "models" / "class_names.json"
TEST = ROOT / "reports" / "test_split.npz"
REPORTS = ROOT / "reports"

def main():
    for p in (MODEL, CLASSES, TEST):
        if not p.exists():
            raise FileNotFoundError(f"{p.relative_to(ROOT)} not found. Run python src/train.py first.")
    model = keras.models.load_model(MODEL)
    class_names = json.loads(CLASSES.read_text(encoding="utf-8"))
    data = np.load(TEST)
    X_test, y_test = data["X_test"], data["y_test"]
    probs = model.predict(X_test, verbose=0)
    y_pred = probs.argmax(axis=1)
    report = classification_report(y_test, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45, values_format="d")
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix.png", dpi=160)
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

if __name__ == "__main__":
    main()
