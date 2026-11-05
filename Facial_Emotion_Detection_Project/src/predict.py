from pathlib import Path
import argparse, json
import numpy as np
from tensorflow import keras

import tensorflow as tf
import numpy as np
import json
import cv2

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "emotion_cnn.keras"
CLASSES = ROOT / "models" / "class_names.json"
IMG_SIZE = (128, 128)

def main():
    parser = argparse.ArgumentParser(description="Predict a visible facial-expression class.")
    parser.add_argument("--image", required=True, help="Path to an image")
    args = parser.parse_args()
    if not MODEL.exists() or not CLASSES.exists():
        raise FileNotFoundError("Train first using: python src/train.py")
    img = keras.utils.load_img(args.image, target_size=IMG_SIZE, color_mode="rgb")
    arr = keras.utils.img_to_array(img) / 255.0
    model = keras.models.load_model(MODEL)
    probs = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
    names = json.loads(CLASSES.read_text(encoding="utf-8"))
    order = np.argsort(probs)[::-1]
    print("Predicted visible expression:", names[int(order[0])])
    print("Model score (not a probability of someone's true feelings):", f"{probs[order[0]]:.1%}")
    for i in order:
        print(f"{names[int(i)]}: {probs[i]:.1%}")

if __name__ == "__main__":
    main()



# Load trained model
model = tf.keras.models.load_model(
    "models/emotion_cnn.keras"
)

# Load class names
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# Upload image path
image_path = input("Enter image path: ")

# Read image
img = cv2.imread(image_path)

if img is None:
    print("Image not found!")
    exit()

# Preprocess image
img = cv2.resize(img, (128, 128))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

# Prediction
prediction = model.predict(img)

index = np.argmax(prediction[0])
emotion = class_names[index]
confidence = prediction[0][index] * 100

print("Predicted Emotion:", emotion)
print("Confidence:", round(float(confidence), 2), "%")