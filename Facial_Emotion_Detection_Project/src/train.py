from pathlib import Path
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "dataset" / "labels.csv"
IMAGE_ROOT = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 30
SEED = 42

def load_labelled_rows():
    if not CSV_PATH.exists():
        raise FileNotFoundError("dataset/labels.csv is missing.")
    df = pd.read_csv(CSV_PATH)
    if not {"image", "emotion"}.issubset(df.columns):
        raise ValueError("CSV must contain image and emotion columns.")
    df = df.dropna(subset=["emotion"]).copy()
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    df = df[df["emotion"] != ""]
    df["path"] = df["image"].apply(lambda x: str((IMAGE_ROOT / x).resolve()))
    missing = [p for p in df["path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Image file not found: {missing[0]}")
    if len(df) < 30:
        raise ValueError(
            f"Only {len(df)} labelled images. Add more correctly labelled images. "
            "At least 30 are required by this demo; substantially more are recommended."
        )
    counts = df["emotion"].value_counts()
    valid_classes = counts[counts >= 6].index
    df = df[df["emotion"].isin(valid_classes)].copy()
    if df["emotion"].nunique() < 2:
        raise ValueError("Need at least 2 emotion classes, with 6 or more images per class.")
    print("Labelled images per class:\n", df["emotion"].value_counts())
    if len(df) < 60:
        print("WARNING: Small dataset. Accuracy and generalization will be unreliable.")
    return df

def load_image(path):
    img = keras.utils.load_img(path, target_size=IMG_SIZE, color_mode="rgb")
    return keras.utils.img_to_array(img) / 255.0

def build_model(num_classes):
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    df = load_labelled_rows()
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["emotion"])
    paths = df["path"].to_numpy()
    indices = np.arange(len(df))
    try:
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.30, random_state=SEED, stratify=y
        )
        temp_y = y[temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=SEED, stratify=temp_y
        )
    except ValueError as e:
        raise ValueError(
            "Cannot make stratified train/validation/test split. Add more images "
            "for every emotion class (at least 6 per class is a starting point)."
        ) from e

    X_train = np.stack([load_image(paths[i]) for i in train_idx])
    X_val = np.stack([load_image(paths[i]) for i in val_idx])
    X_test = np.stack([load_image(paths[i]) for i in test_idx])
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    model = build_model(len(encoder.classes_))
    model.summary()
    MODEL_DIR.mkdir(exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_DIR / "emotion_cnn.keras",
                                        monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              class_weight=class_weights, callbacks=callbacks)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(encoder.classes_.tolist(), f, indent=2)
    np.savez(ROOT / "reports" / "test_split.npz", X_test=X_test, y_test=y_test)
    with open(ROOT / "reports" / "split_info.json", "w", encoding="utf-8") as f:
        json.dump({"classes": encoder.classes_.tolist(),
                   "test_loss": float(test_loss), "test_accuracy": float(test_acc)}, f, indent=2)
    print(f"Test accuracy: {test_acc:.4f}")
    print("Model saved to models/emotion_cnn.keras")
    print("Note: with a small dataset, this score is not a reliable estimate of real-world performance.")

if __name__ == "__main__":
    main()
