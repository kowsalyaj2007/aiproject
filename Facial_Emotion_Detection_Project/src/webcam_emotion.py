import cv2
import tensorflow as tf
import numpy as np
import json

# Load trained CNN model
model = tf.keras.models.load_model(
    "models/emotion_cnn.keras"
)

# Load class names
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# Support list or dictionary class names
if isinstance(class_names, dict):
    class_names = [
        name for name, index in
        sorted(class_names.items(), key=lambda x: x[1])
    ]

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not detected!")
    exit()

print("CNN Emotion Detection Started!")
print("Press Q to exit.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for (x, y, w, h) in faces:

        # Extract face
        face = frame[y:y+h, x:x+w]

        # Preprocessing
        face = cv2.cvtColor(
            face, cv2.COLOR_BGR2RGB
        )

        face = cv2.resize(
            face, (128, 128)
        )

        face = face.astype("float32") / 255.0

        face = np.expand_dims(
            face, axis=0
        )

        # CNN prediction
        prediction = model.predict(
            face, verbose=0
        )

        index = np.argmax(prediction[0])

        emotion = class_names[index]

        confidence = float(
            prediction[0][index] * 100
        )

        # Display prediction
        label = f"{emotion}: {confidence:.1f}%"

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "CNN Facial Emotion Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()