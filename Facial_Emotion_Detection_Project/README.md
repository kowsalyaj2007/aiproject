# Facial Emotion Detection — CNN Project (VS Code)

This project includes the 40 face images from the ZIP you uploaded, a labelling template, a CNN training pipeline, evaluation, and single-image prediction.

## Important: model is NOT trained yet
The 40 supplied images are one image per subject and do not contain verified emotion labels. A supervised emotion classifier cannot be trained honestly until each image is labelled with its visible facial-expression category. The labels must not be guessed from subject IDs.

Also, 40 images total is far too small for a dependable multi-class emotion model. This package is a complete project starter and training pipeline, not a pre-trained model. For a stronger project, add a larger, correctly labelled dataset.

## Folder structure

```text
Facial_Emotion_Detection_Project/
├── dataset/
│   ├── raw_images/          # 40 supplied face images
│   └── labels.csv           # fill in the emotion column
├── src/
│   ├── train.py             # CNN training
│   ├── evaluate.py          # test-set metrics + confusion matrix
│   └── predict.py           # predict one image
├── models/                  # trained model saved here
├── reports/                 # evaluation results saved here
├── requirements.txt
└── README.md
```

## Step 1 — Open in VS Code and install

Use Python 3.10 or 3.11.

Windows PowerShell:
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Step 2 — Label the images

Open `dataset/labels.csv` in Excel or VS Code. For every row, fill the `emotion` column with the correct visible expression label. Suggested labels:
- angry
- disgust
- fear
- happy
- sad
- surprise
- neutral

Use only labels that you can confidently verify. If an expression is unclear, leave it blank; blank rows are ignored. Do not infer a person's internal emotional state—these are only visible-expression labels.

For a proper train/validation/test split, aim for many images in every class and different people across splits. With only 40 images, the test metrics will be highly unreliable.

## Step 3 — Add more labelled data (strongly recommended)

Add more face images to `dataset/raw_images/` and add matching rows to `dataset/labels.csv`. Use images you have permission to use, and make sure labels are accurate. Avoid duplicate or near-duplicate images across splits.

## Step 4 — Train

```powershell
python src/train.py
```

The script checks the number of labelled examples and class counts. It will stop if there are too few examples to make a minimally useful split. It saves `models/emotion_cnn.keras` and `models/class_names.json`.

## Step 5 — Evaluate

```powershell
python src/evaluate.py
```

This creates:
- `reports/test_metrics.json`
- `reports/confusion_matrix.png`

## Step 6 — Predict an image

```powershell
python src/predict.py --image "dataset/raw_images/subject_00.png"
```

## Responsible-use note

The model estimates a dataset's visible facial-expression category. It cannot determine a person's true feelings, intentions, personality, or mental state. Do not use it for hiring, grading, policing, medical diagnosis, or other high-impact decisions.
