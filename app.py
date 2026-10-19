from flask import Flask, render_template, request
import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------- GLOBALS ----------------
dataset_preview = None
preprocessing_result = ""
split_result = ""
model_result = ""
prediction_result = []
accuracy_result = 0
report_result = ""
pie_chart = {}
accuracy_chart = {}
confidence_preview = []
trained_model = None


# ---------------- HOME ----------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    global trained_model

    news_text = request.form.get('news', '').strip()

    if not news_text:
        return "Please enter news text."

    if trained_model is None:
        return "Please upload dataset first."

    prediction = trained_model.predict([news_text])[0]

    if prediction == 1:
        result = "Real News"
    else:
        result = "Fake News"

    return f"Result: {result}"


# ---------------- UPLOAD + TRAIN ----------------
@app.route('/upload', methods=['POST'])
def upload():
    global dataset_preview, preprocessing_result, split_result
    global model_result, prediction_result, accuracy_result
    global report_result, pie_chart, accuracy_chart
    global confidence_preview, trained_model

    file = request.files.get('file')

    if file is None or file.filename == '':
        return "No file uploaded"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    df = pd.read_csv(filepath)

    # DISPLAY ALL ROWS IN A SCROLLABLE TABLE.
    # This only changes the preview; training logic below remains the same.
    dataset_preview = (
        '<div style="max-height: 450px; overflow-y: auto; overflow-x: auto; '
        'border: 1px solid #ddd;">'
        + df.to_html(
            classes='table table-bordered',
            index=False,
            justify='center',
            border=0
        )
        + '</div>'
    )

    # ================================
    # 1. AUTO DETECT TEXT COLUMN
    # ================================
    text_cols = ['text', 'title', 'news', 'content', 'article']
    label_cols = ['label', 'class', 'target', 'real', 'fake']

    text_col = None
    label_col = None

    for c in text_cols:
        if c in df.columns:
            text_col = c
            break

    for c in label_cols:
        if c in df.columns:
            label_col = c
            break

    if text_col is None or label_col is None:
        return "Dataset must contain text + label column"

    df = df[[text_col, label_col]].dropna()
    df.columns = ['text', 'label']

    # ================================
    # 2. CLEAN LABELS (VERY IMPORTANT)
    # ================================
    df['label'] = df['label'].astype(str).str.lower()

    df['label'] = df['label'].replace({
        'real': 1,
        'true': 1,
        'fake': 0,
        'false': 0,
        'yes': 1,
        'no': 0
    })

    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna()

    preprocessing_result = f"""
Dataset Shape: {df.shape}<br>
Text Column: {text_col}<br>
Label Column: {label_col}
"""

    # ================================
    # 3. SPLIT
    # ================================
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    split_result = f"""
Train Samples: {len(X_train)}<br>
Test Samples: {len(X_test)}
"""

    # ================================
    # 4. MODEL PIPELINE
    # ================================
    clf = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2)
        )),
        ('model', LogisticRegression(max_iter=1000))
    ])

    clf.fit(X_train, y_train)
    trained_model = clf

    model_result = "Model Trained Successfully (Universal Mode)"

    # ================================
    # 5. PREDICTION
    # ================================
    predictions = clf.predict(X_test)

    prediction_result = [
        "Real News" if p == 1 else "Fake News"
        for p in predictions[:10]
    ]

    # ================================
    # 6. ACCURACY
    # ================================
    accuracy_result = accuracy_score(y_test, predictions)

    report_result = classification_report(y_test, predictions)

    # ================================
    # 7. PIE CHART (REAL vs FAKE)
    # ================================
    real_count = int((predictions == 1).sum())
    fake_count = int((predictions == 0).sum())

    pie_chart = {
        "data": [
            {"type": "Real News", "count": real_count},
            {"type": "Fake News", "count": fake_count}
        ]
    }

    # ================================
    # 8. ACCURACY CHART
    # ================================
    accuracy_chart = {
        "data": [
            {"value": round(accuracy_result * 100, 2)}
        ]
    }

    # ================================
    # 9. CONFIDENCE
    # ================================
    try:
        proba = clf.predict_proba(X_test)

        confidence_preview = [
            {
                "real_confidence": round(p[1] * 100, 2),
                "fake_confidence": round(p[0] * 100, 2)
            }
            for p in proba[:10]
        ]
    except Exception:
        confidence_preview = []

    return render_template(
        'model.html',
        preview=dataset_preview,
        preprocessing=preprocessing_result,
        split=split_result,
        model=model_result,
        pie_chart=pie_chart,
        accuracy_chart=accuracy_chart,
        confidence=confidence_preview,
        report=report_result
    )


# ---------------- RESULT PAGE ----------------
@app.route('/result')
def result():
    return render_template(
        'result.html',
        predictions=prediction_result,
        accuracy=accuracy_result,
        report=report_result
    )


# ---------------- RUN ----------------
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
