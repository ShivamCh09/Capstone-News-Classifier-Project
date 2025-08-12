from flask import Flask, request, jsonify, send_from_directory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib
import re
import string
import os

app = Flask(__name__)

# In-memory cache (loaded on first request)
_vectorizer = None
_rf = None
_lr = None

# ---------- Text preprocessing ----------
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    # keep digits; remove punctuation and extra spaces
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------- Train & save artifacts (RF + LR) ----------
def train_and_save_models():
    try:
        df_true = pd.read_csv('True.csv')
        df_fake = pd.read_csv('Fake.csv')
    except FileNotFoundError:
        return {"error": "CSV files not found. Put True.csv and Fake.csv next to api.py."}

    df_true['label'] = 'true'
    df_fake['label'] = 'fake'
    df = pd.concat([df_true, df_fake], ignore_index=True)

    # robust text
    df['text'] = df.get('text', '').fillna('')
    df['processed_text'] = df['text'].apply(preprocess_text)

    # stratified split (fit tfidf on train only)
    X_train, _, y_train, _ = train_test_split(
        df['processed_text'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )

    # TF-IDF tuned for news
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.7,
        min_df=5,
        max_features=100_000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)

    # Models
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    lr = LogisticRegression(
        max_iter=2000,
        solver='liblinear',
        class_weight='balanced'
    )

    # Fit
    rf.fit(X_train_vec, y_train)
    lr.fit(X_train_vec, y_train)

    # Save artifacts
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(rf, 'random_forest_model.pkl')
    joblib.dump(lr, 'logreg_model.pkl')

    # Clear in-memory cache so next predict reloads the new ones
    global _vectorizer, _rf, _lr
    _vectorizer = _rf = _lr = None

    return {"status": "RF + LR trained", "saved": ["vectorizer.pkl", "random_forest_model.pkl", "logreg_model.pkl"]}

# ---------- Utilities ----------
def _ensure_artifacts():
    """Load artifacts into memory once."""
    global _vectorizer, _rf, _lr
    if _vectorizer is None and os.path.exists('vectorizer.pkl'):
        _vectorizer = joblib.load('vectorizer.pkl')
    if _rf is None and os.path.exists('random_forest_model.pkl'):
        _rf = joblib.load('random_forest_model.pkl')
    if _lr is None and os.path.exists('logreg_model.pkl'):
        _lr = joblib.load('logreg_model.pkl')

def _label_and_confidence_from_model(model, X):
    """Return (label, confidence) from a sklearn classifier."""
    label = model.predict(X)[0]
    conf = None
    try:
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
        p_label = float(proba[classes.index(label)])
        conf = p_label
    except Exception:
        pass
    return label, conf

# ---------- Routes ----------
@app.route('/health')
def health():
    return {"status": "ok"}

@app.route('/train', methods=['GET'])
def train():
    result = train_and_save_models()
    return jsonify(result)

@app.route('/predict', methods=['POST'])
def predict():
    _ensure_artifacts()
    if _vectorizer is None or (_rf is None and _lr is None):
        return jsonify({"error": "Models not found. Visit /train once to create them."}), 500

    data = request.get_json(silent=True) or {}
    text = " ".join([
        data.get('title', ''),
        data.get('text', ''),
        data.get('content', '')
    ]).strip()
    if not text:
        return jsonify({"error": "Provide 'text' (and optionally 'title'/'content')."}), 400

    processed = preprocess_text(text)
    X = _vectorizer.transform([processed])

    # Prefer Logistic Regression; fall back to RF
    if _lr is not None:
        label, conf = _label_and_confidence_from_model(_lr, X)
    else:
        label, conf = _label_and_confidence_from_model(_rf, X)

    # If no proba available, try the other model
    if conf is None and _rf is not None and _lr is not None:
        alt_label, alt_conf = _label_and_confidence_from_model(_rf, X)
        label, conf = (alt_label, alt_conf) if alt_conf is not None else (label, conf)

    response = {"label": label}
    if conf is not None:
        response["confidence"] = round(conf, 4)
    return jsonify(response)

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    print("To train the models, visit /train (this may take a while).")
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
