from pathlib import Path
import os
import sqlite3

from flask import Flask, g, render_template, request
import joblib

from train_model import MODEL_PATH, VECTORIZER_PATH, clean_text, train_model


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = Path(os.environ.get("DATABASE_PATH", BASE_DIR / "fake_news_history.db"))

app = Flask(__name__)


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = get_db()
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    db.commit()


def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        train_model()
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_news(news_text):
    model, vectorizer = load_artifacts()
    features = vectorizer.transform([clean_text(news_text)])
    prediction = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(features).max() * 100)
    else:
        confidence = 0.0

    return prediction, round(confidence, 2)


@app.before_request
def before_request():
    init_db()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news_text", "").strip()
    if not news_text:
        return render_template(
            "index.html",
            error="Please enter news content before checking.",
        )

    prediction, confidence = predict_news(news_text)

    db = get_db()
    db.execute(
        "INSERT INTO predictions (news_text, prediction, confidence) VALUES (?, ?, ?)",
        (news_text, prediction, confidence),
    )
    db.commit()

    return render_template(
        "result.html",
        news_text=news_text,
        prediction=prediction,
        confidence=confidence,
    )


@app.route("/history", methods=["GET"])
def history():
    rows = get_db().execute(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 20"
    ).fetchall()
    return render_template("history.html", rows=rows)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
