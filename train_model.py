from pathlib import Path
import re
import string

import joblib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "sample_news.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def train_model():
    MODEL_DIR.mkdir(exist_ok=True)

    data = pd.read_csv(DATASET_PATH)
    data = data.dropna(subset=["text", "label"])
    data["clean_text"] = data["text"].apply(clean_text)

    x_train, x_test, y_train, y_test = train_test_split(
        data["clean_text"],
        data["label"],
        test_size=0.25,
        random_state=42,
        stratify=data["label"],
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    x_train_vectors = vectorizer.fit_transform(x_train)
    x_test_vectors = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_vectors, y_train)

    predictions = model.predict(x_test_vectors)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"Model trained successfully. Accuracy: {accuracy:.2%}")
    print(classification_report(y_test, predictions, zero_division=0))

    return model, vectorizer


if __name__ == "__main__":
    train_model()
