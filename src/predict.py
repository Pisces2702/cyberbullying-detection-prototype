"""Predict whether a supplied social-media text is cyberbullying."""
from pathlib import Path
import argparse, re, joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'results' / 'model'

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[@#]', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict(text: str):
    vectorizer = joblib.load(MODEL_DIR / 'tfidf_vectorizer.joblib')
    model = joblib.load(MODEL_DIR / 'svm_model.joblib')
    cleaned = clean_text(text)
    pred = int(model.predict(vectorizer.transform([cleaned]))[0])
    return ('Cyberbullying' if pred == 1 else 'Non-Cyberbullying', cleaned)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('text', nargs='+')
    args = parser.parse_args()
    label, cleaned = predict(' '.join(args.text))
    print(f'Prediction: {label}')
    print(f'Cleaned text: {cleaned}')
