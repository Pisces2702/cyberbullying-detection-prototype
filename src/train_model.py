"""Train the final TF-IDF + Linear SVM cyberbullying detector."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'Professor_Friendly_Cyberbullying_Dataset.csv'
MODEL_DIR = ROOT / 'results' / 'model'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
X = df['Cleaned_Text'].fillna('').astype(str)
y = df['Class_Label'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, sublinear_tf=True)
Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)
model = LinearSVC(random_state=42)
model.fit(Xtr, y_train)
pred = model.predict(Xte)
scores = model.decision_function(Xte)
metrics = {
    'Accuracy': accuracy_score(y_test, pred),
    'Precision': precision_score(y_test, pred),
    'Recall': recall_score(y_test, pred),
    'F1-Score': f1_score(y_test, pred),
    'ROC-AUC': roc_auc_score(y_test, scores),
}
print(classification_report(y_test, pred, target_names=['Non-Cyberbullying','Cyberbullying']))
print(metrics)
joblib.dump(model, MODEL_DIR / 'svm_model.joblib')
joblib.dump(vectorizer, MODEL_DIR / 'tfidf_vectorizer.joblib')
pd.DataFrame([metrics]).to_csv(ROOT / 'results' / 'prototype_test_metrics.csv', index=False)
print(f'Saved model artifacts to {MODEL_DIR}')
