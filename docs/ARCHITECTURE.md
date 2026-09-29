# System Architecture

```text
Social-media text
       |
       v
Text cleaning / normalization
       |
       v
TF-IDF vectorization
       |
       v
Linear SVM classifier
       |
       v
Prediction: Cyberbullying / Non-Cyberbullying
       |
       v
Evaluation + saved model artifacts
```

## Components
- Input: social-media text.
- Preprocessing: lowercase, URL removal, mention/hashtag marker removal, non-alphanumeric cleanup and whitespace normalization.
- Feature extraction: TF-IDF with unigram/bigram features.
- Classifier: Linear Support Vector Machine.
- Output: binary class label.
- Persistence: Joblib model and vectorizer artifacts.
