# Cyberbullying Detection on Social Media — Research Prototype

## Project
A reproducible minimum working prototype for detecting cyberbullying in social-media text.

## Research basis
The prototype extends Experiments 5 and 6. Experiment 5 cleaned and engineered the cyberbullying dataset. Experiment 6 compared five TF-IDF-based classifiers and used 5-fold cross-validation. Linear SVM was selected for the prototype because it achieved the highest mean cross-validation F1-score (0.6561) among the tested models.

## Dataset
`data/Professor_Friendly_Cyberbullying_Dataset.csv` contains 8,792 records and the columns `Record_ID`, `Original_Text`, `Cleaned_Text`, `Text_Length`, `Word_Count`, `Hashtag_Count`, `Mention_Count`, `Class_Label`, and `Class_Name`.

## Method
1. Load cleaned text.
2. Split data into stratified training/test sets (80/20).
3. Extract TF-IDF features (up to 5,000 features, unigram/bigram).
4. Train Linear SVM.
5. Evaluate accuracy, precision, recall, F1 and ROC-AUC.
6. Save the model and vectorizer for reproducible inference.

## Installation
```bash
pip install -r requirements.txt
```

## Train
```bash
python src/train_model.py
```

## Predict
After training:
```bash
python src/predict.py "you are a terrible person"
```

## Test
```bash
pytest -q
```

## Reproducibility
Use Python 3.10+ and install the pinned/minimum dependencies from `requirements.txt`. The training script uses a fixed random seed (`42`) and saves the fitted vectorizer and model under `results/model/`.

## Version history
- V1.0 — Initial prototype and dataset loading
- V1.1 — Added preprocessing and TF-IDF feature extraction
- V1.2 — Added baseline classifiers and evaluation
- V1.3 — Added prediction interface and validation test
- V2.0 — Final tested SVM prototype and documentation

## Limitations
The prototype is text-only, is based on the available labelled dataset, and may struggle with sarcasm, multilingual content, context, images, emojis, and videos. It should be treated as a research prototype rather than an automated moderation decision-maker.

## Future scope
Multilingual and multimodal detection, contextual modelling, lightweight transformer models, better sarcasm detection, and deployment with human review.

## Push to GitHub

See [`GITHUB_PUSH.md`](GITHUB_PUSH.md) for instructions to connect this repository to a new GitHub remote while preserving its existing version history. Check dataset redistribution permissions before making the repository public.
