# Experiment 7 — Prototype Development, Version Control, Technical Documentation, and Reproducible Research

## Aim
To develop a working prototype for the research problem of cyberbullying detection on social media, manage the development process using version control, document the system, and make the experiment reproducible.

## Problem Statement
Cyberbullying on social-media platforms can cause significant social and psychological harm. The research prototype aims to automatically classify social-media text as cyberbullying or non-cyberbullying using machine-learning techniques.

## Objectives
1. Convert the research problem into a functional MVP.
2. Reuse the cleaned dataset and findings from Experiments 5 and 6.
3. Implement text preprocessing and TF-IDF feature extraction.
4. Develop a Linear SVM classifier.
5. Evaluate the prototype using standard classification metrics.
6. Maintain multiple development versions using Git.
7. Provide installation, execution and reproducibility instructions.

## Technologies
Python, Pandas, NumPy, Scikit-learn, Joblib, Pytest, Git, Jupyter/VS Code.

## Dataset
The cleaned dataset contains 8,792 records. It includes cleaned text and binary class labels for Cyberbullying and Non-Cyberbullying.

## Methodology
Dataset → preprocessing → TF-IDF → train/test split → Linear SVM → evaluation → saved model → prediction.

## Model Selection
Experiment 6 evaluated Logistic Regression, Decision Tree, Random Forest, SVM and KNN. SVM produced the highest 5-fold mean F1-score (0.6561, SD 0.0160), so it was selected for the final prototype.

## Version Plan
| Version | Development change |
|---|---|
| V1.0 | Initial project setup and dataset loading |
| V1.1 | Added preprocessing and TF-IDF |
| V1.2 | Added model training and evaluation |
| V1.3 | Added prediction interface and tests |
| V2.0 | Final tested prototype and documentation |

## Expected Output
A user can supply a text message and receive a binary prediction: Cyberbullying or Non-Cyberbullying.

## Reproducibility
1. Install dependencies using `pip install -r requirements.txt`.
2. Run `python src/train_model.py`.
3. Run `python src/predict.py "your text here"`.
4. Run `pytest -q`.
5. The trained model and vectorizer are saved under `results/model/`.

## Limitations
Text-only input, dataset/domain dependence, limited handling of sarcasm and context, and no multimodal inputs.

## Future Scope
Multilingual, multimodal and context-aware cyberbullying detection; lightweight transformer models; improved sarcasm handling; and human-in-the-loop deployment.
