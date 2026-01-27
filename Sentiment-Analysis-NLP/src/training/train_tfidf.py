import json
import os

import mlflow
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib


def main():
    # Load config
    with open("configs/tfidf.yaml") as f:
        config = yaml.safe_load(f)

    max_features = config["max_features"]
    ngram_range = tuple(config["ngram_range"])
    C = config["C"]
    max_iter = config["max_iter"]

    # Load data
    train_df = pd.read_csv("data/processed/train_clean.csv")
    test_df = pd.read_csv("data/processed/test_clean.csv")

    x_train = train_df["text_clean"]
    y_train = train_df["label"]
    x_test = test_df["text_clean"]
    y_test = test_df["label"]

    # MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Sentiment Analysis Baselines")

    with mlflow.start_run(run_name="tfidf-logreg"):
        mlflow.log_param("vectorizer", "tfidf")
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("ngram_range", str(ngram_range))
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)

        # Vectorize
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        x_train_tfidf = vectorizer.fit_transform(x_train)
        x_test_tfidf = vectorizer.transform(x_test)

        # Train
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(x_train_tfidf, y_train)

        # Evaluate
        y_pred = model.predict(x_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "tfidf-logreg-model")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))

    # Save model artifacts
    os.makedirs("models/tfidf_logreg", exist_ok=True)
    joblib.dump(model, "models/tfidf_logreg/model.joblib")
    joblib.dump(vectorizer, "models/tfidf_logreg/vectorizer.joblib")

    # Save DVC metrics
    os.makedirs("metrics", exist_ok=True)
    metrics = {"accuracy": accuracy, "f1_score": f1}
    with open("metrics/tfidf.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
