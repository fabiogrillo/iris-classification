import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from gensim.models import Word2Vec
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import pipeline

from src.models.lstm import LSTMClassifier, text_to_sequences, tokenize


def evaluate_tfidf(x_test, y_test):
    vectorizer = joblib.load("models/tfidf_logreg/vectorizer.joblib")
    model = joblib.load("models/tfidf_logreg/model.joblib")

    x_test_tfidf = vectorizer.transform(x_test)
    y_pred = model.predict(x_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    return y_pred, accuracy, f1


def evaluate_lstm(x_test, y_test):
    with open("configs/lstm.yaml") as f:
        config = yaml.safe_load(f)

    embedding_dim = config["embedding_dim"]
    hidden_dim = config["hidden_dim"]
    n_layers = config["n_layers"]
    dropout = config["dropout"]
    bidirectional = config["bidirectional"]
    max_len = config["max_len"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Word2Vec model
    w2v_model = Word2Vec.load("models/lstm/word2vec.model")

    # Prepare sequences
    test_tokens = [tokenize(text) for text in x_test]
    X_test_seq = np.array(
        [text_to_sequences(t, w2v_model, embedding_dim, max_len) for t in test_tokens]
    )

    # Load LSTM model
    output_dim = 2
    model = LSTMClassifier(
        embedding_dim, hidden_dim, output_dim, n_layers, dropout, bidirectional
    ).to(device)
    model.load_state_dict(torch.load("models/lstm/model.pt", map_location=device))
    model.eval()

    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test_seq, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.long),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )

    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    y_pred = np.array(all_preds)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    return y_pred, accuracy, f1


def evaluate_transformer(x_test, y_test):
    classifier = pipeline(
        "sentiment-analysis",
        model="models/distilbert-final",
        tokenizer="models/distilbert-final",
        device=0 if torch.cuda.is_available() else -1,
    )

    # Predict in batches
    results = classifier(list(x_test), batch_size=32, truncation=True)
    y_pred = np.array(
        [1 if r["label"] == "LABEL_1" else 0 for r in results]
    )

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    return y_pred, accuracy, f1


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def main():
    # Load test data
    test_df = pd.read_csv("data/processed/test_clean.csv")
    x_test = test_df["text_clean"]
    y_test = test_df["label"]

    os.makedirs("plots", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    all_metrics = {}

    # TF-IDF
    print("Evaluating TF-IDF + LogReg...")
    y_pred_tfidf, acc_tfidf, f1_tfidf = evaluate_tfidf(x_test, y_test)
    all_metrics["tfidf_accuracy"] = acc_tfidf
    all_metrics["tfidf_f1_score"] = f1_tfidf
    plot_confusion_matrix(
        y_test, y_pred_tfidf,
        f"TF-IDF + LogReg - Accuracy: {acc_tfidf:.2%}",
        "plots/confusion_matrix_tfidf.png",
    )
    print(f"  Accuracy: {acc_tfidf:.4f}, F1: {f1_tfidf:.4f}")

    # LSTM
    print("Evaluating Word2Vec + BiLSTM...")
    y_pred_lstm, acc_lstm, f1_lstm = evaluate_lstm(x_test, y_test)
    all_metrics["lstm_accuracy"] = acc_lstm
    all_metrics["lstm_f1_score"] = f1_lstm
    plot_confusion_matrix(
        y_test, y_pred_lstm,
        f"Word2Vec + BiLSTM - Accuracy: {acc_lstm:.2%}",
        "plots/confusion_matrix_lstm.png",
    )
    print(f"  Accuracy: {acc_lstm:.4f}, F1: {f1_lstm:.4f}")

    # Transformer
    print("Evaluating DistilBERT...")
    y_pred_transformer, acc_transformer, f1_transformer = evaluate_transformer(
        x_test, y_test
    )
    all_metrics["transformer_accuracy"] = acc_transformer
    all_metrics["transformer_f1_score"] = f1_transformer
    plot_confusion_matrix(
        y_test, y_pred_transformer,
        f"DistilBERT - Accuracy: {acc_transformer:.2%}",
        "plots/confusion_matrix_transformer.png",
    )
    print(f"  Accuracy: {acc_transformer:.4f}, F1: {f1_transformer:.4f}")

    # Find best model
    models = {
        "tfidf": acc_tfidf,
        "lstm": acc_lstm,
        "transformer": acc_transformer,
    }
    best_model = max(models, key=models.get)
    all_metrics["best_model"] = best_model
    all_metrics["best_accuracy"] = models[best_model]

    # Save combined metrics
    with open("metrics/evaluation.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nBest model: {best_model} (accuracy: {models[best_model]:.4f})")


if __name__ == "__main__":
    main()
