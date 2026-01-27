import json
import os

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.models.lstm import LSTMClassifier, tokenize, text_to_sequences


def main():
    # Load config
    with open("configs/lstm.yaml") as f:
        config = yaml.safe_load(f)

    embedding_dim = config["embedding_dim"]
    hidden_dim = config["hidden_dim"]
    n_layers = config["n_layers"]
    dropout = config["dropout"]
    bidirectional = config["bidirectional"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    max_len = config["max_len"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_df = pd.read_csv("data/processed/train_clean.csv")
    test_df = pd.read_csv("data/processed/test_clean.csv")

    x_train = train_df["text_clean"]
    y_train = train_df["label"]
    x_test = test_df["text_clean"]
    y_test = test_df["label"]

    # Tokenize
    train_tokens = [tokenize(text) for text in x_train]
    test_tokens = [tokenize(text) for text in x_test]

    # Train Word2Vec embeddings
    w2v_model = Word2Vec(
        sentences=train_tokens,
        vector_size=embedding_dim,
        window=5,
        min_count=2,
        workers=4,
    )

    # Convert to sequences
    X_train_seq = np.array(
        [text_to_sequences(t, w2v_model, embedding_dim, max_len) for t in train_tokens]
    )
    X_test_seq = np.array(
        [text_to_sequences(t, w2v_model, embedding_dim, max_len) for t in test_tokens]
    )

    # MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Sentiment Analysis Baselines")

    with mlflow.start_run(run_name="word2vec-lstm"):
        mlflow.log_param("embedding", "word2vec")
        mlflow.log_param("embedding_dim", embedding_dim)
        mlflow.log_param("model", "lstm")
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("n_layers", n_layers)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("bidirectional", bidirectional)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        # Build model
        output_dim = 2
        model = LSTMClassifier(
            embedding_dim, hidden_dim, output_dim, n_layers, dropout, bidirectional
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # DataLoader
        train_dataset = TensorDataset(
            torch.tensor(X_train_seq, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.long),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        torch.cuda.empty_cache()
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        test_dataset = TensorDataset(
            torch.tensor(X_test_seq, dtype=torch.float32),
            torch.tensor(y_test.values, dtype=torch.long),
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("loss", epoch_loss, step=epochs - 1)
        mlflow.pytorch.log_model(model, "word2vec-lstm-model")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))

    # Save model artifacts
    os.makedirs("models/lstm", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm/model.pt")
    w2v_model.save("models/lstm/word2vec.model")

    # Save DVC metrics
    os.makedirs("metrics", exist_ok=True)
    metrics = {"accuracy": accuracy, "f1_score": f1}
    with open("metrics/lstm.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
