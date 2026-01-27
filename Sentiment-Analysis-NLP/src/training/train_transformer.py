import json
import os

import evaluate
import mlflow
import pandas as pd
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def main():
    # Load config
    with open("configs/transformer.yaml") as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    num_train_epochs = config["num_train_epochs"]
    per_device_train_batch_size = config["per_device_train_batch_size"]
    per_device_eval_batch_size = config["per_device_eval_batch_size"]
    warmup_steps = config["warmup_steps"]
    weight_decay = config["weight_decay"]
    learning_rate = config["learning_rate"]
    max_length = config["max_length"]

    # MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Sentiment Analysis Transformers")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # Load data
    train_df = pd.read_csv("data/processed/train_clean.csv")
    test_df = pd.read_csv("data/processed/test_clean.csv")

    train_dataset = Dataset.from_pandas(train_df[["text_clean", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["text_clean", "label"]])

    # Tokenization
    def tokenize_function(examples):
        return tokenizer(
            examples["text_clean"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/distilbert",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir="logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = predictions.argmax(axis=-1)
        return {
            "accuracy": accuracy_metric.compute(
                predictions=preds, references=labels
            )["accuracy"],
            "f1": f1_metric.compute(predictions=preds, references=labels)["f1"],
        }

    # Train with MLflow
    with mlflow.start_run(run_name="distilbert-finetuned"):
        mlflow.log_param("model", model_name)
        mlflow.log_param("epochs", num_train_epochs)
        mlflow.log_param("batch_size", per_device_train_batch_size)
        mlflow.log_param("learning_rate", learning_rate)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # Evaluate and log metrics
        eval_results = trainer.evaluate()
        mlflow.log_metrics(eval_results)

        # Save the final model
        trainer.save_model("models/distilbert-final")
        tokenizer.save_pretrained("models/distilbert-final")

    # Extract metrics for DVC
    accuracy = eval_results.get("eval_accuracy", 0.0)
    f1 = eval_results.get("eval_f1", 0.0)

    os.makedirs("metrics", exist_ok=True)
    metrics = {"accuracy": accuracy, "f1_score": f1}
    with open("metrics/transformer.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()
