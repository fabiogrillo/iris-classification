import mlflow
import os

def setup_mlflow(experiment_name: str = "sentiment-analysis"):
    """
    Set up MLflow experiment for tracking.

    Args:
        experiment_name (str): Name of the MLflow experiment.
    """
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment '{experiment_name}' is set up.")

    # Enable autologging for sklearn and pyTorch
    mlflow.sklearn.autolog()

    return mlflow
