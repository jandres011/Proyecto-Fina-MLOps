import os
from pipeline import setup_mlflow, train_model

if __name__ == "__main__":
    setup_mlflow()
    result = train_model()
    print(f"Entrenamiento completado. Accuracy: {result['accuracy']:.4f}")