"""
Entrena la regresión logística, la evalúa en test y permite predecir reseñas nuevas.

Orden: preparar texto → TF-IDF → regresión logística → guardar → evaluar → predecir.

Uso:
    python -m src.models.train
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.models.classify import regresion_logistica
from src.nlp.preprocess import preparar_documento, tfidf

RAIZ = Path(__file__).resolve().parents[2]
TRAIN = RAIZ / "data" / "processed" / "train.csv"
TEST = RAIZ / "data" / "processed" / "test.csv"
RUTA_MODELO = RAIZ / "src" / "models" / "modelo.joblib"


def entrenar(X, y):
    """Entrena solo la regresión logística sobre la matriz TF-IDF."""
    modelo = regresion_logistica()
    modelo.fit(X, y)
    return modelo


def evaluar(pipe, ruta_test=TEST):
    """Métricas en test: el TF-IDF del pipe solo transforma, no se vuelve a ajustar."""
    df = pd.read_csv(ruta_test)
    docs = [preparar_documento(t) for t in df["review"]]
    y_true = df["label"]
    y_pred = pipe.predict(docs)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "matriz_confusion": confusion_matrix(y_true, y_pred, labels=pipe.classes_),
    }


def predecir(pipe, reseña):
    """Clase y probabilidades de una reseña nueva."""
    probs = pipe.predict_proba([preparar_documento(reseña)])[0]
    probabilidades = {c: round(float(p), 4) for c, p in zip(pipe.classes_, probs)}
    return {
        "clase": max(probabilidades, key=probabilidades.get),
        "probabilidades": probabilidades,
    }


if __name__ == "__main__":
    MIN_DF = 5
    MAX_DF = 0.9
    MAX_FEATURES = None

    df = pd.read_csv(TRAIN)
    docs = [preparar_documento(t) for t in df["review"]]
    y = df["label"]

    vectorizer = tfidf(min_df=MIN_DF, max_df=MAX_DF, max_features=MAX_FEATURES)
    X = vectorizer.fit_transform(docs)
    modelo = entrenar(X, y)

    pipe = Pipeline([("tfidf", vectorizer), ("clf", modelo)])
    joblib.dump(pipe, RUTA_MODELO)

    metricas = evaluar(pipe)
    print(f"Accuracy:        {metricas['accuracy']:.4f}")
    print(f"Macro-Precision: {metricas['precision_macro']:.4f}")
    print(f"Macro-Recall:    {metricas['recall_macro']:.4f}")
    print(f"Macro-F1:        {metricas['f1_macro']:.4f}")
    print(f"Matriz de confusión {list(pipe.classes_)}:\n{metricas['matriz_confusion']}")

    print(predecir(pipe, "La comida estuvo riquísima, volveré"))
