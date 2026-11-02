"""
Entrena la regresión logística, la evalúa en test y permite predecir reseñas nuevas.

Orden: preparar texto → TF-IDF → regresión logística → guardar → evaluar → predecir.

Uso:
    python -m src.models.train
"""
import json
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
from sklearn.preprocessing import FunctionTransformer

from src.models.classify import regresion_logistica
from src.nlp.preprocess import preparar_lote, tfidf

RAIZ = Path(__file__).resolve().parents[2]
TRAIN = RAIZ / "data" / "processed" / "train.csv"
TEST = RAIZ / "data" / "processed" / "test.csv"
DIR_MODELOS = RAIZ / "models"
RUTA_MODELO = DIR_MODELOS / "modelo.joblib"
RUTA_METRICAS = DIR_MODELOS / "metricas.json"


def construir_pipeline(min_df, max_df, max_features=None):
    """Texto crudo → preparar_documento → TF-IDF → regresión logística."""
    return Pipeline([
        ("prep", FunctionTransformer(preparar_lote)),
        ("tfidf", tfidf(min_df=min_df, max_df=max_df, max_features=max_features)),
        ("clf", regresion_logistica()),
    ])


def evaluar(pipe, ruta_test=TEST):
    """Métricas en test: el pipe recibe el texto crudo y lo prepara él mismo."""
    df = pd.read_csv(ruta_test)
    y_true = df["label"]
    y_pred = pipe.predict(df["review"])

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "matriz_confusion": confusion_matrix(y_true, y_pred, labels=pipe.classes_).tolist(),
        "clases": list(pipe.classes_),
    }


def predecir(pipe, reseña):
    """Clase y probabilidades de una reseña nueva."""
    probs = pipe.predict_proba([reseña])[0]
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
    pipe = construir_pipeline(MIN_DF, MAX_DF, MAX_FEATURES)
    pipe.fit(df["review"], df["label"])

    DIR_MODELOS.mkdir(exist_ok=True)
    joblib.dump(pipe, RUTA_MODELO)

    metricas = evaluar(pipe)
    with open(RUTA_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    print(f"Accuracy:        {metricas['accuracy']:.4f}")
    print(f"Macro-Precision: {metricas['precision_macro']:.4f}")
    print(f"Macro-Recall:    {metricas['recall_macro']:.4f}")
    print(f"Macro-F1:        {metricas['f1_macro']:.4f}")
    print(f"Matriz de confusión {metricas['clases']}:\n{metricas['matriz_confusion']}")

    print(predecir(pipe, "La comida no estaba rico y la atencion pesima"))
