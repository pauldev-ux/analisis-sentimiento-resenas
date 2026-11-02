"""
Entrena y compara Naive Bayes vs. Regresión Logística, y guarda el mejor modelo.

Un solo comando, desde los datos crudos hasta el modelo guardado:
    python -m src.models.train

Pasos:
  1. Prepara los datos (src/data/prepare.py): limpieza, muestra balanceada y split.
  2. Preprocesa el texto con preparar_documento (normalizar, stop words, stemming, uni+bigramas).
  3. Ajusta hiperparámetros de cada modelo con GridSearchCV (validación cruzada de 5 folds, F1-macro).
  4. Elige el modelo con mejor F1-macro en validación cruzada (el test no se usa para elegir).
  5. Evalúa ambos en test y los compara contra una línea base (predecir siempre la misma clase).
  6. Guarda en models/: modelo.joblib (pipeline completo), metricas.json y errores_test.csv.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data.prepare import preparar
from src.models.classify import naive_bayes, regresion_logistica
from src.nlp.preprocess import preparar_documento, preparar_documentos, tfidf

RAIZ = Path(__file__).resolve().parents[2]
TRAIN = RAIZ / "data" / "processed" / "train.csv"
TEST = RAIZ / "data" / "processed" / "test.csv"
CARPETA = RAIZ / "models"

SEMILLA = 42
CLASES = ["negativo", "neutro", "positivo"]

# Parámetros del TF-IDF (documentados en el informe)
MIN_DF = 5        # ignora palabras que aparecen en menos de 5 reseñas (ruido, errores de tipeo)
MAX_DF = 0.9      # ignora palabras que aparecen en más del 90 % (no distinguen clases)

# Modelos a comparar y los hiperparámetros que prueba GridSearchCV
MODELOS = {
    "naive_bayes": (naive_bayes(), {"clf__alpha": [0.1, 0.5, 1.0]}),
    "regresion_logistica": (regresion_logistica(), {"clf__C": [0.1, 1.0, 10.0]}),
}


# Calcula las métricas de un modelo sobre el conjunto de test
def evaluar(y_true, y_pred) -> dict:
    reporte = classification_report(y_true, y_pred, labels=CLASES, output_dict=True, zero_division=0)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "por_clase": {
            c: {"precision": round(reporte[c]["precision"], 4),
                "recall": round(reporte[c]["recall"], 4),
                "f1": round(reporte[c]["f1-score"], 4)}
            for c in CLASES
        },
        "matriz_confusion": confusion_matrix(y_true, y_pred, labels=CLASES).tolist(),
    }


def entrenar(omitir_preparacion: bool = False) -> dict:
    # 1. Datos crudos -> train.csv / test.csv
    if not omitir_preparacion:
        preparar(n_por_clase=20_000, min_palabras=3, test_size=0.2)
    train, test = pd.read_csv(TRAIN), pd.read_csv(TEST)

    # 2. Preprocesa una sola vez para que GridSearchCV no repita este paso en cada fold
    docs_train = preparar_documentos(train["review"])
    docs_test = preparar_documentos(test["review"])

    metricas = {"clases": CLASES, "modelos": {}, "mejores_parametros": {}, "f1_cv": {}}
    mejores = {}

    # 3. Ajuste de hiperparámetros con validación cruzada para cada modelo
    for nombre, (clf, grilla) in MODELOS.items():
        pipe = Pipeline([("tfidf", tfidf(min_df=MIN_DF, max_df=MAX_DF)), ("clf", clf)])
        busqueda = GridSearchCV(pipe, grilla, cv=5, scoring="f1_macro", n_jobs=-1)
        busqueda.fit(docs_train, train["label"])
        mejores[nombre] = busqueda.best_estimator_
        metricas["mejores_parametros"][nombre] = busqueda.best_params_
        metricas["f1_cv"][nombre] = round(busqueda.best_score_, 4)
        # 5. Evaluación en test
        metricas["modelos"][nombre] = evaluar(test["label"], busqueda.predict(docs_test))
        print(f"{nombre}: F1 CV={busqueda.best_score_:.4f} | F1 test={metricas['modelos'][nombre]['f1_macro']:.4f} | {busqueda.best_params_}")

    # Línea base: predecir siempre la clase más frecuente
    base = DummyClassifier(strategy="most_frequent").fit(docs_train, train["label"])
    metricas["linea_base"] = evaluar(test["label"], base.predict(docs_test))

    # 4. El modelo en producción se elige por F1 en validación cruzada, no por test
    ganador = max(metricas["f1_cv"], key=metricas["f1_cv"].get)
    metricas["modelo_en_produccion"] = ganador

    # 6. Pipeline final: preprocesamiento + TF-IDF + modelo, recibe texto crudo
    final = Pipeline([
        ("preprocesar", FunctionTransformer(preparar_documentos)),
        ("tfidf", mejores[ganador].named_steps["tfidf"]),
        ("clf", mejores[ganador].named_steps["clf"]),
    ])
    CARPETA.mkdir(exist_ok=True)
    joblib.dump(final, CARPETA / "modelo.joblib")
    (CARPETA / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

    # Reseñas mal clasificadas por el modelo ganador (para el análisis de errores)
    pred = mejores[ganador].predict(docs_test)
    errores = test.assign(prediccion=pred)[test["label"] != pred]
    errores.sample(min(200, len(errores)), random_state=SEMILLA).to_csv(CARPETA / "errores_test.csv", index=False)

    print(f"\nEn producción: {ganador} | Línea base F1={metricas['linea_base']['f1_macro']:.4f}")
    print(f"Guardado en {CARPETA}")
    return metricas


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Entrena y compara los modelos de sentimiento.")
    p.add_argument("--omitir-preparacion", action="store_true",
                   help="usa los train.csv/test.csv existentes sin regenerarlos")
    a = p.parse_args()
    entrenar(a.omitir_preparacion)
