"""
Entrena el modelo final y lo deja listo para la API, en un solo comando.

Pasos:
  1. Prepara el dataset (src/data/prepare.py): data/raw/reviews.csv -> data/processed/{train,test}.csv
  2. Aplica el NLP una sola vez (normalizar → tokenizar → stop words → stemming → uni+bigramas)
  3. Entrena TF-IDF + regresión logística (modelo en producción) y Naive Bayes (comparación)
  4. Guarda models/modelo.joblib (pipeline que recibe texto crudo) y models/metricas.json

Uso:
    python -m src.models.train                  # desde el CSV crudo hasta el modelo
    python -m src.models.train --sin-preparar   # reutiliza data/processed (reentrenar rápido)

La búsqueda de hiperparámetros se hace en el notebook (secciones 2.6 y 3.2);
aquí solo se entrena la versión final con los valores ganadores.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data.prepare import MIN_PALABRAS, N_POR_CLASE, OUT, RAW, TEST_SIZE, preparar
from src.models.classify import SEMILLA, naive_bayes, regresion_logistica
from src.nlp.preprocess import preparar_lote, tfidf

RAIZ = Path(__file__).resolve().parents[2]
TRAIN = OUT / "train.csv"
TEST = OUT / "test.csv"
RUTA_STATS = OUT / "stats.json"
DIR_MODELOS = RAIZ / "models"
RUTA_MODELO = DIR_MODELOS / "modelo.joblib"
RUTA_METRICAS = DIR_MODELOS / "metricas.json"

# Hiperparámetros ganadores (notebooks/eda_y_experimentos.ipynb)
# TF-IDF, sección 2.6: configuración sin filtro (la de mayor Macro-F1 en validación cruzada)
MIN_DF = 1
MAX_DF = 1.0
MAX_FEATURES = None
# Sección 3.2: GridSearchCV de 5 folds con f1_macro
C = 0.5          # regresión logística (modelo en producción)
ALPHA = 1.0      # Naive Bayes (solo para la comparación)

MODELO_PRODUCCION = "regresion_logistica"


def entrenar(docs, y, clasificador):
    """TF-IDF + clasificador sobre texto que ya pasó por el NLP."""
    pipe = Pipeline([
        ("tfidf", tfidf(min_df=MIN_DF, max_df=MAX_DF, max_features=MAX_FEATURES)),
        ("clf", clasificador),
    ])
    return pipe.fit(docs, y)


def con_preprocesamiento(pipe):
    """Antepone el paso 'prep': el pipeline guardado recibe texto crudo (prep → tfidf → clf)."""
    prep = FunctionTransformer(preparar_lote)
    return Pipeline([("prep", prep), *pipe.steps])


def calcular_metricas(y_true, y_pred, clases):
    """Métricas en test: accuracy, macro, por clase y matriz de confusión (formato de /metrics)."""
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=clases, zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(prec.mean()), 4),
        "recall_macro": round(float(rec.mean()), 4),
        "f1_macro": round(float(f1.mean()), 4),
        "por_clase": {
            c: {
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1": round(float(f1[i]), 4),
            }
            for i, c in enumerate(clases)
        },
        "matriz_confusion": confusion_matrix(y_true, y_pred, labels=clases).tolist(),
    }


def describir_configuracion(modelos, stats):
    """Configuración usada, tomada de los objetos entrenados (no copiada a mano)."""
    vec = modelos[MODELO_PRODUCCION].named_steps["tfidf"]
    lr = modelos["regresion_logistica"].named_steps["clf"].get_params()
    nb = modelos["naive_bayes"].named_steps["clf"].get_params()
    return {
        "tecnica_nlp": "stemming (SnowballStemmer en español)",
        "pasos_nlp": [
            "normalización (minúsculas, URLs, números, signos y tildes)",
            "tokenización",
            "stop words en español (se conservan las negaciones)",
            "stemming",
            "unigramas + bigramas",
        ],
        "ngramas": "unigramas + bigramas",
        "tfidf": {
            "min_df": vec.min_df,
            "max_df": vec.max_df,
            "max_features": vec.max_features,
            "ngram_range": list(vec.ngram_range),
            "vocabulario": len(vec.vocabulary_),
            "nota": "los bigramas ya vienen unidos con '_' en el texto, por eso ngram_range es (1, 1)",
        },
        "hiperparametros": {
            "regresion_logistica": {"C": lr["C"], "max_iter": lr["max_iter"]},
            "naive_bayes": {"alpha": nb["alpha"]},
        },
        "semilla": SEMILLA,
        "datos": {
            k: stats[k]
            for k in ("n_por_clase", "min_palabras", "train", "test", "semilla")
            if k in stats
        },
    }


def predecir(pipe, reseña):
    """Clase y probabilidades de una reseña nueva (el pipeline recibe texto crudo)."""
    probs = pipe.predict_proba([reseña])[0]
    probabilidades = {c: round(float(p), 4) for c, p in zip(pipe.classes_, probs)}
    return {
        "clase": max(probabilidades, key=probabilidades.get),
        "probabilidades": probabilidades,
    }


def verificar_modelo_guardado(modelo, docs_test, test, n=200):
    """El .joblib cargado desde disco debe predecir con texto crudo igual que el modelo entrenado."""
    cargado = joblib.load(RUTA_MODELO)
    esperado = list(modelo.predict(docs_test[:n]))
    obtenido = list(cargado.predict(test["review"].head(n)))
    if esperado != obtenido:
        raise RuntimeError("El modelo guardado no coincide con el entrenado: revisa el paso 'prep'.")
    return cargado


def main():
    ap = argparse.ArgumentParser(description="Entrena el modelo desde el CSV crudo.")
    ap.add_argument(
        "--sin-preparar",
        action="store_true",
        help="no vuelve a preparar el dataset: usa data/processed (reentrenar rápido)",
    )
    args = ap.parse_args()

    if args.sin_preparar:
        if not (TRAIN.exists() and TEST.exists()):
            sys.exit(f"No existen {TRAIN} y {TEST}. Ejecuta sin --sin-preparar para generarlos.")
        stats = json.loads(RUTA_STATS.read_text(encoding="utf-8")) if RUTA_STATS.exists() else {}
        print("1/4 Dataset: se reutiliza data/processed (--sin-preparar)")
    else:
        if not RAW.exists():
            sys.exit(
                f"No se encontró {RAW}.\n"
                "Descarga el dataset de Kaggle como indica el README, "
                "o usa --sin-preparar si ya tienes data/processed."
            )
        print("1/4 Preparando el dataset…")
        stats = preparar(N_POR_CLASE, MIN_PALABRAS, TEST_SIZE)

    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    print(f"    train={len(train)}  test={len(test)}")

    print("2/4 Aplicando el NLP (una sola vez)…")
    docs_train = preparar_lote(train["review"])
    docs_test = preparar_lote(test["review"])

    print("3/4 Entrenando…")
    modelos = {
        "naive_bayes": entrenar(docs_train, train["label"], naive_bayes(alpha=ALPHA)),
        "regresion_logistica": entrenar(docs_train, train["label"], regresion_logistica(C=C)),
    }
    clases = list(modelos[MODELO_PRODUCCION].classes_)
    metricas = {
        "modelo_en_produccion": MODELO_PRODUCCION,
        "clases": clases,
        "modelos": {
            nombre: calcular_metricas(test["label"], m.predict(docs_test), clases)
            for nombre, m in modelos.items()
        },
        "configuracion": describir_configuracion(modelos, stats),
    }

    print("4/4 Guardando…")
    DIR_MODELOS.mkdir(exist_ok=True)
    joblib.dump(con_preprocesamiento(modelos[MODELO_PRODUCCION]), RUTA_MODELO)
    cargado = verificar_modelo_guardado(modelos[MODELO_PRODUCCION], docs_test, test)
    RUTA_METRICAS.write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nModelo en producción: {MODELO_PRODUCCION} (C={C})  ->  {RUTA_MODELO}")
    print(f"Métricas y configuración                     ->  {RUTA_METRICAS}\n")
    print(f"{'modelo':<22}{'accuracy':>10}{'f1_macro':>10}")
    for nombre, m in metricas["modelos"].items():
        print(f"{nombre:<22}{m['accuracy']:>10.4f}{m['f1_macro']:>10.4f}")
    print(f"Matriz de confusión {clases}:\n{metricas['modelos'][MODELO_PRODUCCION]['matriz_confusion']}")
    print(predecir(cargado, "La comida no estaba rico y la atencion pesima"))


if __name__ == "__main__":
    main()
