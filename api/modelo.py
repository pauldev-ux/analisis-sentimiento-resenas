import json
from pathlib import Path

import joblib

from src.nlp.preprocess import normalizar

# Ruta donde train.py guarda el pipeline entrenado (TF-IDF + modelo)
RUTA_MODELO = Path(__file__).resolve().parents[1] / "models" / "modelo.joblib"


# Ruta donde train.py guarda las métricas de evaluación de los modelos
RUTA_METRICAS = Path(__file__).resolve().parents[1] / "models" / "metricas.json"

# Métricas de ejemplo con el mismo formato que debe generar train.py (solo para desarrollo)
METRICAS_DEMO = {
    "modelo_en_produccion": "regresion_logistica",
    "clases": ["negativo", "neutro", "positivo"],
    "modelos": {
        "naive_bayes": {
            "accuracy": 0.70, "f1_macro": 0.69,
            "por_clase": {
                "negativo": {"precision": 0.74, "recall": 0.76, "f1": 0.75},
                "neutro": {"precision": 0.60, "recall": 0.55, "f1": 0.57},
                "positivo": {"precision": 0.75, "recall": 0.79, "f1": 0.77},
            },
            "matriz_confusion": [[3040, 760, 200], [880, 2200, 920], [190, 650, 3160]],
        },
        "regresion_logistica": {
            "accuracy": 0.73, "f1_macro": 0.72,
            "por_clase": {
                "negativo": {"precision": 0.77, "recall": 0.79, "f1": 0.78},
                "neutro": {"precision": 0.63, "recall": 0.60, "f1": 0.61},
                "positivo": {"precision": 0.78, "recall": 0.80, "f1": 0.79},
            },
            "matriz_confusion": [[3160, 680, 160], [760, 2400, 840], [180, 620, 3200]],
        },
    },
}


# Modelo provisional con la misma interfaz que un pipeline de scikit-learn.
# Solo sirve para desarrollar la API y el frontend mientras no exista el modelo real.
class ModeloDemo:
    classes_ = ["negativo", "neutro", "positivo"]
    POSITIVAS = {"rico", "excelente", "bueno", "buena", "delicioso", "recomendado"}
    NEGATIVAS = {"malo", "mala", "pesimo", "pesima", "horrible", "no", "nunca"}

    # Devuelve [p_negativo, p_neutro, p_positivo] contando palabras clave
    def predict_proba(self, textos):
        resultado = []
        for texto in textos:
            palabras = set(normalizar(texto).split())
            pos = len(palabras & self.POSITIVAS)
            neg = len(palabras & self.NEGATIVAS)
            if pos > neg:
                resultado.append([0.1, 0.2, 0.7])
            elif neg > pos:
                resultado.append([0.7, 0.2, 0.1])
            else:
                resultado.append([0.25, 0.5, 0.25])
        return resultado


# Carga el modelo real si existe; si no, usa el modelo demo
def cargar_modelo():
    if RUTA_MODELO.exists():
        return joblib.load(RUTA_MODELO), "entrenado"
    return ModeloDemo(), "demo"


MODELO, TIPO_MODELO = cargar_modelo()

# Lee las métricas guardadas por train.py; si no existen, devuelve las de ejemplo
def cargar_metricas() -> dict:
    if RUTA_METRICAS.exists():
        return {**json.loads(RUTA_METRICAS.read_text(encoding="utf-8")), "origen": "entrenado"}
    return {**METRICAS_DEMO, "origen": "demo"}


# Predice la clase y las probabilidades de cada texto
def predecir(textos: list[str]) -> list[dict]:
    resultados = []
    for probs in MODELO.predict_proba(textos):
        por_clase = {c: round(float(p), 4) for c, p in zip(MODELO.classes_, probs)}
        clase = max(por_clase, key=por_clase.get)
        resultados.append({
            "clase": clase,
            "probabilidad": por_clase[clase],
            "probabilidades": por_clase,
        })
    return resultados