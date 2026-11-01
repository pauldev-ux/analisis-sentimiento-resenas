from pathlib import Path

import joblib

from src.nlp.preprocess import normalizar

# Ruta donde train.py guarda el pipeline entrenado (TF-IDF + modelo)
RUTA_MODELO = Path(__file__).resolve().parents[1] / "models" / "modelo.joblib"


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