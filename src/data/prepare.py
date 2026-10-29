"""
Prepara el dataset Peruvian Food Reviews (Kaggle) para entrenar.

Uso:
    python -m src.data.prepare
"""
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RAW = RAIZ / "data" / "raw" / "reviews.csv"


def preparar() -> dict:
    stats = {}

    # 1. Cargar solo las columnas que usamos
    df = pd.read_csv(RAW, usecols=["id_review", "review", "score"])
    stats["filas_originales"] = len(df)

    # 2. Quitar reseñas sin texto (~40 % solo tienen estrellas)
    df["review"] = df["review"].astype("string").str.strip()
    df = df[df["review"].notna() & (df["review"] != "")]
    stats["con_texto"] = len(df)

    return stats


if __name__ == "__main__":
    print(json.dumps(preparar(), indent=2, ensure_ascii=False))