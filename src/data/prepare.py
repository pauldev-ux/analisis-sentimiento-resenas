"""
Prepara el dataset Peruvian Food Reviews (Kaggle) para entrenar.

Uso:
    python -m src.data.prepare
    python -m src.data.prepare --min-palabras 2
"""
import argparse
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RAW = RAIZ / "data" / "raw" / "reviews.csv"

# 1-2★ = negativo, 3★ = neutro, 4-5★ = positivo
MAPA_ETIQUETAS = {1: "negativo", 2: "negativo", 3: "neutro", 4: "positivo", 5: "positivo"}


def preparar(min_palabras: int) -> dict:
    stats = {}

    # 1. Cargar solo las columnas que usamos
    df = pd.read_csv(RAW, usecols=["id_review", "review", "score"])
    stats["filas_originales"] = len(df)

    # 2. Quitar reseñas sin texto (~40 % solo tienen estrellas)
    df["review"] = df["review"].astype("string").str.strip()
    df = df[df["review"].notna() & (df["review"] != "")]
    stats["con_texto"] = len(df)

    # 3. Crear la etiqueta a partir de las estrellas
    df["score"] = df["score"].astype(int)
    df["label"] = df["score"].map(MAPA_ETIQUETAS)
    stats["por_clase_con_texto"] = df["label"].value_counts().to_dict()

    # 4. Quitar reseñas muy cortas ("Bueno", "Excelente" aportan poco a TF-IDF)
    df = df[df["review"].str.split().str.len() >= min_palabras]
    stats[f"con_{min_palabras}_o_mas_palabras"] = len(df)

    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Limpia y etiqueta el dataset.")
    p.add_argument("--min-palabras", type=int, default=3)
    a = p.parse_args()
    print(json.dumps(preparar(a.min_palabras), indent=2, ensure_ascii=False))