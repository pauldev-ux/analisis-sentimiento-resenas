"""
Prepara el dataset Peruvian Food Reviews (Kaggle) para entrenar.

Pasos:
  1. Carga data/raw/reviews.csv.
  2. Quita reseñas sin texto (~40 % de las filas).
  3. Etiqueta: 1-2★ = negativo, 3★ = neutro, 4-5★ = positivo.
  4. Quita reseñas con menos de MIN_PALABRAS palabras.
  5. Quita textos con etiquetas contradictorias y deduplica ANTES del split
     (evita fuga de datos entre train y test).
  6. Muestra balanceada de N_POR_CLASE por clase (tratamiento del desbalance).
  7. Split 80/20 estratificado con semilla fija.

Uso:
    python -m src.data.prepare
    python -m src.data.prepare --n-por-clase 10000 --min-palabras 2
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[2]
RAW = RAIZ / "data" / "raw" / "reviews.csv"
OUT = RAIZ / "data" / "processed"

SEMILLA = 42
# 1-2★ = negativo, 3★ = neutro, 4-5★ = positivo
MAPA_ETIQUETAS = {1: "negativo", 2: "negativo", 3: "neutro", 4: "positivo", 5: "positivo"}


def preparar(n_por_clase: int, min_palabras: int, test_size: float) -> dict:
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

    # 5. Quitar textos repetidos con etiquetas distintas (ruido) y luego duplicados
    clave = df["review"].str.lower()
    etiquetas_por_texto = df.groupby(clave)["label"].nunique()
    contradictorios = etiquetas_por_texto[etiquetas_por_texto > 1].index
    df = df[~clave.isin(contradictorios)]
    stats["textos_contradictorios_quitados"] = len(contradictorios)

    df = df.loc[~df["review"].str.lower().duplicated()]
    stats["sin_duplicados"] = len(df)
    stats["disponibles_por_clase"] = df["label"].value_counts().to_dict()

    # 6. Muestra balanceada: mismo número de reseñas por clase
    minimo = min(stats["disponibles_por_clase"].values())
    if n_por_clase > minimo:
        raise ValueError(f"n_por_clase={n_por_clase} supera la clase más chica ({minimo})")
    muestra = df.groupby("label").sample(n=n_por_clase, random_state=SEMILLA)

    # 7. Split train/test estratificado (misma proporción de clases en ambos)
    train, test = train_test_split(
        muestra[["id_review", "review", "score", "label"]],
        test_size=test_size,
        stratify=muestra["label"],
        random_state=SEMILLA,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "train.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)

    stats.update(
        n_por_clase=n_por_clase,
        min_palabras=min_palabras,
        semilla=SEMILLA,
        train=len(train),
        test=len(test),
        train_por_clase=train["label"].value_counts().to_dict(),
        test_por_clase=test["label"].value_counts().to_dict(),
    )
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Limpia, balancea y divide el dataset.")
    p.add_argument("--n-por-clase", type=int, default=20_000)
    p.add_argument("--min-palabras", type=int, default=3)
    p.add_argument("--test-size", type=float, default=0.2)
    a = p.parse_args()
    print(json.dumps(preparar(a.n_por_clase, a.min_palabras, a.test_size), indent=2, ensure_ascii=False))