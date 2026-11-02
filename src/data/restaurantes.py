"""
Genera data/processed/restaurantes.json con los restaurantes que más reseñas
escritas tienen y una muestra de sus reseñas. Lo usa la API (/restaurantes).

Uso:
    python -m src.data.restaurantes
    python -m src.data.restaurantes --n-restaurantes 50 --resenas-por-restaurante 20
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RAW = RAIZ / "data" / "raw"
SALIDA = RAIZ / "data" / "processed" / "restaurantes.json"
SEMILLA = 42

# Traduce la fecha relativa del dataset: "2 years ago" -> "hace 2 años"
UNIDADES = {"year": ("año", "años"), "month": ("mes", "meses"), "week": ("semana", "semanas"),
            "day": ("día", "días"), "hour": ("hora", "horas")}


def traducir_fecha(texto: str) -> str:
    m = re.match(r"(\d+)\s+(year|month|week|day|hour)s?\s+ago", str(texto))
    if not m:
        return str(texto)
    n, unidad = int(m.group(1)), m.group(2)
    singular, plural = UNIDADES[unidad]
    return f"hace {n} {singular if n == 1 else plural}"


# Separa el campo tag ("$$ - $$$||Peruana||Mariscos||") en precio y tipos de comida
def separar_tag(tag: str) -> tuple[str | None, list[str]]:
    partes = [p.strip() for p in str(tag).split("||") if p.strip()]
    precio = next((p for p in partes if p.startswith("$")), None)
    tipos = [p for p in partes if not p.startswith("$") and p != "Restaurante"]
    return precio, tipos


# Arma el catálogo: los restaurantes con más reseñas y una muestra de cada uno
def generar(n_restaurantes: int, resenas_por_restaurante: int) -> dict:
    resenas = pd.read_csv(RAW / "reviews.csv", usecols=["review", "score", "service", "date"])
    resenas = resenas.dropna(subset=["review"])
    resenas = resenas[resenas["review"].str.split().str.len() >= 3]  # mismo filtro que prepare.py

    restaurantes = pd.read_csv(RAW / "restaurants.csv")

    # Restaurantes con más reseñas escritas (la columna "service" es el id del restaurante)
    conteo = resenas["service"].value_counts().head(n_restaurantes)
    elegidos = restaurantes[restaurantes["id"].isin(conteo.index)].copy()
    elegidos["n_resenas_texto"] = elegidos["id"].map(conteo)
    elegidos = elegidos.sort_values("n_resenas_texto", ascending=False)

    salida = []
    for _, r in elegidos.iterrows():
        precio, tipos = separar_tag(r["tag"])
        propias = resenas[resenas["service"] == r["id"]]
        muestra = propias.sample(min(resenas_por_restaurante, len(propias)), random_state=SEMILLA)
        salida.append({
            "id": int(r["id"]),
            "nombre": r["name"],
            "distrito": str(r["district"]).strip().title() or "Sin distrito",
            "direccion": r["direction"],
            "estrellas": round(r["stars"] / 10, 1),       # 45 -> 4.5
            "n_resenas": int(r["n_reviews"]),
            "precio": precio,
            "tipos": tipos,
            "plataforma": r["platform"],
            "resenas": [
                {"texto": t, "estrellas": int(s), "fecha": traducir_fecha(f)}
                for t, s, f in zip(muestra["review"], muestra["score"], muestra["date"])
            ],
        })

    SALIDA.write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"restaurantes": len(salida), "resenas": sum(len(r["resenas"]) for r in salida)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Genera el catálogo de restaurantes para la API.")
    p.add_argument("--n-restaurantes", type=int, default=100)
    p.add_argument("--resenas-por-restaurante", type=int, default=30)
    a = p.parse_args()
    print(generar(a.n_restaurantes, a.resenas_por_restaurante))