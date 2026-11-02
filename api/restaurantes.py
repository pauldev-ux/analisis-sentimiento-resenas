import json
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.modelo import predecir

# Catálogo generado por src/data/restaurantes.py
RUTA = Path(__file__).resolve().parents[1] / "data" / "processed" / "restaurantes.json"

# Agrupa los endpoints bajo /restaurantes
router = APIRouter(prefix="/restaurantes", tags=["restaurantes"])


# Carga el catálogo y clasifica todas sus reseñas con el modelo (una sola vez, al iniciar la API)
def cargar_restaurantes() -> dict[int, dict]:
    if not RUTA.exists():
        return {}
    restaurantes = json.loads(RUTA.read_text(encoding="utf-8"))
    textos = [res["texto"] for r in restaurantes for res in r["resenas"]]
    predicciones = iter(predecir(textos))
    for r in restaurantes:
        for res in r["resenas"]:
            p = next(predicciones)
            res["sentimiento"] = p["clase"]
            res["probabilidad"] = p["probabilidad"]
        r["sentimiento"] = resumir(r["resenas"])
    return {r["id"]: r for r in restaurantes}


# Porcentaje de reseñas negativas, neutras y positivas según el modelo
def resumir(resenas: list[dict]) -> dict[str, float]:
    conteo = Counter(res["sentimiento"] for res in resenas)
    total = len(resenas) or 1
    return {c: round(conteo.get(c, 0) / total, 3) for c in ["negativo", "neutro", "positivo"]}


RESTAURANTES = cargar_restaurantes()


# Datos de un restaurante sin la lista de reseñas (para el listado)
def sin_resenas(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "resenas"}


# GET /restaurantes: lista con filtro opcional por nombre o tipo de comida y por distrito
@router.get("")
def listar(buscar: str = "", distrito: str = ""):
    resultado = list(RESTAURANTES.values())
    if buscar:
        b = buscar.lower()
        resultado = [r for r in resultado if b in r["nombre"].lower() or any(b in t.lower() for t in r["tipos"])]
    if distrito:
        resultado = [r for r in resultado if r["distrito"].lower() == distrito.lower()]
    return {"total": len(resultado), "restaurantes": [sin_resenas(r) for r in resultado]}


# GET /restaurantes/distritos: distritos disponibles para el filtro
@router.get("/distritos")
def distritos():
    return sorted({r["distrito"] for r in RESTAURANTES.values()})


# GET /restaurantes/{id}: detalle con sus reseñas y el sentimiento de cada una
@router.get("/{id_restaurante}")
def detalle(id_restaurante: int):
    if id_restaurante not in RESTAURANTES:
        raise HTTPException(404, "restaurante no encontrado")
    return RESTAURANTES[id_restaurante]