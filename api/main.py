import csv
import io
from collections import Counter

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from api.modelo import TIPO_MODELO, cargar_metricas, predecir
from api.restaurantes import router as router_restaurantes

# Funciones de NLP reutilizadas (la API no reimplementa el preprocesamiento)
from src.nlp.preprocess import STOP_WORDS, normalizar, quitar_stop_words, tokenizar

# Límite de caracteres por reseña
MAX_CARACTERES = 2000
# Límite de reseñas por archivo en /predict/batch
MAX_RESENAS_ARCHIVO = 500


# Crea la aplicación; título y descripción aparecen en /docs
app = FastAPI(
    title="API de Análisis de Sentimiento",
    description="Clasifica reseñas de restaurantes como negativo, neutro o positivo.",
    version="0.1.0",
)


# Endpoints de restaurantes (definidos en api/restaurantes.py)
app.include_router(router_restaurantes)

# Define el JSON que recibe la API: {"texto": "..."}, obligatorio y con largo máximo
class TextoEntrada(BaseModel):
    texto: str = Field(
        ...,
        max_length=MAX_CARACTERES,
        description=f"Reseña a analizar (máximo {MAX_CARACTERES} caracteres).",
        examples=["La comida estuvo deliciosa pero la atención fue lenta"],
    )

    # Rechaza textos vacíos o que no tienen ninguna palabra
    @field_validator("texto")
    @classmethod
    def no_vacio(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("el texto no puede estar vacío")
        if not normalizar(v):
            raise ValueError("el texto no tiene palabras (solo números, signos o emojis)")
        return v


# Convierte los errores de validación en mensajes claros en español (código 422)
@app.exception_handler(RequestValidationError)
async def error_validacion(request: Request, exc: RequestValidationError):
    errores = []
    for e in exc.errors():
        campo = ".".join(str(p) for p in e["loc"] if p != "body") or "body"
        if e["type"] == "string_too_long":
            msg = f"el texto supera el máximo de {MAX_CARACTERES} caracteres"
        elif e["type"] == "missing":
            msg = "falta el campo obligatorio"
        elif e["type"] == "string_type":
            msg = "debe ser un texto"
        else:
            msg = e["msg"].removeprefix("Value error, ")
        errores.append({"campo": campo, "error": msg})
    return JSONResponse(status_code=422, content={"detalle": errores})


# GET /health: indica si la API está funcionando y qué modelo tiene cargado
@app.get("/health", tags=["estado"])
def health():
    return {"status": "ok", "modelo": TIPO_MODELO}


# POST /preprocess: devuelve el texto en cada paso del NLP (para la vista paso a paso)
@app.post("/preprocess", tags=["nlp"])
def preprocess(entrada: TextoEntrada):
    normalizado = normalizar(entrada.texto)
    tokens = tokenizar(normalizado)
    sin_stop = quitar_stop_words(tokens)
    return {
        "original": entrada.texto,
        "normalizado": normalizado,
        "tokens": tokens,
        "stop_words_eliminadas": [t for t in tokens if t in STOP_WORDS],
        "sin_stop_words": sin_stop,
    }


# POST /predict: clasifica una reseña y devuelve la clase con su probabilidad
@app.post("/predict", tags=["modelo"])
def predict(entrada: TextoEntrada):
    return {"texto": entrada.texto, **predecir([entrada.texto])[0], "modelo": TIPO_MODELO}



# Lee un .txt (una reseña por línea) o un .csv (columna "texto" o la primera) y devuelve las reseñas
def leer_archivo(nombre: str, contenido: bytes) -> list[str]:
    try:
        texto = contenido.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(400, "el archivo debe estar en UTF-8")
    if nombre.lower().endswith(".txt"):
        filas = texto.splitlines()
    elif nombre.lower().endswith(".csv"):
        lector = csv.DictReader(io.StringIO(texto))
        if not lector.fieldnames:
            raise HTTPException(400, "el CSV está vacío")
        columna = "texto" if "texto" in lector.fieldnames else lector.fieldnames[0]
        filas = [fila[columna] or "" for fila in lector]
    else:
        raise HTTPException(400, "formato no soportado: usa .txt o .csv")
    return [f.strip() for f in filas if f.strip()]


# POST /predict/batch: clasifica varias reseñas subidas en un archivo .txt o .csv
@app.post("/predict/batch", tags=["modelo"])
async def predict_batch(archivo: UploadFile = File(..., description="Archivo .txt o .csv con reseñas")):
    resenas = leer_archivo(archivo.filename or "", await archivo.read())
    if not resenas:
        raise HTTPException(400, "el archivo no tiene reseñas")
    if len(resenas) > MAX_RESENAS_ARCHIVO:
        raise HTTPException(400, f"máximo {MAX_RESENAS_ARCHIVO} reseñas por archivo")
    # Se descartan las líneas que no tienen ninguna palabra
    validas = [r[:MAX_CARACTERES] for r in resenas if normalizar(r)]
    resultados = [{"texto": t, **p} for t, p in zip(validas, predecir(validas))]
    return {
        "total": len(resultados),
        "descartadas": len(resenas) - len(validas),
        "resumen": dict(Counter(r["clase"] for r in resultados)),
        "resultados": resultados,
        "modelo": TIPO_MODELO,
    }


# GET /metrics: devuelve las métricas de los modelos y su comparación
@app.get("/metrics", tags=["modelo"])
def metrics():
    return cargar_metricas()

