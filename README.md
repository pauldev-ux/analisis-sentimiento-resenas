# Análisis de sentimiento de reseñas de restaurantes

Proyecto Integrador – Tópicos y Programación Avanzada (UAGRM).
Clasifica reseñas de restaurantes en español como **negativo / neutro / positivo** con NLP (TF-IDF) y modelos de scikit-learn, y lo expone con una API (FastAPI) y una app web (React).

## Requisitos
- Python 3.11 o superior
- Node.js 20.19 o superior (para el frontend)
- Git

## Dataset
| | |
|---|---|
| Nombre | Peruvian Food Reviews |
| Origen | Kaggle (autor: `lazaro97`): https://www.kaggle.com/datasets/lazaro97/peruvian-food-reviews |
| Licencia | _Completar con la licencia que indica la página del dataset en Kaggle_ |
| Fecha de obtención | 28/10/2026 |
| Tamaño | `reviews.csv`, 159 MB (1 258 435 filas) y `restaurants.csv`, 1,5 MB |

Los archivos originales no se suben al repo (superan el límite de GitHub), por eso `data/raw/` está en `.gitignore`.

1. Entra a la página de Kaggle (hace falta una cuenta gratuita) y pulsa **Download**.
2. Descomprime el zip y copia `data/data/reviews.csv` y `data/data/restaurants.csv` a `data/raw/`.

> La muestra procesada (`data/processed/`) y el catálogo de restaurantes sí están en el repo, así que la API y el frontend funcionan sin el dataset original. Solo hace falta para regenerar los datos desde cero.

## Instalación

```bash
git clone https://github.com/pauldev-ux/analisis-sentimiento-resenas.git
cd analisis-sentimiento-resenas

python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python -m spacy download es_core_news_sm

cd frontend
npm install
cd ..
```

## Entrenar el modelo (un solo comando)

```bash
python -m src.models.train                 # desde el CSV crudo (necesita data/raw/reviews.csv)
python -m src.models.train --sin-preparar  # reutiliza data/processed (no necesita el dataset original)
```

Prepara el dataset (limpieza, etiquetas, balanceo y split 80/20), aplica el NLP (stemming + unigramas y bigramas), entrena Naive Bayes y Regresión Logística, y deja:

- `models/modelo.joblib`: pipeline completo (`preprocesamiento → TF-IDF → regresión logística`); recibe texto crudo.
- `models/metricas.json`: métricas en test de cada modelo y la configuración usada. Lo sirve el endpoint `/metrics`.

Ambos se regeneran con el comando y no se suben a git. **Hay que entrenar antes de levantar la API**; si no, la API usa un modelo de prueba.

Los hiperparámetros se buscan en el notebook (`notebooks/eda_y_experimentos.ipynb`, secciones 2.6 y 3.2) y `src/models/train.py` los usa como constantes (`C`, `ALPHA`, `MIN_DF`, `MAX_DF`).

## Levantar el sistema

Dos terminales, con el entorno virtual activado:

```bash
# Terminal 1 – API (http://127.0.0.1:8000, documentación en /docs)
uvicorn api.main:app --reload

# Terminal 2 – Frontend (http://localhost:5173)
cd frontend
npm run dev
```

## Pruebas

```bash
python -m pytest
```

## Endpoints de la API

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/health` | Estado del servicio y modelo cargado |
| POST | `/preprocess` | Texto en cada paso del NLP |
| POST | `/predict` | Clase y probabilidad de una reseña |
| POST | `/predict/batch` | Clasifica un archivo `.txt` o `.csv` |
| GET | `/metrics` | Métricas y comparación de modelos |
| GET | `/restaurantes` | Restaurantes con el sentimiento de sus reseñas (filtros `buscar` y `distrito`) |
| GET | `/restaurantes/distritos` | Distritos disponibles |
| GET | `/restaurantes/{id}` | Detalle de un restaurante con sus reseñas |

## Estructura
```
data/raw/          dataset original (ignorado por git)
data/processed/    muestra limpia, split train/test y catálogo de restaurantes
notebooks/         EDA y experimentos
src/data/          preparación de datos y catálogo de restaurantes
src/nlp/           preprocesamiento de texto
src/models/        clasificadores y entrenamiento
models/            modelo entrenado y métricas (se generan con train.py)
api/               backend FastAPI
frontend/          app web React + Vite
tests/             pruebas de la API
docs/              informe técnico
```

