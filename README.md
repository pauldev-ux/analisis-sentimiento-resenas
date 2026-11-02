# Análisis de sentimiento de reseñas de restaurantes

Proyecto Integrador – Tópicos y Programación Avanzada (UAGRM).
Clasifica reseñas en español como negativo / neutro / positivo usando NLP (TF-IDF) y modelos de scikit-learn.

## Dataset
| | |
|---|---|
| Nombre | Peruvian Food Reviews |
| Origen | Kaggle (autor: `lazaro97`): https://www.kaggle.com/datasets/lazaro97/peruvian-food-reviews |
| Licencia | _Pendiente de completar: copiar la licencia que indica la página del dataset en Kaggle._ |
| Fecha de obtención | _Pendiente de completar (fecha en que se descargó)._ |
| Tamaño | `reviews.csv`, 159 MB (1 258 435 filas) |

No se sube al repo (159 MB, supera el límite de GitHub), por eso `data/raw/` está en `.gitignore`.

Cómo obtenerlo:
1. Entra a la página de Kaggle (hace falta una cuenta gratuita) y pulsa **Download**.
2. Descomprime el zip y copia `data/data/reviews.csv` a `data/raw/reviews.csv`.

## Entrenar el modelo (un solo comando)
Con `data/raw/reviews.csv` en su sitio, desde la raíz del proyecto:

```bash
python -m src.models.train
```

Va del CSV crudo al modelo: prepara el dataset (limpieza, etiquetas, balanceo y split 80/20), aplica el NLP
(stemming + unigramas y bigramas), entrena TF-IDF + regresión logística y deja:

- `models/modelo.joblib`: el pipeline completo (`prep → TF-IDF → regresión logística`); recibe texto crudo.
- `models/metricas.json`: métricas en test de cada modelo y la configuración usada (técnica de NLP, n-gramas,
  parámetros de TF-IDF, hiperparámetros y semilla). Lo sirve el endpoint `/metrics`.

Ambos archivos se regeneran con el comando y no se suben a git.

Para reentrenar rápido sin volver a preparar el dataset (por ejemplo, tras cambiar un hiperparámetro):

```bash
python -m src.models.train --sin-preparar
```

Los hiperparámetros se buscan en el notebook (`notebooks/eda_y_experimentos.ipynb`, secciones 2.6 y 3.2) y
`src/models/train.py` los usa como constantes (`C`, `MIN_DF`, `MAX_DF`, `MAX_FEATURES`).

## Estructura
- `data/raw/` – dataset original (ignorado por git)
- `data/processed/` – datos limpios
- `notebooks/` – EDA
- `src/data/`, `src/nlp/`, `src/models/` – datos, preprocesamiento y modelos
- `models/` – modelos entrenados
- `api/` – backend
- `frontend/` – app web
- `tests/` – pruebas
- `docs/` – informe
