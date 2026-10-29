# Análisis de sentimiento de reseñas de restaurantes

Proyecto Integrador – Tópicos y Programación Avanzada (UAGRM).
Clasifica reseñas en español como negativo / neutro / positivo usando NLP (TF-IDF) y modelos de scikit-learn.

## Dataset
Peruvian Food Reviews (Kaggle): https://www.kaggle.com/datasets/lazaro97/peruvian-food-reviews

No se sube al repo (159 MB). Descarga el zip y copia `data/data/reviews.csv` a `data/raw/reviews.csv`.

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
