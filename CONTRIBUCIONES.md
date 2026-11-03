# Contribuciones

Este archivo resume lo que hizo cada integrante. Coincide con el historial de commits del repositorio.

## Paul Moreno Montenegro 220001960 (`pauldev-ux`)
- **Datos:** elección del tema y del dataset, análisis exploratorio y `src/data/prepare.py` (limpieza, etiquetado, deduplicación, muestra balanceada y split estratificado).
- **NLP:** normalización, tokenización y stop words en español conservando negaciones (`src/nlp/preprocess.py`).
- **Notebook:** EDA con gráficos y ejemplos antes/después del pipeline de NLP.
- **Backend:** API en FastAPI (`/health`, `/preprocess`, `/predict`, `/predict/batch`, `/metrics`), validación de entradas, carga del modelo y pruebas.
- **Restaurantes:** catálogo de restaurantes (`src/data/restaurantes.py`) y endpoints `/restaurantes`.
- **Frontend:** app en React + Vite con las pestañas de análisis, archivo, restaurantes y métricas.
- **Repositorio:** estructura, ramas, protección de `main` y revisión de pull requests.

## Josefine Nataly Condori Verbo  217076653 (`quantworld95 y stellar123po`)
- **NLP:** stemming (SnowballStemmer) y lematización (spaCy), unigramas y bigramas, TF-IDF.
- **Notebook:** comparación de stemming vs. lematización, de n-gramas y de configuraciones de TF-IDF; ajuste de modelos con GridSearchCV.
- **Modelos:** entrenamiento de Regresión Logística y Naive Bayes, pipeline con el preprocesamiento incluido y entrenamiento reproducible desde el CSV crudo (`src/models/train.py`, `src/models/classify.py`).

## 3er integrante (`CrisJoelSH`)
- _Completar con sus commits._

## 4to integrante (`RounerFM`)
- _Completar con sus commits._