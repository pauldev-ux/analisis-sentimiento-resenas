# Informe técnico – Análisis de sentimiento de reseñas de restaurantes

**Materia:** Tópicos y Programación Avanzada – UAGRM
**Integrantes:** Paul Moreno Montenegro, Nataly Condori Verbo

## 1. Problema y objetivo

Un restaurante acumula reseñas en Google y TripAdvisor más rápido de lo que se pueden leer. Las quejas quedan mezcladas con los elogios y se atienden tarde.

El objetivo es clasificar cada reseña en español en una de tres clases —**negativa, neutra o positiva**— y devolver la probabilidad de cada clase. Con eso se puede ver en segundos cómo se sienten los clientes y priorizar las quejas.

El alcance es el sentimiento general de la reseña. El sistema no indica si la queja es de la comida, del precio o de la atención.

## 2. Dataset

Se usó **Peruvian Food Reviews** (Kaggle, autor `lazaro97`): reseñas de restaurantes de Lima en Google Places y TripAdvisor, entre 2010 y 2021.

| | |
|---|---|
| Origen | https://www.kaggle.com/datasets/lazaro97/peruvian-food-reviews |
| Tamaño | `reviews.csv`, 159 MB, 1 258 435 filas y 9 871 restaurantes |
| Columnas usadas | Texto en `review`, estrellas (1 a 5) en `score`, restaurante en `service` |
| Licencia y fecha de descarga | No especificada en Kaggle; uso académico. Descargado el 28/10/2026 |

El CSV no se sube al repositorio. `data/raw/` está en `.gitignore`.

**Etiqueta.** Las estrellas se convierten en clase: 1–2★ = negativo, 3★ = neutro, 4–5★ = positivo. El 3★ se trata como neutro porque esas reseñas suelen mezclar elogio y queja.

### Análisis exploratorio

El detalle está en `notebooks/eda_y_experimentos.ipynb`, sección 1. Los conteos de la preparación salen de `data/processed/stats.json`.

| Hallazgo | Dato | Decisión |
|---|---|---|
| Reseñas sin texto | 503 369 de 1 258 435 (40 %). Quedan 755 066 con texto | Se eliminan: no hay qué clasificar |
| Desbalance (sobre las que tienen texto) | Positivo 599 674 (79 %), neutro 90 142 (12 %), negativo 65 250 (9 %) | Muestra balanceada y **F1-macro** como métrica principal |
| Reseñas muy cortas | Mediana de 7 palabras; 166 726 (22 % de las que tienen texto) tienen menos de 3 | Se descartan. «Bueno» o «Excelente» aportan poco a TF-IDF. Quedan 588 340 |
| Textos repetidos y etiquetas contradictorias | 2 715 textos aparecen con estrellas de clases distintas | Se quitan esos textos y se deduplica **antes** del split. Quedan 537 838 reseñas únicas |
| Palabras frecuentes | «no» es la más frecuente en las negativas | Se **excluye** de las stop words, junto con otras negaciones e intensificadores |

Tras la limpieza, la clase más pequeña alcanza 54 421 reseñas (negativo), así que una muestra de 20 000 por clase es viable.

**Dataset de entrenamiento** (`src/data/prepare.py`): 20 000 reseñas por clase, split 80/20 estratificado, semilla 42. Resultado: 48 000 de entrenamiento (16 000 por clase) y 12 000 de prueba (4 000 por clase). El conjunto de prueba no se usa para elegir técnica ni hiperparámetros.

## 3. Pipeline de NLP

Implementado en `src/nlp/preprocess.py`. El modelo guardado aplica esta cadena completa y recibe el texto crudo.

| Paso | Qué hace | Por qué |
|---|---|---|
| Normalización | Minúsculas; quita URLs, tildes (conserva la ñ), números, signos y emojis; reduce letras repetidas («riquísimaaa» → «riquisimaa») | Une variantes de la misma palabra. La ñ se conserva porque cambia el significado (año ≠ ano) |
| Tokenización | Separación por espacios | Después de normalizar no quedan signos |
| Stop words | Lista de NLTK en español, **sin** quitar: no, ni, sin, nada, nunca, tampoco, muy, poco, mucho, más, pero | Sin «no», «no me gustó» quedaría como «gustó». «pero» marca el contraste típico de las neutras |
| Stemming | SnowballStemmer en español. También se implementó lematización con spaCy y se comparó en el notebook (sección 2.4) | Se eligió stemming: F1 similar y mucho más rápido, necesario para reentrenar |
| N-gramas | Unigramas frente a unigramas + bigramas (sección 2.5) | Se eligió **uni + bigramas**. Capturan «no_gust» o «muy_buen». Van unidos con `_` |
| TF-IDF | `min_df=1`, `max_df=1.0`, sin tope de vocabulario (261 180 términos). `ngram_range=(1, 1)` porque los bigramas ya vienen armados | Fue la configuración de mayor F1-macro en validación cruzada (sección 2.6) |

## 4. Modelos y resultados

Se compararon **Naive Bayes multinomial** y **regresión logística**, los dos sobre el mismo TF-IDF (`src/models/classify.py` y `src/models/train.py`).

Los hiperparámetros se buscaron con **GridSearchCV de 5 folds** y F1-macro (notebook, sección 3.2). Ganadores: `alpha = 1.0` (Naive Bayes) y `C = 0.5` (regresión logística, `max_iter = 1000`). Esos valores quedan fijos en el entrenamiento; el notebook no se vuelve a ejecutar para producir el modelo.

La línea base, en un test balanceado, es predecir siempre la misma clase: accuracy 0,333 y F1-macro 0,167.

### Resultados en prueba (12 000 reseñas)

| Modelo | Accuracy | F1-macro | F1 negativo | F1 neutro | F1 positivo |
|---|---|---|---|---|---|
| Línea base (clase fija) | 0,333 | 0,167 | – | – | – |
| Naive Bayes | 0,697 | 0,694 | 0,758 | 0,568 | 0,756 |
| **Regresión logística** (producción) | **0,708** | **0,705** | **0,772** | **0,580** | **0,762** |

**Modelo en producción: regresión logística.** Supera a Naive Bayes en accuracy, en F1-macro y en las tres clases, y queda más de 0,5 de F1-macro por encima de la línea base. Además entrega probabilidades por clase, que la aplicación muestra.

### Matriz de confusión (regresión logística)

Cada fila suma 4 000, el tamaño de la clase en prueba.

| Real ↓ / Predicha → | negativo | neutro | positivo |
|---|---|---|---|
| negativo | 3 157 | 678 | 165 |
| neutro | 869 | 2 200 | 931 |
| positivo | 150 | 713 | 3 137 |

### Errores (notebook, sección 3.5)

- **El neutro es la clase difícil** (F1 0,580): de 4 000 reseñas neutras, el modelo acierta 2 200 y reparte el resto hacia negativo (869) y positivo (931). El texto de 3★ mezcla elogio y queja («rico pero caro»).
- **Los extremos se separan bien.** Solo 165 negativas se predicen como positivas y 150 positivas como negativas.
- Errores típicos: ironía, reseñas que elogian la comida y critican el servicio, y estrellas que no coinciden con el texto (ruido de la etiqueta).

## 5. Arquitectura

```
data/raw/reviews.csv
   │  src/data/prepare.py     limpieza, etiquetas, balanceo, split 80/20
   ▼
data/processed/train.csv, test.csv, stats.json
   │  src/models/train.py     NLP → TF-IDF → Naive Bayes y regresión logística
   ▼
models/modelo.joblib         pipeline: prep → TF-IDF → regresión logística
models/metricas.json         métricas de test y la configuración usada
   │  api/ (FastAPI)          carga el pipeline; no reimplementa el NLP
   ▼
frontend/ (React + Vite)     consume la API
```

**Reproducibilidad.** Desde la raíz, `python -m src.models.train` va del CSV crudo al modelo. `python -m src.models.train --sin-preparar` reentrena sobre `data/processed/` sin repetir la limpieza. Versiones fijas en `requirements.txt` y semilla 42 en el split y en la regresión logística. El `.joblib` y `metricas.json` se regeneran con ese comando y no se versionan.

**Backend (FastAPI).** Se eligió porque valida el cuerpo con Pydantic y publica la documentación en `/docs`.

| Método y ruta | Función |
|---|---|
| `GET /health` | Estado de la API y si el modelo cargado es el entrenado |
| `POST /predict` | Clase y probabilidades de una reseña |
| `POST /predict/batch` | Hasta 500 reseñas en un `.txt` o `.csv` (UTF-8), con resumen por clase |
| `POST /preprocess` | Normalización, tokens y stop words eliminadas |
| `GET /metrics` | Accuracy, F1, matriz de confusión y configuración |
| `GET /restaurantes` | Catálogo, con filtro por nombre, tipo de comida y distrito |

Validación en español: texto vacío, texto sin palabras, más de 2 000 caracteres y archivo con formato incorrecto (respuesta 422 o 400). Hay 16 pruebas con pytest en `tests/test_api.py`. Si no existe `models/modelo.joblib`, la API arranca con un modelo de demostración por palabras clave para poder desarrollar la interfaz; en producción se usa el pipeline entrenado.

El catálogo (`src/data/restaurantes.py`) no lista los 9 871 locales. Toma los que más reseñas escritas tienen y una muestra de cada uno, y la API les asigna el sentimiento al iniciar.

**Frontend (React 19 + Vite).** Cuatro pestañas: analizar una reseña (probabilidades y pasos de preprocesamiento), subir un archivo, ver restaurantes con el sentimiento de sus reseñas, y el panel de métricas (comparación de modelos y matriz de confusión, con Recharts).

## 6. Limitaciones y trabajo futuro

- El modelo dice si la reseña es negativa, neutra o positiva. No dice **de qué** se queja el cliente. Un análisis por aspectos (comida, precio, atención) cubriría ese hueco.
- El neutro sigue siendo la clase de menor F1 (0,580). Un modelo de lenguaje en español, por ejemplo BETO, es el candidato natural para subir ese número.
- La etiqueta sale de las estrellas. Cuando el texto y las estrellas no coinciden, el error es de la etiqueta, no solo del clasificador.

## 7. Conclusiones

Con limpieza previa al split, stemming, unigramas y bigramas, y TF-IDF, la regresión logística llega a **0,708 de accuracy y 0,705 de F1-macro** en 12 000 reseñas balanceadas, por encima de Naive Bayes y de la línea base. El sistema distingue bien negativo de positivo. El error se concentra en el neutro, que es mixto por construcción de la etiqueta. El mismo pipeline queda servido por una API y una aplicación web, y se puede volver a entrenar con un solo comando.

## 8. Uso de inteligencia artificial

Se usó un asistente de IA (Claude, de Anthropic) para planificar el proyecto, revisar el dataset, generar y revisar código (preparación de datos, API, frontend y entrenamiento), redactar documentación y explicar conceptos. El grupo revisó, probó y adaptó todo el código. Cada integrante puede explicar y modificar su parte.


