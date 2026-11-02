"""
Clasificadores para análisis de sentimiento.

Los hiperparámetros se reciben como argumentos (sin valores escondidos): el notebook los
ajusta con validación cruzada y train.py usa los ganadores como constantes.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

SEMILLA = 42


def naive_bayes(alpha=1.0):
    """Naive Bayes multinomial. alpha: suavizado de Laplace (más alto = más suavizado)."""
    return MultinomialNB(alpha=alpha)


def regresion_logistica(C=1.0, max_iter=1000, random_state=SEMILLA):
    """Regresión logística. C: inverso de la regularización (más bajo = más regularización)."""
    return LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
