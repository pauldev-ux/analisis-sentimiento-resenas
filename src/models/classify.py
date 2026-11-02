"""
Clasificadores para análisis de sentimiento.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

SEMILLA = 42


def naive_bayes():
    return MultinomialNB()


def regresion_logistica(max_iter=1000, random_state=SEMILLA):
    return LogisticRegression(max_iter=max_iter, random_state=random_state)
