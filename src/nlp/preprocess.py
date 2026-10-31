"""
Preprocesamiento de texto para reseñas en español.
"""
import re
import unicodedata

import nltk
from nltk.corpus import stopwords

URL = re.compile(r"https?://\S+|www\.\S+")
NO_LETRAS = re.compile(r"[^a-zñ\s]")        # todo lo que no sea letra o espacio
REPETIDAS = re.compile(r"(.)\1{2,}")         # 3 o más letras iguales seguidas
ESPACIOS = re.compile(r"\s+")


def quitar_tildes(texto: str) -> str:
    """Quita tildes y diéresis pero conserva la ñ (año ≠ ano)."""
    texto = texto.replace("ñ", "\0")
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )
    return sin_tildes.replace("\0", "ñ")


def normalizar(texto: str) -> str:
    texto = texto.lower()
    texto = URL.sub(" ", texto)
    texto = quitar_tildes(texto)
    texto = NO_LETRAS.sub(" ", texto)          # números, signos, emojis
    texto = REPETIDAS.sub(r"\1\1", texto)      # "buenazooo" -> "buenazoo"
    return ESPACIOS.sub(" ", texto).strip()


def tokenizar(texto: str) -> list[str]:
    """Separa el texto normalizado en palabras."""
    return texto.split()


# Palabras que NLTK trae como stop words pero que cambian el sentimiento
CONSERVAR = {"no", "ni", "sin", "nada", "nunca", "tampoco", "muy", "poco", "mucho", "mas", "pero"}


def cargar_stop_words() -> set[str]:
    """Stop words de NLTK en español, normalizadas y sin las de CONSERVAR."""
    try:
        lista = stopwords.words("spanish")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        lista = stopwords.words("spanish")
    return {normalizar(p) for p in lista} - CONSERVAR


STOP_WORDS = cargar_stop_words()


def quitar_stop_words(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOP_WORDS]


if __name__ == "__main__":
    ejemplos = [
        "¡¡La comida estuvo RIQUÍSIMAAA!! 10/10, volveré ",
        "No me gustó, el año pasado era mejor. Ver: https://tripadvisor.com",
        "Atención pésima... esperamos 45 min por un pingüino de helado",
    ]
    for t in ejemplos:
        n = normalizar(t)
        tokens = tokenizar(n)
        print(f"Original:    {t}\nNormalizado: {n}\nTokens:      {tokens}")
        print(f"Sin stop:    {quitar_stop_words(tokens)}\n")