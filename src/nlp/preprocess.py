"""
Preprocesamiento de texto para reseñas en español.
"""
import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer

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


STEMMER = SnowballStemmer("spanish")


def aplicar_stemming(tokens: list[str]) -> list[str]:
    return [STEMMER.stem(t) for t in tokens]


_NLP = None


def get_nlp():
    """Carga spaCy en español una sola vez (sin parser ni NER)."""
    global _NLP
    if _NLP is None:
        import spacy

        try:
            _NLP = spacy.load("es_core_news_sm", disable=["parser", "ner"])
        except OSError as e:
            raise OSError(
                "Falta el modelo spaCy 'es_core_news_sm'. "
                "Instálalo con: python -m spacy download es_core_news_sm"
            ) from e
    return _NLP


def aplicar_lemmatizacion(tokens: list[str], nlp=None) -> list[str]:
    if not tokens:
        return []
    nlp = nlp or get_nlp()
    return [t.lemma_ for t in nlp(" ".join(tokens)) if not t.is_space]


def generar_bigramas(tokens: list[str]) -> list[str]:
    return ["_".join(par) for par in ngrams(tokens, 2)]


def aplicar_unigramas(tokens: list[str]) -> list[str]:
    return tokens


def aplicar_unigramas_bigramas(tokens: list[str]) -> list[str]:
    return tokens + generar_bigramas(tokens)


def tfidf(min_df, max_df, max_features=None):
    """TF-IDF con unigramas: los bigramas ya vienen unidos con '_' en el texto.
    min_df, max_df (y max_features) se reciben desde quien llama (train/main).
    """
    return TfidfVectorizer(
        ngram_range=(1, 1),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )


def preparar_documento(texto: str) -> str:
    """Normalizar → tokenizar → stop words → stemming → uni+bigramas."""
    tokens = aplicar_stemming(quitar_stop_words(tokenizar(normalizar(texto))))
    return " ".join(aplicar_unigramas_bigramas(tokens))

# Aplica preparar_documento a una lista de textos (se usa dentro del pipeline del modelo)
def preparar_documentos(textos) -> list[str]:
    return [preparar_documento(t) for t in textos]




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
        sin_stop = quitar_stop_words(tokens)
        print(f"Sin stop:    {sin_stop}")
        stem = aplicar_stemming(sin_stop)
        print(f"Stemming:    {stem}")
        print(f"Uni+bigramas: {' '.join(aplicar_unigramas_bigramas(stem))}\n")