
##Preprocesamiento de texto para reseñas en español.

import re
import unicodedata

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




if __name__ == "__main__":
    ejemplos = [
        "¡¡La comida estuvo RIQUÍSIMAAA!! 10/10, volveré",
        "No me gustó, el año pasado era mejor. Ver: https://tripadvisor.com",
        "Atención pésima... esperamos 45 min por un pingüino de helado",
    ]
    for t in ejemplos:
        print(f"Original:    {t}\nNormalizado: {normalizar(t)}\n")

    
    
    n = normalizar(t)
    print(f"Original:    {t}\nNormalizado: {n}\nTokens:      {tokenizar(n)}\n")

