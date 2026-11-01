from fastapi.testclient import TestClient

from api.main import app

# Simula peticiones HTTP sin levantar el servidor
client = TestClient(app)


# Verifica que /health responda 200 con status ok
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# Verifica que /preprocess normalice, conserve el "no" y quite stop words
def test_preprocess_muestra_cada_paso():
    r = client.post("/preprocess", json={"texto": "No me gustó, el año pasado era MEJOR!!"})
    assert r.status_code == 200
    datos = r.json()
    assert datos["normalizado"] == "no me gusto el año pasado era mejor"
    assert "no" in datos["sin_stop_words"]
    assert "el" in datos["stop_words_eliminadas"]


# Verifica que se rechace un texto vacío o solo con espacios
def test_rechaza_texto_vacio():
    r = client.post("/preprocess", json={"texto": "   "})
    assert r.status_code == 422
    assert r.json()["detalle"][0]["error"] == "el texto no puede estar vacío"


# Verifica que se rechace un texto sin palabras (solo números, signos o emojis)
def test_rechaza_texto_sin_palabras():
    r = client.post("/preprocess", json={"texto": "10/10 !!! 😍"})
    assert r.status_code == 422


# Verifica que se rechace un texto de más de 2000 caracteres
def test_rechaza_texto_demasiado_largo():
    r = client.post("/preprocess", json={"texto": "a" * 2001})
    assert r.status_code == 422
    assert "máximo" in r.json()["detalle"][0]["error"]


# Verifica que se rechace un tipo incorrecto o un campo faltante
def test_rechaza_formato_incorrecto():
    assert client.post("/preprocess", json={"texto": 123}).status_code == 422
    assert client.post("/preprocess", json={"mensaje": "hola"}).status_code == 422

# Verifica que /predict devuelva una clase válida y probabilidades que suman 1
def test_predict_devuelve_clase_y_probabilidad():
    r = client.post("/predict", json={"texto": "La comida estuvo excelente"})
    assert r.status_code == 200
    datos = r.json()
    assert datos["clase"] in ["negativo", "neutro", "positivo"]
    assert 0 <= datos["probabilidad"] <= 1
    assert abs(sum(datos["probabilidades"].values()) - 1) < 0.01


# Verifica que /predict también valide la entrada
def test_predict_rechaza_texto_vacio():
    assert client.post("/predict", json={"texto": ""}).status_code == 422
