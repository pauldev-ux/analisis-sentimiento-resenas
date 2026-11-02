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


# Verifica que /predict/batch clasifique un .txt con una reseña por línea
def test_batch_txt():
    contenido = "Excelente comida\nPésimo servicio\n\n10/10\n".encode("utf-8")
    r = client.post("/predict/batch", files={"archivo": ("resenas.txt", contenido, "text/plain")})
    assert r.status_code == 200
    datos = r.json()
    assert datos["total"] == 2          # la línea vacía se ignora
    assert datos["descartadas"] == 1    # "10/10" no tiene palabras


# Verifica que /predict/batch lea un .csv usando la columna "texto"
def test_batch_csv():
    contenido = "id,texto\n1,Muy rico todo\n2,No me gustó\n".encode("utf-8")
    r = client.post("/predict/batch", files={"archivo": ("resenas.csv", contenido, "text/csv")})
    assert r.status_code == 200
    assert r.json()["total"] == 2


# Verifica que se rechacen formatos no soportados
def test_batch_rechaza_formato():
    r = client.post("/predict/batch", files={"archivo": ("resenas.pdf", b"hola", "application/pdf")})
    assert r.status_code == 400


# Verifica que /metrics devuelva las métricas de ambos modelos con su matriz de confusión
def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    datos = r.json()
    assert datos["modelo_en_produccion"] in datos["modelos"]
    for metricas in datos["modelos"].values():
        assert 0 <= metricas["f1_macro"] <= 1
        assert len(metricas["matriz_confusion"]) == 3



# Verifica que /restaurantes liste restaurantes con su resumen de sentimiento
def test_listar_restaurantes():
    r = client.get("/restaurantes")
    assert r.status_code == 200
    datos = r.json()
    assert datos["total"] > 0
    primero = datos["restaurantes"][0]
    assert "resenas" not in primero
    assert abs(sum(primero["sentimiento"].values()) - 1) < 0.01


# Verifica el filtro por distrito
def test_filtrar_por_distrito():
    distrito = client.get("/restaurantes/distritos").json()[0]
    datos = client.get("/restaurantes", params={"distrito": distrito}).json()
    assert all(r["distrito"] == distrito for r in datos["restaurantes"])


# Verifica que el detalle traiga las reseñas con su sentimiento
def test_detalle_restaurante():
    id_restaurante = client.get("/restaurantes").json()["restaurantes"][0]["id"]
    datos = client.get(f"/restaurantes/{id_restaurante}").json()
    assert len(datos["resenas"]) > 0
    assert datos["resenas"][0]["sentimiento"] in ["negativo", "neutro", "positivo"]


# Verifica que un id inexistente devuelva 404
def test_restaurante_inexistente():
    assert client.get("/restaurantes/1").status_code == 404