// Funciones para llamar a la API (el proxy de Vite redirige /api a FastAPI)

// Convierte la respuesta en JSON o lanza un error con el mensaje de la API
async function procesar(respuesta) {
  const datos = await respuesta.json()
  if (!respuesta.ok) {
    const mensaje = datos.detalle?.[0]?.error ?? datos.detail ?? 'Error en la API'
    throw new Error(mensaje)
  }
  return datos
}

// POST con un JSON {texto}
function enviarTexto(ruta, texto) {
  return fetch(`/api${ruta}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texto }),
  }).then(procesar)
}

export const predecir = (texto) => enviarTexto('/predict', texto)
export const preprocesar = (texto) => enviarTexto('/preprocess', texto)
export const obtenerMetricas = () => fetch('/api/metrics').then(procesar)
export const obtenerEstado = () => fetch('/api/health').then(procesar)

// Sube un archivo .txt o .csv a /predict/batch
export function predecirArchivo(archivo) {
  const formulario = new FormData()
  formulario.append('archivo', archivo)
  return fetch('/api/predict/batch', { method: 'POST', body: formulario }).then(procesar)
}

// Restaurantes: listado (con filtros), distritos y detalle con reseñas
export function obtenerRestaurantes(buscar = '', distrito = '') {
  const params = new URLSearchParams({ buscar, distrito })
  return fetch(`/api/restaurantes?${params}`).then(procesar)
}
export const obtenerDistritos = () => fetch('/api/restaurantes/distritos').then(procesar)
export const obtenerRestaurante = (id) => fetch(`/api/restaurantes/${id}`).then(procesar)
