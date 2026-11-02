import { useEffect, useState } from 'react'
import { obtenerEstado } from './api'

export default function App() {
  const [estado, setEstado] = useState(null)

  // Al abrir la app, consulta /health para saber si la API responde y qué modelo usa
  useEffect(() => {
    obtenerEstado().then(setEstado).catch(() => setEstado({ status: 'error' }))
  }, [])

  return (
    <div className="app">
      <header>
        <h1>Análisis de sentimiento de reseñas</h1>
        <p className="subtitulo">Restaurantes de Lima · negativo, neutro o positivo</p>
        <EstadoApi estado={estado} />
      </header>
    </div>
  )
}

// Indicador del estado de la API y del tipo de modelo cargado
function EstadoApi({ estado }) {
  if (!estado) return <span className="estado">Conectando…</span>
  if (estado.status !== 'ok') return <span className="estado error">API sin conexión</span>
  return (
    <span className="estado ok">
      API conectada · modelo {estado.modelo === 'demo' ? 'de prueba' : 'entrenado'}
    </span>
  )
}