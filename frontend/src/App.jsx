import { useEffect, useState } from 'react'
import { obtenerEstado } from './api'
import AnalisisTexto from './components/AnalisisTexto'

// Pestañas de la aplicación
const PESTANAS = [
  { id: 'texto', nombre: 'Analizar reseña', componente: AnalisisTexto },
]

export default function App() {
  const [activa, setActiva] = useState('texto')
  const [estado, setEstado] = useState(null)

  // Al abrir la app, consulta /health para saber si la API responde y qué modelo usa
  useEffect(() => {
    obtenerEstado().then(setEstado).catch(() => setEstado({ status: 'error' }))
  }, [])

  // Componente de la pestaña seleccionada
  const Componente = PESTANAS.find((p) => p.id === activa).componente

  return (
    <div className="app">
      <header>
        <h1>Análisis de sentimiento de reseñas</h1>
        <p className="subtitulo">Restaurantes de Lima · negativo, neutro o positivo</p>
        <EstadoApi estado={estado} />
      </header>

      <nav className="pestanas">
        {PESTANAS.map((p) => (
          <button key={p.id} className={p.id === activa ? 'activa' : ''} onClick={() => setActiva(p.id)}>
            {p.nombre}
          </button>
        ))}
      </nav>

      <main>
        <Componente />
      </main>
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