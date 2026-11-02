import { useEffect, useState } from 'react'
import { obtenerDistritos, obtenerRestaurante, obtenerRestaurantes } from '../api'
import { COLORES } from './Probabilidades'

export default function Restaurantes() {
  const [lista, setLista] = useState([])
  const [distritos, setDistritos] = useState([])
  const [buscar, setBuscar] = useState('')
  const [distrito, setDistrito] = useState('')
  const [elegido, setElegido] = useState(null)
  const [error, setError] = useState('')

  // Carga los distritos una sola vez para el filtro
  useEffect(() => {
    obtenerDistritos().then(setDistritos).catch(() => {})
  }, [])

  // Vuelve a pedir la lista cada vez que cambia la búsqueda o el distrito
  useEffect(() => {
    obtenerRestaurantes(buscar, distrito)
      .then((d) => setLista(d.restaurantes))
      .catch((e) => setError(e.message))
  }, [buscar, distrito])

  // Abre el detalle de un restaurante
  async function abrir(id) {
    setElegido(await obtenerRestaurante(id))
  }

  if (elegido) return <Detalle r={elegido} volver={() => setElegido(null)} />

  return (
    <section>
      <div className="filtros">
        <input placeholder="Buscar por nombre o tipo de comida (ej. mariscos)" value={buscar} onChange={(e) => setBuscar(e.target.value)} />
        <select value={distrito} onChange={(e) => setDistrito(e.target.value)}>
          <option value="">Todos los distritos</option>
          {distritos.map((d) => <option key={d}>{d}</option>)}
        </select>
      </div>
      {error && <p className="mensaje-error">⚠ {error}</p>}
      <p className="ayuda">{lista.length} restaurantes</p>

      <div className="grilla">
        {lista.map((r) => (
          <button key={r.id} className="tarjeta restaurante" onClick={() => abrir(r.id)}>
            <h3>{r.nombre}</h3>
            <p className="ayuda">{r.distrito}{r.tipos.length > 0 && ` · ${r.tipos.slice(0, 2).join(', ')}`}</p>
            <p>⭐ {r.estrellas} <span className="ayuda">({r.n_resenas} reseñas)</span></p>
            <BarraSentimiento sentimiento={r.sentimiento} />
          </button>
        ))}
      </div>
    </section>
  )
}

// Barra con el % de reseñas negativas, neutras y positivas según el modelo
function BarraSentimiento({ sentimiento }) {
  return (
    <>
      <div className="barra-apilada">
        {Object.entries(sentimiento).map(([c, v]) => v > 0 && (
          <div key={c} style={{ width: `${v * 100}%`, background: COLORES[c] }} title={`${c}: ${(v * 100).toFixed(0)} %`} />
        ))}
      </div>
      <p className="ayuda">😊 {(sentimiento.positivo * 100).toFixed(0)} % · 😐 {(sentimiento.neutro * 100).toFixed(0)} % · 😞 {(sentimiento.negativo * 100).toFixed(0)} %</p>
    </>
  )
}

// Detalle: datos del restaurante y sus reseñas con el sentimiento de cada una
function Detalle({ r, volver }) {
  return (
    <section>
      <button className="chip" onClick={volver}>← Volver</button>
      <div className="tarjeta">
        <h2>{r.nombre}</h2>
        <p className="ayuda">{r.direccion}</p>
        <p>⭐ {r.estrellas} en {r.plataforma === 'tripadvisor' ? 'TripAdvisor' : 'Google'} · {r.n_resenas} reseñas{r.precio && ` · ${r.precio}`}</p>
        <h3>Lo que dicen las reseñas según nuestro modelo</h3>
        <BarraSentimiento sentimiento={r.sentimiento} />
      </div>
      {r.resenas.map((res, i) => (
        <div key={i} className="tarjeta resena" style={{ borderLeftColor: COLORES[res.sentimiento] }}>
          <p>{res.texto}</p>
          <p className="ayuda">
            {'★'.repeat(res.estrellas)}{'☆'.repeat(5 - res.estrellas)} · {res.fecha} ·{' '}
            <b style={{ color: COLORES[res.sentimiento] }}>{res.sentimiento}</b> ({(res.probabilidad * 100).toFixed(0)} %)
          </p>
        </div>
      ))}
    </section>
  )
}
