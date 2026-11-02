import { useState } from 'react'
import { predecirArchivo } from '../api'
import { COLORES } from './Probabilidades'

export default function AnalisisArchivo() {
  const [archivo, setArchivo] = useState(null)
  const [datos, setDatos] = useState(null)
  const [error, setError] = useState('')
  const [cargando, setCargando] = useState(false)

  // Sube el archivo a /predict/batch
  async function analizar() {
    setCargando(true)
    setError('')
    try {
      setDatos(await predecirArchivo(archivo))
    } catch (e) {
      setError(e.message)
      setDatos(null)
    } finally {
      setCargando(false)
    }
  }

  return (
    <section>
      <p className="ayuda">
        Sube un <b>.txt</b> (una reseña por línea) o un <b>.csv</b> (columna <code>texto</code>). Máximo 500 reseñas.
      </p>
      <div className="acciones">
        <input type="file" accept=".txt,.csv" onChange={(e) => setArchivo(e.target.files[0] ?? null)} />
        <button className="primario" onClick={analizar} disabled={!archivo || cargando}>
          {cargando ? 'Analizando…' : 'Analizar archivo'}
        </button>
      </div>

      {error && <p className="mensaje-error">⚠ {error}</p>}

      {datos && (
        <div className="tarjeta">
          <h2>{datos.total} reseñas analizadas{datos.descartadas > 0 && ` · ${datos.descartadas} descartadas`}</h2>
          <Resumen resumen={datos.resumen} total={datos.total} />
          <table>
            <thead><tr><th>Reseña</th><th>Clase</th><th>Confianza</th></tr></thead>
            <tbody>
              {datos.resultados.map((r, i) => (
                <tr key={i}>
                  <td>{r.texto}</td>
                  <td style={{ color: COLORES[r.clase], fontWeight: 600 }}>{r.clase}</td>
                  <td>{(r.probabilidad * 100).toFixed(1)} %</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

// Barra apilada con el porcentaje de cada clase en el archivo
function Resumen({ resumen, total }) {
  const clases = ['negativo', 'neutro', 'positivo']
  return (
    <>
      <div className="barra-apilada">
        {clases.map((c) => resumen[c] ? (
          <div key={c} style={{ width: `${(resumen[c] / total) * 100}%`, background: COLORES[c] }} />
        ) : null)}
      </div>
      <div className="leyenda">
        {clases.map((c) => (
          <span key={c}><i style={{ background: COLORES[c] }} />{c}: {resumen[c] ?? 0}</span>
        ))}
      </div>
    </>
  )
}
