import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { obtenerMetricas } from '../api'

// Nombres legibles de los modelos
const NOMBRES = { naive_bayes: 'Naive Bayes', regresion_logistica: 'Regresión Logística' }
const COLORES_MODELO = ['#4c6ef5', '#f08c00']

export default function Metricas() {
  const [datos, setDatos] = useState(null)
  const [error, setError] = useState('')

  // Al abrir la pestaña, pide las métricas a /metrics
  useEffect(() => {
    obtenerMetricas().then(setDatos).catch((e) => setError(e.message))
  }, [])

  if (error) return <p className="mensaje-error">⚠ {error}</p>
  if (!datos) return <p>Cargando métricas…</p>

  const modelos = Object.keys(datos.modelos)
  // Una fila por métrica, una columna por modelo (formato que pide Recharts)
  const comparacion = [
    { metrica: 'Accuracy', ...Object.fromEntries(modelos.map((m) => [m, datos.modelos[m].accuracy])) },
    { metrica: 'F1-macro', ...Object.fromEntries(modelos.map((m) => [m, datos.modelos[m].f1_macro])) },
    ...datos.clases.map((c) => ({
      metrica: `F1 ${c}`,
      ...Object.fromEntries(modelos.map((m) => [m, datos.modelos[m].por_clase[c].f1])),
    })),
  ]

  return (
    <section>
      {datos.origen === 'demo' && (
        <p className="aviso">Métricas de ejemplo: todavía no hay un modelo entrenado.</p>
      )}
      <div className="tarjeta">
        <h2>Comparación de modelos</h2>
        <p>En producción: <b>{NOMBRES[datos.modelo_en_produccion] ?? datos.modelo_en_produccion}</b></p>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={comparacion}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="metrica" />
            <YAxis domain={[0, 1]} />
            <Tooltip formatter={(v) => v.toFixed(3)} />
            <Legend />
            {modelos.map((m, i) => (
              <Bar key={m} dataKey={m} name={NOMBRES[m] ?? m} fill={COLORES_MODELO[i]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="matrices">
        {modelos.map((m) => (
          <MatrizConfusion key={m} titulo={NOMBRES[m] ?? m} clases={datos.clases} matriz={datos.modelos[m].matriz_confusion} />
        ))}
      </div>
    </section>
  )
}

// Matriz de confusión: filas = clase real, columnas = clase predicha
function MatrizConfusion({ titulo, clases, matriz }) {
  const maximo = Math.max(...matriz.flat())
  return (
    <div className="tarjeta">
      <h3>Matriz de confusión · {titulo}</h3>
      <table className="matriz">
        <thead>
          <tr><th>Real ↓ / Predicha →</th>{clases.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {matriz.map((fila, i) => (
            <tr key={clases[i]}>
              <th>{clases[i]}</th>
              {fila.map((valor, j) => (
                // Más oscuro = más casos; la diagonal son los aciertos
                <td key={j} style={{ background: `rgba(76, 110, 245, ${valor / maximo})`, color: valor / maximo > 0.5 ? '#fff' : '#222' }}>
                  {valor}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
