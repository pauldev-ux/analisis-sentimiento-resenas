import { useState } from 'react'
import { predecir, preprocesar } from '../api'
import Probabilidades, { COLORES } from './Probabilidades'

const EJEMPLOS = [
  'La comida estuvo riquísima y la atención excelente, volveré!!',
  'No me gustó nada, el pollo estaba frío y tardaron muchísimo.',
  'Rico el ceviche pero bastante caro para lo que sirven.',
]

export default function AnalisisTexto() {
  const [texto, setTexto] = useState('')
  const [resultado, setResultado] = useState(null)
  const [pasos, setPasos] = useState(null)
  const [error, setError] = useState('')
  const [cargando, setCargando] = useState(false)

  // Llama a /predict y /preprocess al mismo tiempo
  async function analizar() {
    setCargando(true)
    setError('')
    try {
      const [pred, prep] = await Promise.all([predecir(texto), preprocesar(texto)])
      setResultado(pred)
      setPasos(prep)
    } catch (e) {
      setError(e.message)
      setResultado(null)
      setPasos(null)
    } finally {
      setCargando(false)
    }
  }

  return (
    <section>
      <textarea
        value={texto}
        onChange={(e) => setTexto(e.target.value)}
        placeholder="Escribe una reseña de un restaurante…"
        rows={4}
        maxLength={2000}
      />
      <div className="acciones">
        <button className="primario" onClick={analizar} disabled={cargando || !texto.trim()}>
          {cargando ? 'Analizando…' : 'Analizar'}
        </button>
        <span className="contador">{texto.length} / 2000</span>
      </div>
      <div className="ejemplos">
        Probar con:
        {EJEMPLOS.map((e) => (
          <button key={e} className="chip" onClick={() => setTexto(e)}>{e.slice(0, 32)}…</button>
        ))}
      </div>

      {error && <p className="mensaje-error">⚠ {error}</p>}

      {resultado && (
        <div className="tarjeta">
          <h2>Resultado</h2>
          <p className="clase" style={{ color: COLORES[resultado.clase] }}>
            {resultado.clase.toUpperCase()}
            <span> · {(resultado.probabilidad * 100).toFixed(1)} % de confianza</span>
          </p>
          <Probabilidades probabilidades={resultado.probabilidades} />
        </div>
      )}

      {pasos && <PasoAPaso pasos={pasos} />}
    </section>
  )
}

// Muestra cómo queda el texto en cada paso del pipeline de NLP
function PasoAPaso({ pasos }) {
  return (
    <div className="tarjeta">
      <h2>Preprocesamiento paso a paso</h2>
      <ol className="pasos">
        <li><b>Original</b><p>{pasos.original}</p></li>
        <li><b>Normalizado</b><p>{pasos.normalizado}</p></li>
        <li><b>Tokens</b><Fichas lista={pasos.tokens} /></li>
        <li><b>Stop words eliminadas</b><Fichas lista={pasos.stop_words_eliminadas} tipo="tachada" /></li>
        <li><b>Sin stop words</b><Fichas lista={pasos.sin_stop_words} /></li>
        {pasos.raices && <li><b>Raíces / lemas</b><Fichas lista={pasos.raices} tipo="resaltada" /></li>}
      </ol>
    </div>
  )
}

// Lista de palabras como fichas
function Fichas({ lista, tipo = '' }) {
  if (!lista.length) return <p className="vacio">(ninguna)</p>
  return <div className="fichas">{lista.map((t, i) => <span key={i} className={`ficha ${tipo}`}>{t}</span>)}</div>
}
