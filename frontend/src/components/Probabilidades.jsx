// Barras horizontales con la probabilidad de cada clase
export const COLORES = { negativo: '#d64545', neutro: '#8a8f98', positivo: '#2f9e5b' }

export default function Probabilidades({ probabilidades }) {
  return (
    <div className="probabilidades">
      {Object.entries(probabilidades).map(([clase, valor]) => (
        <div key={clase} className="fila-prob">
          <span className="etiqueta">{clase}</span>
          <div className="barra">
            <div style={{ width: `${valor * 100}%`, background: COLORES[clase] }} />
          </div>
          <span className="valor">{(valor * 100).toFixed(1)} %</span>
        </div>
      ))}
    </div>
  )
}
