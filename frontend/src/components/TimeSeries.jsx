import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import './ChartCard.css'

const LINE_CFG = [
  { key: 'Pozytywny', color: '#22c55e' },
  { key: 'Neutralny', color: '#6366f1' },
  { key: 'Negatywny', color: '#ef4444' },
]

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '0.3rem' }}>
      <span style={{ color: 'var(--subtext)', fontSize: '0.78rem', marginBottom: '0.15rem' }}>{label}</span>
      {payload.map(p => (
        <div key={p.dataKey} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span className="tooltip-dot" style={{ background: p.color }} />
          <span className="tooltip-name">{p.dataKey}:</span>
          <span className="tooltip-val">{p.value}</span>
        </div>
      ))}
    </div>
  )
}

export default function TimeSeries({ timeSeries }) {
  if (!timeSeries?.length) {
    return (
      <div className="chart-card">
        <h3 className="chart-title">Sentyment w czasie</h3>
        <p style={{ color: 'var(--muted)', fontSize: '0.85rem', textAlign: 'center', padding: '2rem 0' }}>
          Brak danych chronologicznych
        </p>
      </div>
    )
  }

  return (
    <div className="chart-card">
      <h3 className="chart-title">Sentyment w czasie</h3>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={timeSeries} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e3054" />
          <XAxis
            dataKey="month"
            tick={{ fill: '#7a90b8', fontSize: 11 }}
            axisLine={{ stroke: '#1e3054' }}
            tickLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fill: '#7a90b8', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            formatter={val => <span style={{ color: '#7a90b8', fontSize: 12 }}>{val}</span>}
            iconType="circle"
            iconSize={8}
          />
          {LINE_CFG.map(({ key, color }) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={color}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 0 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
