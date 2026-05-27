import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import './ChartCard.css'

const BAR_COLORS = [
  '#6366f1', '#22c55e', '#f59e0b', '#ef4444',
  '#06b6d4', '#a78bfa', '#fb7185', '#34d399',
]

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <span className="tooltip-dot" style={{ background: payload[0].fill }} />
      <span className="tooltip-name">{label}:</span>
      <span className="tooltip-val">{payload[0].value.toLocaleString('pl-PL')} komentarzy</span>
    </div>
  )
}

export default function LanguageChart({ languages }) {
  if (!languages?.length) return null

  const data = languages.slice(0, 10)

  return (
    <div className="chart-card">
      <h3 className="chart-title">Języki komentarzy</h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e3054" vertical={false} />
          <XAxis
            dataKey="language"
            tick={{ fill: '#7a90b8', fontSize: 12 }}
            axisLine={{ stroke: '#1e3054' }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: '#7a90b8', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(99,102,241,0.08)' }} />
          <Bar dataKey="count" radius={[6, 6, 0, 0]}>
            {data.map((_, i) => (
              <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
