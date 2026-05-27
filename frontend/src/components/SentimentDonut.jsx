import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import './ChartCard.css'

const COLORS = {
  'Pozytywny': '#22c55e',
  'Neutralny': '#6366f1',
  'Negatywny': '#ef4444',
}

const RADIAN = Math.PI / 180

function CustomLabel({ cx, cy, midAngle, innerRadius, outerRadius, percent }) {
  if (percent < 0.04) return null
  const radius = innerRadius + (outerRadius - innerRadius) * 0.55
  const x = cx + radius * Math.cos(-midAngle * RADIAN)
  const y = cy + radius * Math.sin(-midAngle * RADIAN)

  return (
    <text x={x} y={y} fill="#fff" textAnchor="middle" dominantBaseline="central"
      fontSize={13} fontWeight={600}>
      {`${(percent * 100).toFixed(1)}%`}
    </text>
  )
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const { name, value } = payload[0]
  return (
    <div className="chart-tooltip">
      <span className="tooltip-dot" style={{ background: COLORS[name] }} />
      <span className="tooltip-name">{name}</span>
      <span className="tooltip-val">{value.toLocaleString('pl-PL')} komentarzy</span>
    </div>
  )
}

export default function SentimentDonut({ distribution }) {
  const data = Object.entries(distribution).map(([name, value]) => ({ name, value }))

  return (
    <div className="chart-card">
      <h3 className="chart-title">Rozkład sentymentu</h3>
      <ResponsiveContainer width="100%" height={260}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={68}
            outerRadius={105}
            paddingAngle={3}
            dataKey="value"
            labelLine={false}
            label={CustomLabel}
          >
            {data.map(entry => (
              <Cell key={entry.name} fill={COLORS[entry.name] ?? '#94a3b8'} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
          <Legend
            formatter={val => <span style={{ color: '#7a90b8', fontSize: 13 }}>{val}</span>}
            iconType="circle"
            iconSize={9}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}
