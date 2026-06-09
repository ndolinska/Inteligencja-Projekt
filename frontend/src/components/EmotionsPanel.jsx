import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts'
import './EmotionsPanel.css'

const GROUP_COLORS = {
  Pozytywne:    '#4ade80',
  Negatywne:    '#f87171',
  Zaskoczenie:  '#facc15',
  Ambiwalentne: '#818cf8',
  Neutralne:    '#94a3b8',
}

const GROUP_ICONS = {
  Pozytywne:    '😊',
  Negatywne:    '😠',
  Zaskoczenie:  '😮',
  Ambiwalentne: '🤔',
  Neutralne:    '😐',
}

const EMOTION_BAR_COLORS = [
  '#818cf8','#4ade80','#f87171','#facc15','#fb923c',
  '#34d399','#a78bfa','#38bdf8','#f472b6','#e879f9',
]

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const { name, value } = payload[0]
  return (
    <div className="ep-tooltip">
      <span>{name}</span>
      <strong>{value}%</strong>
    </div>
  )
}

function CustomLabel({ cx, cy, midAngle, innerRadius, outerRadius, percent, name }) {
  if (percent < 0.06) return null
  const RADIAN = Math.PI / 180
  const r = innerRadius + (outerRadius - innerRadius) * 0.55
  const x = cx + r * Math.cos(-midAngle * RADIAN)
  const y = cy + r * Math.sin(-midAngle * RADIAN)
  return (
    <text x={x} y={y} fill="#fff" textAnchor="middle" dominantBaseline="central"
      fontSize={11} fontWeight={600}>
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  )
}

export default function EmotionsPanel({ emotionAnalysis }) {
  if (!emotionAnalysis?.available) return null

  const { top_emotions, groups, dominant_group, total_analyzed } = emotionAnalysis

  // Dane do wykresu kołowego makro-grup
  const pieData = Object.entries(groups)
    .map(([name, data]) => ({ name, value: data.percentage, count: data.count }))
    .filter(d => d.value > 0)

  const maxCount = top_emotions?.[0]?.count ?? 1

  return (
    <div className="chart-card ep-root">
      <div className="ep-header">
        <h3 className="chart-title">Analiza emocji — GoEmotions</h3>
        <span className="ep-badge">🧠 {total_analyzed?.toLocaleString()} komentarzy</span>
      </div>

      <div className="ep-grid">

        {/* Lewa kolumna: wykres kołowy makro-grup */}
        <div className="ep-left">
          <p className="ep-section-label">Makro-grupy emocji</p>
          <div className="ep-pie-wrap">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={90}
                  paddingAngle={2}
                  dataKey="value"
                  labelLine={false}
                  label={CustomLabel}
                >
                  {pieData.map((entry) => (
                    <Cell
                      key={entry.name}
                      fill={GROUP_COLORS[entry.name] ?? '#94a3b8'}
                      stroke="transparent"
                    />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>

            {/* Centrum donuta */}
            <div className="ep-donut-center">
              <span className="ep-donut-icon">{GROUP_ICONS[dominant_group]}</span>
              <span className="ep-donut-label">{dominant_group}</span>
            </div>
          </div>

          {/* Legenda */}
          <div className="ep-legend">
            {pieData.map(d => (
              <div key={d.name} className="ep-legend-item">
                <span className="ep-legend-dot" style={{ background: GROUP_COLORS[d.name] }} />
                <span className="ep-legend-name">{GROUP_ICONS[d.name]} {d.name}</span>
                <span className="ep-legend-pct">{d.value}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* Prawa kolumna: top 10 emocji */}
        <div className="ep-right">
          <p className="ep-section-label">Top 10 dominujących emocji</p>
          <div className="ep-bars">
            {top_emotions?.map((em, i) => {
              const pct = Math.round((em.count / maxCount) * 100)
              return (
                <div key={em.emotion} className="ep-bar-row">
                  <div className="ep-bar-meta">
                    <span className="ep-bar-rank">#{i + 1}</span>
                    <span className="ep-bar-name">{em.emotion_pl}</span>
                    <span className="ep-bar-pct">{em.percentage}%</span>
                  </div>
                  <div className="ep-bar-track">
                    <div
                      className="ep-bar-fill"
                      style={{
                        width: `${pct}%`,
                        background: EMOTION_BAR_COLORS[i % EMOTION_BAR_COLORS.length],
                      }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
