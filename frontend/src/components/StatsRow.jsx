import './StatsRow.css'

function StatCard({ label, value, sub, accent }) {
  return (
    <div className={`stat-card ${accent ? `stat-card--${accent}` : ''}`}>
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
      {sub && <span className="stat-sub">{sub}</span>}
    </div>
  )
}

export default function StatsRow({ data }) {
  const {
    total_fetched,
    total_analyzed,
    filtered_out,
    processing_time,
    from_cache,
    percentages,
  } = data

  const positivePct = percentages?.['Pozytywny'] ?? 0
  const negativePct = percentages?.['Negatywny'] ?? 0

  return (
    <div className="stats-row">
      <StatCard
        label="Pobrane komentarze"
        value={total_fetched.toLocaleString('pl-PL')}
      />
      <StatCard
        label="Przeanalizowane"
        value={total_analyzed.toLocaleString('pl-PL')}
        sub={filtered_out > 0 ? `Odfiltrowano: ${filtered_out}` : null}
      />
      <StatCard
        label="Pozytywne"
        value={`${positivePct.toFixed(1)} %`}
        accent="positive"
      />
      <StatCard
        label="Negatywne"
        value={`${negativePct.toFixed(1)} %`}
        accent="negative"
      />
      <StatCard
        label="Czas analizy"
        value={`${processing_time.toFixed(1)} s`}
        sub={from_cache ? '⚡ z cache' : null}
      />
    </div>
  )
}
