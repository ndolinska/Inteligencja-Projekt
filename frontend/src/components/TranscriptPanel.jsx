import './TranscriptPanel.css'

const SENTIMENT_META = {
  Pozytywny: { color: '#4ade80', emoji: '😊', label: 'pozytywny' },
  Neutralny:  { color: '#818cf8', emoji: '😐', label: 'neutralny' },
  Negatywny: { color: '#f87171', emoji: '😠', label: 'negatywny' },
}

const COLORS = {
  Pozytywny: '#4ade80',
  Neutralny:  '#818cf8',
  Negatywny: '#f87171',
}

function MiniBar({ label, pct, color }) {
  return (
    <div className="tp-bar-row">
      <span className="tp-bar-label">{label}</span>
      <div className="tp-bar-track">
        <div className="tp-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="tp-bar-pct">{pct}%</span>
    </div>
  )
}

function CompareRow({ label, transcriptPct, commentsPct, color }) {
  return (
    <div className="tp-compare-row">
      <span className="tp-compare-label">{label}</span>
      <div className="tp-compare-bars">
        <div className="tp-compare-side tp-compare-side--left">
          <div
            className="tp-compare-fill tp-compare-fill--left"
            style={{ width: `${transcriptPct}%`, background: color }}
          />
          <span className="tp-compare-val">{transcriptPct}%</span>
        </div>
        <div className="tp-compare-divider" />
        <div className="tp-compare-side tp-compare-side--right">
          <div
            className="tp-compare-fill tp-compare-fill--right"
            style={{ width: `${commentsPct}%`, background: color }}
          />
          <span className="tp-compare-val">{commentsPct}%</span>
        </div>
      </div>
    </div>
  )
}

export default function TranscriptPanel({ transcript, comments }) {
  // transcript = { available, dominant, distribution, percentages, avg_score, chunks_analyzed, word_count }
  // comments   = { distribution, percentages }

  if (!transcript?.available) {
    return (
      <div className="chart-card tp-unavailable">
        <h3 className="chart-title">Transkrypcja vs Komentarze</h3>
        <div className="tp-unavail-body">
          <span className="tp-unavail-icon">📄</span>
          <p>Transkrypcja niedostępna dla tego wideo.</p>
          <p className="tp-unavail-sub">
            Napisy mogą być wyłączone lub wideo nie ma generowanych napisów.
          </p>
        </div>
      </div>
    )
  }

  const meta = SENTIMENT_META[transcript.dominant] ?? SENTIMENT_META['Neutralny']
  const sentiments = ['Pozytywny', 'Neutralny', 'Negatywny']

  return (
    <div className="chart-card tp-root">
      <h3 className="chart-title">Transkrypcja vs Komentarze</h3>

      {/* Górna sekcja: karta transkrypcji + karta komentarzy */}
      <div className="tp-top-grid">

        {/* Transkrypcja */}
        <div className="tp-side-card">
          <div className="tp-side-header" style={{ borderColor: meta.color }}>
            <span className="tp-side-emoji">{meta.emoji}</span>
            <div>
              <div className="tp-side-title">Wideo — ton treści</div>
              <div className="tp-side-dominant" style={{ color: meta.color }}>
                {meta.label}
              </div>
            </div>
          </div>
          <div className="tp-side-stats">
            <span>{transcript.word_count?.toLocaleString()} słów</span>
            <span>·</span>
            <span>{transcript.chunks_analyzed} fragmentów</span>
          </div>
          <div className="tp-bars">
            {sentiments.map(s => (
              <MiniBar
                key={s}
                label={s}
                pct={transcript.percentages?.[s] ?? 0}
                color={COLORS[s]}
              />
            ))}
          </div>
        </div>

        {/* Komentarze */}
        <div className="tp-side-card">
          {(() => {
            const domComments = sentiments.reduce(
              (best, s) => (comments.percentages?.[s] ?? 0) > (comments.percentages?.[best] ?? 0) ? s : best,
              'Neutralny'
            )
            const metaC = SENTIMENT_META[domComments]
            return (
              <>
                <div className="tp-side-header" style={{ borderColor: metaC.color }}>
                  <span className="tp-side-emoji">{metaC.emoji}</span>
                  <div>
                    <div className="tp-side-title">Komentarze — reakcja widzów</div>
                    <div className="tp-side-dominant" style={{ color: metaC.color }}>
                      {metaC.label}
                    </div>
                  </div>
                </div>
                <div className="tp-side-stats">
                  <span>
                    {comments.distribution
                      ? Object.values(comments.distribution).reduce((a, b) => a + b, 0).toLocaleString()
                      : '—'} komentarzy
                  </span>
                </div>
                <div className="tp-bars">
                  {sentiments.map(s => (
                    <MiniBar
                      key={s}
                      label={s}
                      pct={comments.percentages?.[s] ?? 0}
                      color={COLORS[s]}
                    />
                  ))}
                </div>
              </>
            )
          })()}
        </div>
      </div>

      {/* Porównanie back-to-back */}
      <div className="tp-compare-section">
        <div className="tp-compare-header">
          <span className="tp-compare-col-label">Wideo</span>
          <span className="tp-compare-title">Porównanie</span>
          <span className="tp-compare-col-label">Komentarze</span>
        </div>
        {sentiments.map(s => (
          <CompareRow
            key={s}
            label={s}
            transcriptPct={transcript.percentages?.[s] ?? 0}
            commentsPct={comments.percentages?.[s] ?? 0}
            color={COLORS[s]}
          />
        ))}
      </div>

      {/* Wniosek */}
      <div className="tp-insight">
        {(() => {
          const domT = transcript.dominant
          const domC = sentiments.reduce(
            (best, s) => (comments.percentages?.[s] ?? 0) > (comments.percentages?.[best] ?? 0) ? s : best,
            'Neutralny'
          )
          if (domT === domC) {
            return `✅ Ton wideo i reakcja widzów są zgodne — oba dominująco ${SENTIMENT_META[domT].label}.`
          }
          return `⚡ Rozbieżność: wideo ma ton ${SENTIMENT_META[domT].label}, ale komentarze są dominująco ${SENTIMENT_META[domC].label}.`
        })()}
      </div>
    </div>
  )
}
