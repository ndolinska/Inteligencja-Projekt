import { useState } from 'react'
import './SearchBar.css'

const MODELS = [
  {
    id:    'vader',
    label: 'VADER',
    desc:  'Szybki, słownikowy — wyniki natychmiastowe',
  },
  {
    id:    'xlm-roberta',
    label: 'XLM-RoBERTa',
    desc:  'Transformer wielojęzyczny — wolniej (~1 min), wyższa jakość',
  },
]

export default function SearchBar({ onAnalyze, loading, model, onModelChange }) {
  const [url, setUrl] = useState('')

  function handleSubmit(e) {
    e.preventDefault()
    const trimmed = url.trim()
    if (!trimmed) return
    onAnalyze(trimmed)
  }

  return (
    <div className="search-wrap">
      <form className="search-bar" onSubmit={handleSubmit}>
        <div className="search-input-wrap">
          <span className="search-prefix-icon">▶</span>
          <input
            className="search-input"
            type="text"
            placeholder="Wklej link do filmu YouTube…"
            value={url}
            onChange={e => setUrl(e.target.value)}
            disabled={loading}
            spellCheck={false}
          />
          {url && !loading && (
            <button
              type="button"
              className="search-clear"
              onClick={() => setUrl('')}
              aria-label="Wyczyść"
            >
              ✕
            </button>
          )}
        </div>
        <button
          type="submit"
          className="search-btn"
          disabled={loading || !url.trim()}
        >
          {loading ? (
            <span className="btn-spinner" />
          ) : (
            <>
              <span className="btn-icon">◈</span>
              Analizuj
            </>
          )}
        </button>
      </form>

      {/* Selektor modelu */}
      <div className="model-selector">
        <span className="model-selector-label">Model sentymentu:</span>
        <div className="model-options">
          {MODELS.map(m => (
            <button
              key={m.id}
              type="button"
              className={`model-option ${model === m.id ? 'model-option--active' : ''}`}
              onClick={() => onModelChange(m.id)}
              disabled={loading}
              title={m.desc}
            >
              {m.id === 'xlm-roberta' && (
                <span className="model-badge">🔬</span>
              )}
              {m.label}
              {m.id === 'xlm-roberta' && (
                <span className="model-slow-tag">~1 min</span>
              )}
            </button>
          ))}
        </div>
        <span className="model-desc">
          {MODELS.find(m => m.id === model)?.desc}
        </span>
      </div>
    </div>
  )
}
