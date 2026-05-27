import { useState } from 'react'
import './SearchBar.css'

export default function SearchBar({ onAnalyze, loading }) {
  const [url, setUrl] = useState('')

  function handleSubmit(e) {
    e.preventDefault()
    const trimmed = url.trim()
    if (!trimmed) return
    onAnalyze(trimmed)
  }

  return (
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
  )
}
