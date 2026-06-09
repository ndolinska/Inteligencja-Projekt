import { useState } from 'react'
import SearchBar from './components/SearchBar'
import Dashboard from './components/Dashboard'
import './App.css'

export default function App() {
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [videoUrl, setVideoUrl] = useState('')
  const [model, setModel]     = useState('vader')

  async function handleAnalyze(url) {
    setLoading(true)
    setError(null)
    setData(null)
    setVideoUrl(url)

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_url: url, max_comments: 500, model }),
      })
      if (!res.ok) {
        let errorMsg = `Błąd serwera (HTTP ${res.status})`
        try {
          const err = await res.json()
          errorMsg = err.detail || errorMsg
        } catch {
          // Odpowiedź nie jest JSONem — używamy statusu
        }
        throw new Error(errorMsg)
      }
      setData(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">◈</span>
            <span className="logo-text">SentimentScope</span>
          </div>
          <p className="logo-sub">Analiza nastrojów widzów YouTube</p>
        </div>
      </header>

      <main className="app-main">
        <SearchBar
          onAnalyze={handleAnalyze}
          loading={loading}
          model={model}
          onModelChange={setModel}
        />

        {error && (
          <div className="error-box">
            <span className="error-icon">⚠</span>
            {error}
          </div>
        )}

        {loading && (
          <div className="loading-box">
            <div className="spinner" />
            <p>Pobieranie i analiza komentarzy…</p>
            <p className="loading-sub">To może potrwać kilkanaście sekund</p>
          </div>
        )}

        {data && <Dashboard data={data} videoUrl={videoUrl} />}

        {!data && !loading && !error && (
          <div className="empty-state">
            <div className="empty-icon">▶</div>
            <h2>Wklej link do wideo YouTube</h2>
            <p>Aplikacja pobierze komentarze i przeanalizuje nastroje widzów</p>
          </div>
        )}
      </main>
    </div>
  )
}
