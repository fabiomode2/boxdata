import { useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [users, setUsers] = useState('')
  const [query, setQuery] = useState('')
  const [n, setN] = useState(10)
  const [diversity, setDiversity] = useState(0)
  const [genre, setGenre] = useState('')
  const [minYear, setMinYear] = useState('')
  const [maxYear, setMaxYear] = useState('')
  const [minRating, setMinRating] = useState(6.5)
  const [randomize, setRandomize] = useState(false)
  const [queryWeight, setQueryWeight] = useState(0.3)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [recs, setRecs] = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setRecs(null)

    const body = {
      users: users.split(',').map(u => u.trim()).filter(Boolean),
      n,
      diversity,
      genre: genre.split(',').map(g => g.trim()).filter(Boolean),
      query: query.trim() || null,
      query_weight: queryWeight,
      min_year: minYear ? parseInt(minYear) : null,
      max_year: maxYear ? parseInt(maxYear) : null,
      min_rating: minRating,
      randomize,
    }

    try {
      const res = await fetch(`${API_URL}/api/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'API error')
      }
      const data = await res.json()
      setRecs(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1 className="logo">
            <span className="logo-icon">🎬</span>
            <span className="logo-text">BoxData</span>
          </h1>
          <p className="tagline">Movie recommendations powered by vector embeddings</p>
        </div>
      </header>

      <main className="main">
        <form onSubmit={handleSubmit} className="form">
          <div className="form-grid">
            <div className="form-group span-2">
              <label>Letterboxd users</label>
              <input value={users} onChange={e => setUsers(e.target.value)} placeholder="alice, bob..." />
            </div>
            <div className="form-group span-2">
              <label>Text query (optional)</label>
              <input value={query} onChange={e => setQuery(e.target.value)} placeholder="cyberpunk, noir, 80s sci-fi..." />
            </div>
            <div className="form-group">
              <label>Results</label>
              <input type="number" value={n} onChange={e => setN(+e.target.value)} min={1} max={50} />
            </div>
            <div className="form-group">
              <label>Query weight</label>
              <input type="number" step={0.1} value={queryWeight} onChange={e => setQueryWeight(+e.target.value)} min={0} max={1} />
            </div>
            <div className="form-group">
              <label>Diversity</label>
              <input type="number" step={0.1} value={diversity} onChange={e => setDiversity(+e.target.value)} min={0} max={1} />
            </div>
            <div className="form-group">
              <label>Genre filter</label>
              <input value={genre} onChange={e => setGenre(e.target.value)} placeholder="Mystery, Horror" />
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>Year from</label>
                <input type="number" value={minYear} onChange={e => setMinYear(e.target.value)} placeholder="1920" />
              </div>
              <div className="form-group">
                <label>Year to</label>
                <input type="number" value={maxYear} onChange={e => setMaxYear(e.target.value)} placeholder="2026" />
              </div>
              <div className="form-group">
                <label>Min rating</label>
                <input type="number" step={0.5} value={minRating} onChange={e => setMinRating(+e.target.value)} min={0} max={10} />
              </div>
              <div className="form-group checkbox-group">
                <label className="checkbox-label">
                  <input type="checkbox" checked={randomize} onChange={e => setRandomize(e.target.checked)} />
                  <span>Randomize</span>
                </label>
              </div>
            </div>
          </div>
          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? <span className="btn-loading"><span className="spinner" /> Searching...</span> : 'Find movies →'}
          </button>
        </form>

        {error && <div className="error">{error}</div>}

        {recs && recs.length === 0 && !error && (
          <div className="empty">No recommendations found. Try different users or filters.</div>
        )}

        {recs && recs.length > 0 && (
          <section className="results">
            <h2 className="results-title">Recommendations</h2>
            <div className="grid">
              {recs.map((r, i) => (
                <a key={r.tmdb_id || i} href={r.url} target="_blank" rel="noreferrer" className="card">
                  <div className="card-poster">
                    {r.poster ? (
                      <img src={r.poster} alt={r.title} loading="lazy" />
                    ) : (
                      <div className="no-poster">No poster</div>
                    )}
                    <div className="card-overlay">
                      <div className="card-rank">#{i + 1}</div>
                      <h3 className="card-title">{r.title}</h3>
                      <div className="card-year">{r.year}</div>
                      <div className="card-genres">{r.genres?.slice(0, 40)}</div>
                      <div className="card-stats">
                        <span className="card-rating">★ {r.vote_average?.toFixed(1)}</span>
                        <span className="card-score">{r.rerank_score?.toFixed(4)}</span>
                      </div>
                      <div className="card-link">Open on Letterboxd →</div>
                    </div>
                  </div>
                </a>
              ))}
            </div>
          </section>
        )}

        {!recs && !error && (
          <div className="empty-state">
            <div className="empty-icon">🎥</div>
            <p>Enter some Letterboxd usernames or a text query to get started</p>
          </div>
        )}
      </main>

      <footer className="footer">
        <span>BoxData — Ollama + ChromaDB + kernel regression</span>
      </footer>
    </div>
  )
}
