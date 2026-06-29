import { useState, useRef } from 'react'
import { getAvatarUrl, getNextAvatarUrl, fetchRecommendations } from './api'
import { LocaleContext, LOC } from './locale'
import HowItWorks from './HowItWorks'
import './App.css'

function UserChip({ user, onRemove, t }) {
  const [src, setSrc] = useState(user.avatar)
  const [failed, setFailed] = useState(false)

  function handleError() {
    const next = getNextAvatarUrl(user.name, src)
    if (next) {
      setSrc(next)
    } else {
      setFailed(true)
    }
  }

  return (
    <span className="chip">
      {!failed ? (
        <img className="chip-avatar" src={src} alt="" onError={handleError} />
      ) : (
        <span className="chip-avatar chip-avatar--fallback">
          {user.name[0].toUpperCase()}
        </span>
      )}
      <span className="chip-name">{user.name}</span>
      <button className="chip-remove" onClick={() => onRemove(user.name)} aria-label={t.removeUser}>
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
          <path d="M1 1L9 9M9 1L1 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      </button>
    </span>
  )
}

function ResultCard({ movie, rank, loading, shuffle: s, index: i, entering }) {
  const perUser = movie.per_user
  const entries = perUser ? Object.entries(perUser) : []
  const d = (loading ? s : i) * 60
  const xp = {
    transition: `opacity 0.25s ease ${d}ms`,
    opacity: loading ? 0 : 1,
  }
  const xs = {
    transition: `opacity 0.25s ease ${d}ms`,
    opacity: loading ? 1 : 0,
  }

  return (
    <a
      href={movie.url}
      target="_blank"
      rel="noreferrer"
      className={`card${loading ? ' card--loading' : ''}${entering ? ' card-enter' : ''}`}
      style={{ animationDelay: `${d}ms` }}
    >
      <div className="card-poster">
        <div className="card-poster-wrap">
          {movie.poster && (
            <img
              src={movie.poster}
              alt={movie.title}
              loading="lazy"
              className="card-poster-img"
              style={{
                opacity: loading ? 0 : 1,
                transition: `opacity 0.3s ease ${d}ms, transform 0.4s ease`,
              }}
            />
          )}
          <div className="card-poster-shimmer" style={{ animationDelay: `${d}ms`, ...xs }} />
          {!loading && !movie.poster && (
            <div className="card-poster-fallback">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="2" y="2" width="20" height="20" rx="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <path d="M21 15l-5-5L5 21" />
              </svg>
            </div>
          )}
        </div>
        <div className={`card-rank${loading ? ' card-rank--skeleton' : ''}`} style={xp}>
          {loading ? '' : `#${rank}`}
        </div>
        <div className="card-users" style={xp}>
          {!loading && entries.length > 0 && entries.map(([name, score]) => (
            <span key={name} className={`card-user ${score >= 0.7 ? 'card-user--high' : score >= 0.5 ? 'card-user--mid' : ''}`}>
              {name}
            </span>
          ))}
        </div>
      </div>
      <div className="card-body">
        <div className="card-body-content" style={xp}>
          <h3 className="card-title">{movie.title}</h3>
          <div className="card-meta">
            <span className="card-year">{movie.year}</span>
            <span className="card-rating">★ {movie.vote_average?.toFixed(1)}</span>
          </div>
          {entries.length > 0 && (
            <div className="card-scores">
              {entries.map(([name, score]) => (
                <span key={name} className="card-score-label">
                  {name}: {(score * 100).toFixed(0)}
                </span>
              ))}
            </div>
          )}
        </div>
        <div className="card-body-skeleton" style={xs}>
          <div className="card-skeleton-line card-skeleton-line--title" />
          <div className="card-skeleton-line card-skeleton-line--meta" />
          <div className="card-skeleton-line card-skeleton-line--scores" />
        </div>
      </div>
    </a>
  )
}

function RangeSlider({ label, value, onChange, min, max, step, unit }) {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div className="filter-group">
      <div className="filter-header">
        <label>{label}</label>
        <span className="filter-value">{value}{unit}</span>
      </div>
      <div className="slider-track">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
        />
        <div className="slider-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function LocaleToggle({ locale, setLocale }) {
  return (
    <button
      className="locale-btn"
      onClick={() => setLocale(locale === 'en' ? 'es' : 'en')}
      aria-label={locale === 'en' ? 'Switch to Spanish' : 'Cambiar a inglés'}
    >
      {locale === 'en' ? (
        <svg width="22" height="16" viewBox="0 0 24 16" fill="none">
          <rect width="24" height="16" fill="#012169" />
          <path d="M0 0l24 16M24 0L0 16" stroke="#fff" strokeWidth="3.5" />
          <path d="M0 0l24 16M24 0L0 16" stroke="#C8102E" strokeWidth="1.8" />
          <path d="M12 0v16M0 8h24" stroke="#fff" strokeWidth="3.5" />
          <path d="M12 0v16M0 8h24" stroke="#C8102E" strokeWidth="1.8" />
        </svg>
      ) : (
        <svg width="22" height="16" viewBox="0 0 24 16" fill="none">
          <rect width="24" height="16" fill="#C60B1E" />
          <rect y="3" width="24" height="10" fill="#FFC400" />
          <circle cx="12" cy="8" r="2.5" fill="#C60B1E" opacity="0.35" />
        </svg>
      )}
    </button>
  )
}

const YEAR_MIN = 1888
const YEAR_MAX = 2026

function clampYear(v) {
  if (!v) return ''
  const n = parseInt(v)
  if (isNaN(n)) return ''
  return String(Math.max(YEAR_MIN, Math.min(YEAR_MAX, Math.abs(n))))
}

export default function App() {
  const [locale, setLocale] = useState('en')
  const t = LOC[locale].app

  const [users, setUsers] = useState([])
  const [userInput, setUserInput] = useState('')
  const [query, setQuery] = useState('')
  const [n, setN] = useState(12)
  const [minRating, setMinRating] = useState(6.5)
  const [minYear, setMinYear] = useState('')
  const [maxYear, setMaxYear] = useState('')
  const [diversity, setDiversity] = useState(0.3)
  const [randomize, setRandomize] = useState(false)
  const [refresh, setRefresh] = useState(false)
  const [queryWeight, setQueryWeight] = useState(0.3)
  const [genre, setGenre] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [recs, setRecs] = useState(null)
  const [isRefetch, setIsRefetch] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const shuffleRef = useRef([])
  const userInputRef = useRef(null)

  function addUser(raw) {
    const name = raw.trim().toLowerCase()
    if (!name || users.some(u => u.name === name)) return
    setUserInput('')
    setUsers(prev => [...prev, { name, avatar: getAvatarUrl(name) }])
  }

  function removeUser(name) {
    setUsers(prev => prev.filter(u => u.name !== name))
  }

  function handleUserKey(e) {
    if (e.key === 'Enter') {
      e.preventDefault()
      addUser(userInput)
    }
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (users.length === 0) return
    setError(null)
    setIsRefetch(recs !== null && recs.length > 0)
    setLoading(true)

    const body = {
      users: users.map(u => u.name),
      n,
      diversity,
      genre: genre.split(',').map(g => g.trim()).filter(Boolean),
      query: query.trim() || null,
      query_weight: queryWeight,
      min_year: minYear ? Math.max(YEAR_MIN, Math.min(YEAR_MAX, parseInt(minYear))) : null,
      max_year: maxYear ? Math.max(YEAR_MIN, Math.min(YEAR_MAX, parseInt(maxYear))) : null,
      min_rating: minRating,
      randomize,
      refresh,
    }

    try {
      const data = await fetchRecommendations(body)
      setRecs(data)
      const len = data.length
      const idx = Array.from({ length: len }, (_, i) => i)
      for (let i = len - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [idx[i], idx[j]] = [idx[j], idx[i]]
      }
      shuffleRef.current = idx
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const hasQuery = query.trim().length > 0
  const hasResults = recs !== null && !error

  return (
    <LocaleContext.Provider value={{ locale, setLocale }}>
      <div className="app">
        <div className={`layout ${!hasResults ? 'layout--center' : ''}`}>
          <aside className={`sidebar ${sidebarOpen ? 'sidebar--open' : ''}`}>
            <div className="sidebar-header">
              <h2>{t.sidebarTitle}</h2>
            </div>
            <div className="sidebar-body">
              <RangeSlider label={t.results} value={n} onChange={setN} min={1} max={50} step={1} unit="" />
              <RangeSlider label={t.minRating} value={minRating} onChange={setMinRating} min={0} max={10} step={0.5} unit="" />

              <div className="filter-group">
                <div className="filter-header">
                  <label>{t.yearRange}</label>
                </div>
                <div className="filter-row">
                  <input
                    type="number"
                    value={minYear}
                    onChange={e => {
                      const v = clampYear(e.target.value)
                      setMinYear(v)
                      if (maxYear && parseInt(v) > parseInt(maxYear)) setMaxYear(v)
                    }}
                    placeholder={t.from}
                    className="filter-input"
                    min={YEAR_MIN}
                    max={YEAR_MAX}
                  />
                  <span className="filter-sep">—</span>
                  <input
                    type="number"
                    value={maxYear}
                    onChange={e => {
                      const v = clampYear(e.target.value)
                      setMaxYear(v)
                      if (minYear && parseInt(v) < parseInt(minYear)) setMinYear(v)
                    }}
                    placeholder={t.to}
                    className="filter-input"
                    min={YEAR_MIN}
                    max={YEAR_MAX}
                  />
                </div>
              </div>

              <RangeSlider label={t.diversity} value={diversity} onChange={setDiversity} min={0} max={1} step={0.1} unit="" />

              <div className="filter-group">
                <div className="filter-header">
                  <label>{t.textQuery}</label>
                </div>
                <input
                  type="text"
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  placeholder={t.queryPlaceholder}
                  className="filter-input"
                />
              </div>

              {hasQuery && (
                <RangeSlider label={t.queryWeight} value={queryWeight} onChange={setQueryWeight} min={0} max={1} step={0.1} unit="" />
              )}

              <div className="filter-group">
                <div className="filter-header">
                  <label>{t.genres}</label>
                </div>
                <input
                  type="text"
                  value={genre}
                  onChange={e => setGenre(e.target.value)}
                  placeholder={t.genresPlaceholder}
                  className="filter-input"
                />
              </div>

              <label className="filter-checkbox">
                <input type="checkbox" checked={randomize} onChange={e => setRandomize(e.target.checked)} />
                <span>{t.randomize}</span>
              </label>

              <label className="filter-checkbox filter-checkbox--warn">
                <input type="checkbox" checked={refresh} onChange={e => setRefresh(e.target.checked)} />
                <span>{t.rescrape}</span>
              </label>
            </div>
          </aside>

          {sidebarOpen && <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />}

          <main className="main">
            <div className={`search-area ${hasResults ? 'search-area--top' : 'search-area--center'}`}>
              {!hasResults && (
                <h1 className="search-logo">BoxData</h1>
              )}

              <form onSubmit={handleSubmit} className="search-section">
                <div className="search-top-row">
                  <button
                    type="button"
                    className={`sidebar-toggle ${sidebarOpen ? 'is-active' : ''}`}
                    onClick={() => setSidebarOpen(o => !o)}
                    aria-label={t.toggleFilters}
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                      <path d="M4 7h16" />
                      <path d="M7 12h10" />
                      <path d="M10 17h4" />
                    </svg>
                  </button>
                  <div className="search-users">
                    {users.map(user => (
                      <UserChip key={user.name} user={user} onRemove={removeUser} t={t} />
                    ))}
                    <input
                      ref={userInputRef}
                      className="search-users-input"
                      type="text"
                      value={userInput}
                      onChange={e => setUserInput(e.target.value)}
                      onKeyDown={handleUserKey}
                      placeholder={users.length === 0 ? t.addPlaceholder : t.addAnother}
                      autoFocus={!hasResults}
                    />
                    {userInput.trim() && (
                      <button type="button" className="btn-add" onClick={() => addUser(userInput)}>
                        {t.add}
                      </button>
                    )}
                  </div>
                  <div className="search-top-actions">
                    <button
                      type="submit"
                      className="btn-primary"
                      disabled={loading || users.length === 0}
                    >
                      {loading ? (
                        <span className="btn-loading">
                          <span className="spinner" />
                          {t.finding}
                        </span>
                      ) : (
                        <>
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                            <polygon points="5 3 19 12 5 21 5 3" />
                          </svg>
                          {t.recommend}
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </form>

            </div>

            {error && !loading && <div className="error">{error}</div>}

            {recs && recs.length === 0 && !loading && !error && (
              <div className="empty">
                <p>{t.empty}</p>
              </div>
            )}

            {recs && recs.length > 0 && (
              <section className="results">
                <div className={`results-header${loading && isRefetch ? ' results-header--loading' : ''}`}>
                  <h2>{t.recsTitle}</h2>
                  <span className="results-count">{loading && isRefetch ? '⋯' : t.moviesCount(recs.length)}</span>
                </div>
                <div className="grid">
                  {recs.map((movie, i) => (
                    <ResultCard key={i} movie={movie} rank={i + 1} loading={loading && isRefetch} shuffle={shuffleRef.current[i] ?? i} index={i} entering={!isRefetch && !loading} />
                  ))}
                </div>
              </section>
            )}
          </main>
        </div>
        <div className="bottom-buttons">
          <LocaleToggle locale={locale} setLocale={setLocale} />
          <HowItWorks />
        </div>
      </div>
    </LocaleContext.Provider>
  )
}
