import { useState } from 'react'
import { useLocale, LOC } from './locale'

const accent = '#6366f1'
const WIKI = 'https://en.wikipedia.org/wiki/Word_embedding'

function Arrow() {
  return (
    <svg className="pipe-arrow" width="20" height="20" viewBox="0 0 24 24" fill="none">
      <path d="M5 12h14M13 5l7 7-7 7" stroke={accent} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function PipeStep({ icon, label, sub }) {
  return (
    <div className="pipe-step">
      <div className="pipe-step-icon">{icon}</div>
      <div className="pipe-step-label">{label}</div>
      {sub && <div className="pipe-step-sub">{sub}</div>}
    </div>
  )
}

function Pipeline({ steps }) {
  return (
    <div className="pipeline">
      {steps.map((step, i) => (
        <span key={i} className="pipeline-group">
          <PipeStep {...step} />
          {i < steps.length - 1 && <Arrow />}
        </span>
      ))}
    </div>
  )
}

function EmbedLink({ children }) {
  return (
    <a href={WIKI} target="_blank" rel="noreferrer" className="embed-link">
      {children}
    </a>
  )
}

const DATA_ICONS = [
  <svg key="tmdb" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="2" width="20" height="20" rx="2" /><path d="M7 2v20M17 2v20M2 12h20" />
  </svg>,
  <svg key="text" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 6h16M4 12h16M4 18h12" /><path d="M16 18h4v-4" />
  </svg>,
  <svg key="ai" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="4" /><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
  </svg>,
  <svg key="db" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5" /><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3" />
  </svg>,
]

const REC_ICONS = [
  <svg key="users" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="9" cy="7" r="3" /><circle cx="15" cy="7" r="3" /><path d="M3 21v-2a4 4 0 014-4h10a4 4 0 014 4v2" />
  </svg>,
  <svg key="kernel" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 4l16 16" /><path d="M4 20l16-16" /><circle cx="12" cy="12" r="4" /><path d="M12 8v8M8 12h8" />
  </svg>,
  <svg key="rrf" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
  </svg>,
  <svg key="results" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M8 21V3l8 4-8 4" />
  </svg>,
]

const TEXTS = {
  en: {
    dataText: (
      <>
        Each movie is described based on its features and then{' '}
        <EmbedLink>embedded</EmbedLink> — an AI model transforms it into a
        768-number vector that represents it. Multiple movies are processed
        this way and stored in a specialized vector database.
      </>
    ),
    recText: (
      <>
        For every movie a user has watched that exists in the database, it's
        used as a reference point to create a function that approximates the
        user's taste across the entire space of{' '}
        <EmbedLink>embedded</EmbedLink> movies.
      </>
    ),
    consensusText1: (
      <>
        The interpolation functions of all users are combined into one using the{' '}
        <strong>geometric mean</strong>. Then several points in the space are
        chosen — random or determined by the{' '}
        <EmbedLink>embedding</EmbedLink> of a text query — and the local
        maxima of the group function are found from those points.
      </>
    ),
    consensusText2: (
      <>
        Finally, movies whose vectors are closest to those maxima are retrieved
        and sorted by proximity, popularity, and variety.
      </>
    ),
  },
  es: {
    dataText: (
      <>
        Cada pelicula se describe en base a sus caracteristicas y se{' '}
        <EmbedLink>embedea</EmbedLink>, es decir, se usa un modelo de
        inteligencia artificial para transformarla en un vector de 768 numeros
        que la representa. Se procesan multiples peliculas de esta manera y se
        guardan en una base de datos especializada en vectores.
      </>
    ),
    recText: (
      <>
        Para cada pelicula vista por un usuario que este en la base de datos,
        se usa como punto de referencia para crear una funcion que aproxima el
        gusto del usuario en todo el espacio de peliculas{' '}
        <EmbedLink>embebidas</EmbedLink>.
      </>
    ),
    consensusText1: (
      <>
        Se combinan las funciones de interpolacion de todos los usuarios en una
        sola usando la <strong>media geometrica</strong>. Despues se escogen
        varios puntos en el espacio — aleatorios o determinados por el{' '}
        <EmbedLink>embedding</EmbedLink> de una busqueda de texto — y se
        buscan los maximos locales de la funcion del grupo desde esos puntos.
      </>
    ),
    consensusText2: (
      <>
        Por ultimo, se recuperan las peliculas de la base de datos cuyos
        vectores esten mas cerca de esos maximos y se ordenan combinando
        cercania, popularidad y variedad.
      </>
    ),
  },
}

export default function HowItWorks() {
  const [open, setOpen] = useState(false)
  const { locale } = useLocale()
  const t = LOC[locale].how
  const tx = TEXTS[locale]

  const dataSteps = [
    { icon: DATA_ICONS[0], label: 'TMDB', sub: t.dataSubs[0] },
    { icon: DATA_ICONS[1], label: 'Text Profile', sub: t.dataSubs[1] },
    { icon: DATA_ICONS[2], label: 'AI Embedding', sub: t.dataSubs[2] },
    { icon: DATA_ICONS[3], label: 'ChromaDB', sub: t.dataSubs[3] },
  ]

  const recSteps = [
    { icon: REC_ICONS[0], label: 'Taste Vectors', sub: t.recSubs[0] },
    { icon: REC_ICONS[1], label: 'Kernel Reg.', sub: t.recSubs[1] },
    { icon: REC_ICONS[2], label: 'RRF + MMR', sub: t.recSubs[2] },
    { icon: REC_ICONS[3], label: 'Results', sub: t.recSubs[3] },
  ]

  return (
    <>
      <button className="info-btn" onClick={() => setOpen(true)} aria-label="How it works">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="16" x2="12" y2="12" />
          <line x1="12" y1="8" x2="12.01" y2="8" />
        </svg>
      </button>

      {open && (
        <div className="info-overlay" onClick={() => setOpen(false)}>
          <div className="info-panel" onClick={e => e.stopPropagation()}>
            <button className="info-close" onClick={() => setOpen(false)} aria-label="Close">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>

            <div className="info-hero">
              <h2 className="info-title">{t.title}</h2>
            </div>

            <p className="info-desc">{t.desc}</p>

            <section className="info-section">
              <h3>{t.dataTitle}</h3>
              <Pipeline steps={dataSteps} />
              <p className="info-section-text">{tx.dataText}</p>
            </section>

            <section className="info-section">
              <h3>{t.recTitle}</h3>
              <Pipeline steps={recSteps} />
              <p className="info-section-text">{tx.recText}</p>
            </section>

            <section className="info-section">
              <h3>{t.consensusTitle}</h3>
              <p className="info-section-text">{tx.consensusText1}</p>
              <div className="info-formula">
                <div className="info-formula-row">
                  <span className="info-formula-label">{t.formulaWeight}</span>
                  <span className="info-formula-op">=</span>
                  <span className="info-formula-expr">exp(−dist² / σ²)</span>
                </div>
                <div className="info-formula-row">
                  <span className="info-formula-label">{t.formulaDist}</span>
                  <span className="info-formula-op">=</span>
                  <span className="info-formula-expr">1 − similitud entre vectores</span>
                </div>
                <div className="info-formula-divider" />
                <div className="info-formula-row">
                  <span className="info-formula-label">{t.formulaRating}</span>
                  <span className="info-formula-op">=</span>
                  <span className="info-formula-expr">Σ(peso × nota) / Σ(peso)</span>
                </div>
              </div>
              <p className="info-section-text">{tx.consensusText2}</p>
            </section>

          </div>
        </div>
      )}
    </>
  )
}
