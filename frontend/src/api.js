const API_URL = import.meta.env.VITE_API_URL || ''

const AVATAR_SOURCES = [
  u => `https://letterboxd.com/${u}/avatar/`,
  u => `https://unavatar.io/letterboxd/${u}`,
]

export function getAvatarUrl(username) {
  return AVATAR_SOURCES[0](username)
}

export function getNextAvatarUrl(username, currentUrl) {
  const idx = AVATAR_SOURCES.findIndex(fn => fn(username) === currentUrl)
  const next = AVATAR_SOURCES[idx + 1]
  return next ? next(username) : null
}

export async function fetchRecommendations(body) {
  const res = await fetch(`${API_URL}/api/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'API error')
  }
  return res.json()
}
