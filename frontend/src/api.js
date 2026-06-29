const API_URL = import.meta.env.VITE_API_URL || ''

export async function resolveAvatar(username) {
  try {
    const res = await fetch(
      `https://api.microlink.io/?url=https://letterboxd.com/${username}/`
    )
    if (!res.ok) return null
    const data = await res.json()
    return data?.data?.image?.url || null
  } catch {
    return null
  }
}

const API_KEY = import.meta.env.VITE_API_KEY || ''

export async function fetchRecommendations(body) {
  const headers = { 'Content-Type': 'application/json' }
  if (API_KEY) headers['X-API-Key'] = API_KEY
  const res = await fetch(`${API_URL}/api/recommend`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'API error')
  }
  return res.json()
}
