export async function resolveAvatar(username) {
  try {
    const res = await fetch(`/api/avatar/${encodeURIComponent(username)}`)
    if (!res.ok) return null
    const data = await res.json()
    return data?.avatar_url || null
  } catch {
    return null
  }
}

export async function fetchRecommendations(body) {
  const res = await fetch(`/api/recommend`, {
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
