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

export async function fetchRecommendations(body, onProgress) {
  // 1. Start the background job
  const startRes = await fetch(`/api/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!startRes.ok) {
    const err = await startRes.json().catch(() => ({}))
    throw new Error(err.detail || 'API error')
  }
  const { job_id } = await startRes.json()
  if (!job_id) throw new Error('No job_id returned')

  // 2. Poll for results
  const maxAttempts = 60 // 120s at 2s intervals
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise(r => setTimeout(r, 2000))

    const statusRes = await fetch(`/api/job/${job_id}`)
    if (!statusRes.ok) continue

    const job = await statusRes.json()

    if (typeof onProgress === 'function') {
      onProgress(job.progress || {})
    }

    if (job.status === 'done') {
      return job.results || []
    }
    if (job.status === 'error') {
      throw new Error(job.error || 'Job failed')
    }
  }

  throw new Error('Recommendation timed out')
}
