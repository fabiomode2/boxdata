const VPS_API = 'https://boxdata.cesura.duckdns.org'

export async function resolveAvatar(username) {
  try {
    const res = await fetch(`/api/avatar?username=${encodeURIComponent(username)}`)
    if (!res.ok) return null
    const data = await res.json()
    return data?.avatar_url || null
  } catch {
    return null
  }
}

export async function fetchRecommendations(body, onProgress) {
  // 1. Start the background job via Vercel function
  const startRes = await fetch(`/api/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!startRes.ok) {
    const err = await startRes.json().catch(() => ({}))
    throw new Error(err.detail || err.error || 'API error')
  }
  const { job_id } = await startRes.json()
  if (!job_id) throw new Error('No job_id returned')

  // 2. Poll the VPS directly for status and progress
  const maxAttempts = 60 // 120s
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise(r => setTimeout(r, 2000))

    try {
      const statusRes = await fetch(`${VPS_API}/api/job/${job_id}`)
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
    } catch (e) {
      if (e.message !== 'Job failed') continue
      throw e
    }
  }

  throw new Error('Recommendation timed out')
}
