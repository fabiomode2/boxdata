export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" })
  }

  const apiKey = process.env.API_KEY
  const apiUrl = process.env.VPS_API_URL

  if (!apiKey || !apiUrl) {
    return res.status(500).json({ error: "Server misconfiguration" })
  }

  try {
    // 1. Start the background job
    const startRes = await fetch(`${apiUrl}/api/recommend`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": apiKey,
      },
      body: JSON.stringify(req.body),
    })
    if (!startRes.ok) {
      const err = await startRes.json().catch(() => ({ detail: "API error" }))
      return res.status(startRes.status).json(err)
    }

    const { job_id } = await startRes.json()
    if (!job_id) {
      return res.status(502).json({ error: "No job_id returned" })
    }

    // 2. Poll for results (max 120s, every 2s)
    const maxAttempts = 60
    for (let i = 0; i < maxAttempts; i++) {
      await new Promise(r => setTimeout(r, 2000))

      const statusRes = await fetch(`${apiUrl}/api/job/${job_id}`, {
        headers: { "X-API-Key": apiKey },
      })
      if (!statusRes.ok) continue

      const job = await statusRes.json()

      if (job.status === "done") {
        return res.status(200).json(job.results || [])
      }
      if (job.status === "error") {
        return res.status(502).json({ error: job.error || "Job failed" })
      }
      // status === "queued" | "scraping" → keep polling
    }

    return res.status(504).json({ error: "Job timed out" })
  } catch (err) {
    return res.status(502).json({ error: "Upstream API error" })
  }
}
