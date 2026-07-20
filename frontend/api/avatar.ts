export default async function handler(req, res) {
  if (req.method !== "GET") {
    return res.status(405).json({ error: "Method not allowed" })
  }

  const username = req.query.username
  if (!username || !/^[a-zA-Z0-9_-]{1,50}$/.test(username)) {
    return res.status(400).json({ error: "Invalid username" })
  }

  const apiKey = process.env.API_KEY
  const apiUrl = process.env.VPS_API_URL

  if (!apiKey || !apiUrl) {
    return res.status(500).json({ error: "Server misconfiguration" })
  }

  try {
    const response = await fetch(
      `${apiUrl}/api/avatar/${encodeURIComponent(username)}`,
      {
        headers: { "X-API-Key": apiKey },
      }
    )
    const data = await response.json()
    return res.status(response.status).json(data)
  } catch {
    return res.status(502).json({ avatar_url: null })
  }
}
