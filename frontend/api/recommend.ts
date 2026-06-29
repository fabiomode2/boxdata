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
    const response = await fetch(`${apiUrl}/api/recommend`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": apiKey,
      },
      body: JSON.stringify(req.body),
    })

    const data = await response.json()
    return res.status(response.status).json(data)
  } catch (err) {
    return res.status(502).json({ error: "Upstream API error" })
  }
}
