# Seguridad — BoxData

## Arquitectura: modelo de confianza

```
Browser → Vercel (serverless function) → VPS (FastAPI)
```

- `API_KEY` y `VPS_API_URL` solo existen en variables de entorno de Vercel (server-side).
- El navegador nunca recibe la clave ni la URL directa del VPS.
- La Vercel function inyecta `X-API-Key` al hacer proxy hacia el VPS.
- **Esto es correcto.** No hay exposición de secretos al cliente.

## Riesgos actuales (aplicados)

| Gravedad | Riesgo | Impacto | Mitigación |
|----------|--------|---------|------------|
| **Baja** | Sin CSP/HSTS/security headers en frontend | Vulnerable a XSS si hay un fallo en el renderizado. | Añadir cabeceras en `vercel.json`. |
| **Baja** | Dependencias sin versión fija (`>=`) | Build puede romperse o traer un CVE sin control. | Usar `==` o lockfile. |
| **Baja** | Endpoint `/health` sin autenticación | Expone `count_movies()` (información menor). | No urgente, pero idealmente requiere auth. |

## Riesgos resueltos

| Riesgo | Solución |
|--------|----------|
| Sin rate limiting en VPS | `slowapi` implementado: 30 req/min en `/api/recommend` y `/health`, 60 req/min en `/api/avatar`. |
| `API_KEY` opcional | Validación en startup: `raise RuntimeError` si falta. |
| microlink.io recibía usernames | Endpoint propio `/api/avatar/{username}` — extrae OG image del HTML de Letterboxd. Sin terceros. |
| `API_KEY` y TMDB same key | Son claves distintas: `API_TOKEN` (TMDB, config.py), `API_KEY` (BoxData, server.py), `OPENCODE_API_KEY` (LLM). |
| Sin límite en `query` | `max_length=500` en campo `query` de `RecommendRequest`. |
| Docker no era `-slim` | Cambiado a `python:3.11-slim`. |

## Recomendaciones pendientes

### 1. Security headers en frontend

Crear `frontend/vercel.json`:

```json
{
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "Content-Security-Policy", "value": "default-src 'self'; img-src 'self' image.tmdb.org https:; style-src 'self' 'unsafe-inline'; connect-src 'self'" },
        { "key": "X-Content-Type-Options", "value": "nosniff" },
        { "key": "X-Frame-Options", "value": "DENY" },
        { "key": "Referrer-Policy", "value": "strict-origin-when-cross-origin" }
      ]
    }
  ]
}
```

### 2. Dependencias fijas

En `pyproject.toml`, cambiar `>=` por `==` para versiones conocidas, o generar lockfile con `pip freeze`.

### 3. Endpoint `/health` con auth opcional

No urgente, pero añadir `Depends(verify_api_key)` opcionalmente.
