# Seguridad — BoxData

## Modelo de confianza

```
Browser → Vercel (serverless) → VPS (FastAPI) → Ollama (red privada)
```

- `API_KEY` y `VPS_API_URL` viven solo en variables de entorno de Vercel (server-side).
- El navegador nunca recibe la clave. La URL pública del VPS sí es visible; los jobs se consultan con IDs aleatorios de 128 bits.
- La función de Vercel inyecta `X-API-Key` en las llamadas autenticadas al VPS.
- El VPS no publica el puerto de la API: solo es accesible desde la red de Docker del reverse proxy.

## Secretos y repositorio

Auditoría del repositorio y de todo el historial de Git:

- `.env` nunca estuvo trackeado. Solo se publica `.env.example` con placeholders.
- Sin tokens TMDB, claves de API, tokens de GitHub ni claves privadas en ningún commit.
- `frontend/.env.local` y `.vercel/` están ignorados.
- `AGENTS.md` y artefactos generados (`misc/rating_distribution.html`) no se publican.

Los secretos reales están solo en `.env` del VPS (permisos `600`) y en las variables de entorno de Vercel. Si una clave se filtra, hay que rotarla; el código no la contiene.

## Controles aplicados

| Control | Implementación |
|---|---|
| Autenticación | `X-API-Key` obligatoria en `POST /api/recommend` y `/api/avatar`; comparación en tiempo constante con `secrets.compare_digest`. |
| Rate limiting | `slowapi`: 30/min recommend y health, 60/min avatar, 120/min polling. |
| Tamaño de request | Middleware rechaza cuerpos > 64 KB antes de parsearlos. |
| Validación de entrada | Usernames `[a-zA-Z0-9_-]{1,50}`, query ≤ 500, géneros ≤ 20 × 50, años 1888–2100, `n` ≤ 50. |
| Jobs | Máximo 4 concurrentes (`MAX_CONCURRENT_JOBS`), IDs de 128 bits, retención 5 min, limpieza de jobs colgados a las 2 h. |
| Errores | El cliente recibe un mensaje genérico; el detalle queda solo en logs del servidor. |
| CORS | Orígenes explícitos, métodos `GET/POST/OPTIONS`, headers `Content-Type/X-API-Key`. |
| Docs | `/docs`, `/redoc` y `/openapi.json` deshabilitados en producción (`DISABLE_DOCS=1`). |
| SSRF | El scraper de avatares solo consulta `letterboxd.com`; username validado por regex. |
| Contenedor | API corre con el UID/GID del host (`BOXDATA_UID`/`BOXDATA_GID`), `cap_drop: ALL`, `no-new-privileges`, sin puertos publicados. |
| Red | Ollama aislado en `backend` (interna); solo la API está en `proxy-network`. |

## Riesgos pendientes (menores)

| Gravedad | Riesgo | Mitigación sugerida |
|---|---|---|
| Baja | `/health` sin auth y ejecuta `count_movies()` | Añadir caché del conteo o protegerlo en el reverse proxy. |
| Baja | Dependencias con `>=` sin lockfile | Generar lockfile y actualizar con Dependabot/Renovate. |
| Baja | Sin CSP/HSTS en el frontend de Vercel | Añadir headers en `frontend/vercel.json`. |
| Baja | Sin límite de intentos a nivel de host | Mantener fail2ban + firewall y limitar SSH a Tailscale/VPN si es posible. |

## Operación

- Rotar `API_KEY` periódicamente y tras cualquier sospecha.
- No commitear `.env`; usar `.env.example` para documentar variables.
- Revisar logs del contenedor sin volcar valores de entorno.
- Mantener el reverse proxy y las imágenes Docker actualizadas.
