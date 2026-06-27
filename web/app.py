from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse

logging.basicConfig(level=logging.WARNING)

app = FastAPI(title="BoxData")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Create a symlink or copy of the embedding map if it exists
output_map = Path(__file__).resolve().parent.parent / "output" / "embedding_map.html"
if output_map.exists():
    import shutil
    shutil.copy(str(output_map), str(STATIC_DIR / "embedding_map.html"))

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _page(body: str, has_map: bool = False) -> str:
    map_link = '<a class="map-link" href="/static/embedding_map.html" target="_blank">\U0001f5fa View embedding map</a>' if has_map else ""
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BoxData</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d0d0d; color: #e0e0e0; padding: 2rem; max-width: 960px; margin: auto; }}
h1 {{ font-size: 1.8rem; margin-bottom: 0.3rem; color: #fff; }}
.subtitle {{ color: #999; margin-bottom: 2rem; }}
form {{ background: #1a1a2e; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; }}
.row {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }}
.field {{ flex: 1; min-width: 140px; }}
label {{ display: block; font-size: 0.8rem; color: #aaa; margin-bottom: 0.3rem; }}
input, select {{ width: 100%; padding: 0.5rem; border-radius: 6px; border: 1px solid #333; background: #16213e; color: #e0e0e0; font-size: 0.9rem; }}
input:focus {{ outline: none; border-color: #e94560; }}
button {{ width: 100%; padding: 0.7rem; background: #e94560; color: #fff; border: none; border-radius: 6px; font-size: 1rem; cursor: pointer; margin-top: 0.5rem; }}
button:hover {{ background: #c73650; }}
.error {{ background: #3d0f0f; border: 1px solid #e94560; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
th {{ text-align: left; padding: 0.5rem; color: #999; font-size: 0.8rem; border-bottom: 1px solid #333; }}
td {{ padding: 0.5rem; border-bottom: 1px solid #222; }}
.num {{ color: #e94560; font-weight: bold; }}
.title {{ color: #fff; }}
.genres {{ color: #888; font-size: 0.85rem; }}
.score {{ font-family: monospace; color: #4ade80; }}
.url {{ color: #60a5fa; text-decoration: none; font-size: 0.85rem; }}
.url:hover {{ text-decoration: underline; }}
.poster {{ width: 50px; border-radius: 4px; vertical-align: middle; }}
.check-label {{ font-size: 0.85rem; color: #ccc; cursor: pointer; }}
.check-label input {{ width: auto; margin-right: 0.4rem; }}
.poster-cell {{ width: 55px; }}
.empty {{ color: #888; text-align: center; padding: 2rem; }}
footer {{ margin-top: 3rem; color: #555; font-size: 0.8rem; text-align: center; }}
.map-link {{ display: inline-block; margin-top: 1rem; color: #60a5fa; }}
</style>
</head>
<body>
<h1>\U0001f3ac BoxData</h1>
<p class="subtitle">Movie recommendations powered by embeddings + kernel regression</p>
{body}
<footer>BoxData — Ollama + nomic-embed-text + ChromaDB + kernel regression {map_link}</footer>
</body>
</html>"""


def _form(users: str = "", n: int = 10, diversity: float = 0.0,
          genre: str = "", query: str = "", query_weight: float = 0.3,
          auto_sigma: bool = True, min_year: str = "", max_year: str = "",
          min_rating: float = 6.5, randomize: bool = False) -> str:
    return f"""<form method="post" action="/recommend">
<div class="row">
<div class="field" style="flex:2">
<label>Letterboxd users (comma-separated)</label>
<input name="users" placeholder="alice, bob" value="{users}">
</div>
<div class="field">
<label>Results</label>
<input name="n" type="number" value="{n}" min="1" max="50">
</div>
</div>
<div class="row">
<div class="field">
<label>Texto libre (semántica)</label>
<input name="query" placeholder="cyberpunk, noir, mystery..." value="{query}">
</div>
<div class="field">
<label>Query weight</label>
<input name="query_weight" type="number" step="0.1" value="{query_weight}" min="0" max="1">
</div>
</div>
<div class="row">
<div class="field">
<label>Filtro por género (TMDB)</label>
<input name="genre" placeholder="Mystery, Horror..." value="{genre}">
</div>
<div class="field">
<label>Diversity</label>
<input name="diversity" type="number" step="0.1" value="{diversity}" min="0" max="1">
</div>
<div class="field">
<label>Sigma</label>
<select name="auto_sigma">
<option value="true"{" selected" if auto_sigma else ""}>Auto (CV)</option>
<option value="false"{" selected" if not auto_sigma else ""}>Fixed 0.3</option>
</select>
</div>
</div>
<div class="row">
<div class="field">
<label>Año desde</label>
<input name="min_year" type="number" placeholder="1920" value="{min_year}" min="1900" max="2030">
</div>
<div class="field">
<label>Año hasta</label>
<input name="max_year" type="number" placeholder="2026" value="{max_year}" min="1900" max="2030">
</div>
<div class="field">
<label>Rating mínimo ⭐</label>
<input name="min_rating" type="number" step="0.5" value="{min_rating}" min="0" max="10">
</div>
<div class="field">
<label>Aleatorizar</label>
<label class="check-label"><input type="checkbox" name="randomize" value="true"{' checked' if randomize else ''}> semillas variables</label>
</div>
</div>
<button type="submit">Recommend</button>
</form>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    has_map = (STATIC_DIR / "embedding_map.html").exists()
    body = _form() + '<div class="empty">Enter usernames (or a text query) and click Recommend</div>'
    return _page(body, has_map=has_map)


@app.post("/recommend", response_class=HTMLResponse)
async def do_recommend(
    users: str = Form(""),
    n: int = Form(10),
    diversity: float = Form(0.0),
    genre: str = Form(""),
    query: str = Form(""),
    query_weight: float = Form(0.3),
    auto_sigma: bool = Form(True),
    min_year: str = Form(""),
    max_year: str = Form(""),
    min_rating: float = Form(6.5),
    randomize: bool = Form(False),
):
    from src.recommender import (
        fetch_letterboxd_watched,
        get_taste_vector,
        recommend_for_group,
        recommend_from_query,
        recommend_hybrid,
    )

    user_weighted: list[dict[int, float]] = []
    error = None
    recs = []
    user_list = [u.strip() for u in users.split(",") if u.strip()]
    min_year_i = int(min_year) if min_year.strip() else None
    max_year_i = int(max_year) if max_year.strip() else None

    if user_list:
        for username in user_list:
            try:
                weighted = fetch_letterboxd_watched(username)
                vec, found_ids, total = get_taste_vector(weighted)
                if found_ids:
                    user_weighted.append(weighted)
            except Exception as e:
                error = f"Error scraping {username}: {e}"
                break

    KNOWN_GENRES = {'action','adventure','animation','comedy','crime','documentary','drama','family','fantasy','history','horror','music','mystery','romance','science fiction','thriller','war','western'}
    if query and query.lower().strip() in KNOWN_GENRES:
        genre = query
        genre_filter = [query.strip().title()]
        query = ""
    else:
        genre_filter = [g.strip() for g in genre.split(",") if g.strip()] if genre else None
    effective_sigma = None if auto_sigma else 0.3

    if not error:
        try:
            if user_weighted and query:
                recs = recommend_hybrid(
                    user_weighted, query, n=n, query_weight=query_weight,
                    sigma=effective_sigma, diversity=diversity,
                    genre_filter=genre_filter,
                    min_year=min_year_i, max_year=max_year_i,
                    min_rating=min_rating, randomize=randomize,
                )
            elif user_weighted:
                recs = recommend_for_group(
                    user_weighted, n=n, sigma=effective_sigma,
                    diversity=diversity, genre_filter=genre_filter,
                    min_year=min_year_i, max_year=max_year_i,
                    min_rating=min_rating, randomize=randomize,
                )
            elif query:
                recs = recommend_from_query(
                    query, n=n, diversity=diversity, genre_filter=genre_filter,
                    min_year=min_year_i, max_year=max_year_i,
                    min_rating=min_rating, randomize=randomize,
                )
        except Exception as e:
            error = str(e)

    form = _form(users, n, diversity, genre, query, query_weight, auto_sigma, min_year, max_year, min_rating, randomize)

    if error:
        return _page(form + f'<div class="error">{error}</div>')

    if not recs:
        return _page(form + '<div class="empty">Enter usernames (or a text query) and click Recommend</div>')

    rows = "\n".join(
        f"""<tr>
<td class="num">{i}</td>
<td class="poster-cell">{'<img class="poster" src="' + r['poster'] + '">' if r.get('poster') else ''}</td>
<td><div class="title">{r['title']}</div><a class="url" href="{r['url']}" target="_blank">Letterboxd \u2192</a></td>
<td>{r['year']}</td>
<td class="genres">{r['genres'][:40]}</td>
<td class="score">{r['vote_average']:.1f}</td>
</tr>"""
        for i, r in enumerate(recs, 1)
    )
    table = f"""<table><tr><th>#</th><th></th><th>Title</th><th>Year</th><th>Genres</th><th>TMDB ⭐</th></tr>{rows}</table>"""
    return _page(form + table)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
