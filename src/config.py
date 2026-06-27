import os
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")

TMDB_BASE = "https://api.themoviedb.org/3"
HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {API_TOKEN}",
}

IMDB_BASE = "https://api.imdbapi.dev"
