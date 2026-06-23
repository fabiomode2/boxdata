import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_TOKEN = os.getenv("API_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {API_TOKEN}",
}
