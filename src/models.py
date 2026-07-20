from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any


@dataclass
class Movie:
    tmdb_id: int
    title: str
    overview: str
    release_date: date | None
    vote_average: float
    vote_count: int
    genres: list[str] = field(default_factory=list)
    director: str | None = None
    cast: list[str] = field(default_factory=list)
    poster_path: str | None = None
    top_reviews: list[str] = field(default_factory=list)
    imdb_id: str | None = None
    imdb_rating: float | None = None
    imdb_vote_count: int | None = None
    metacritic_score: int | None = None
    metacritic_review_count: int | None = None
    interests: list[str] = field(default_factory=list)
    origin_countries: list[str] = field(default_factory=list)
    spoken_languages: list[str] = field(default_factory=list)

    def enriched_text(self) -> str:
        year = self.release_date.year if self.release_date else ""
        genres = ", ".join(self.genres) if self.genres else ""
        interests = ", ".join(self.interests) if self.interests else ""

        parts = [f"{self.title} is a {year} film."]
        if genres:
            parts.append(f"Genres: {genres}.")
        if interests:
            parts.append(f"Subgenres: {interests}.")
        if self.director:
            parts.append(f"Directed by {self.director}.")
        if self.cast:
            parts.append(f"Starring {', '.join(self.cast[:5])}.")
        if self.origin_countries:
            parts.append(f"Origin: {', '.join(self.origin_countries)}.")
        if self.spoken_languages:
            parts.append(f"Languages: {', '.join(self.spoken_languages)}.")
        if self.overview:
            parts.append(self.overview)

        rating = self.imdb_rating or self.vote_average
        votes = self.imdb_vote_count or self.vote_count
        if votes > 0:
            if rating >= 8.0:
                quality = "widely acclaimed"
            elif rating >= 6.0:
                quality = "well-received"
            elif rating >= 4.0:
                quality = "mixed"
            else:
                quality = "poorly received"

            if votes >= 1000000:
                audience = "a massive audience"
            elif votes >= 100000:
                audience = "a very large audience"
            elif votes >= 10000:
                audience = "a large audience"
            elif votes >= 1000:
                audience = "a modest audience"
            else:
                audience = "a limited audience"
            parts.append(f"It has been {quality} by {audience}.")

        if self.top_reviews:
            parts.append(f"Audience comments: {' '.join(self.top_reviews[:3])}")

        return " ".join(parts)

    def metadata(self) -> dict:
        return {
            "title": self.title,
            "director": self.director or "",
            "genres": ",".join(self.genres),
            "year": self.release_date.year if self.release_date else 0,
            "vote_average": self.vote_average,
            "vote_count": self.vote_count,
            "imdb_rating": self.imdb_rating if self.imdb_rating is not None else 0.0,
            "imdb_vote_count": self.imdb_vote_count if self.imdb_vote_count is not None else 0,
            "metacritic_score": self.metacritic_score if self.metacritic_score is not None else 0,
            "interests": ",".join(self.interests),
            "poster_path": self.poster_path or "",
            "cast": ",".join(self.cast) if self.cast else "",
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["release_date"] = str(self.release_date) if self.release_date else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Movie":
        d = dict(d)
        if d.get("release_date"):
            d["release_date"] = date.fromisoformat(d["release_date"])
        return cls(**d)
