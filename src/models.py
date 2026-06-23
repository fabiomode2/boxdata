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

    def enriched_text(self) -> str:
        year = self.release_date.year if self.release_date else ""
        genres = ", ".join(self.genres) if self.genres else ""

        parts = [f"{self.title} is a {year} film."]
        if genres:
            parts.append(f"Genres: {genres}.")
        if self.director:
            parts.append(f"Directed by {self.director}.")
        if self.cast:
            parts.append(f"Starring {', '.join(self.cast[:5])}.")
        if self.overview:
            parts.append(self.overview)

        if self.vote_count > 0:
            if self.vote_average >= 8.0:
                quality = "widely acclaimed"
            elif self.vote_average >= 6.0:
                quality = "well-received"
            elif self.vote_average >= 4.0:
                quality = "mixed"
            else:
                quality = "poorly received"

            if self.vote_count >= 10000:
                audience = "a massive audience"
            elif self.vote_count >= 5000:
                audience = "a very large audience"
            elif self.vote_count >= 1000:
                audience = "a large audience"
            elif self.vote_count >= 100:
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
