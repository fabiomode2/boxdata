from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", host: str | None = None) -> None:
        import ollama
        self._client = ollama.Client(host=host)
        self._model = model

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            resp = self._client.embeddings(model=self._model, prompt=text)
            results.append(resp["embedding"])
        return results
