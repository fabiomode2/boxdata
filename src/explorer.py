from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from hdbscan import HDBSCAN
from umap import UMAP

from src.vector_store import get_collection

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def _load_all_embeddings() -> tuple[np.ndarray, list[dict]]:
    col = get_collection()
    total = col.count()
    logger.info("Loading %d embeddings from ChromaDB...", total)
    batch_size = 1000
    all_embs: list[np.ndarray] = []
    all_metas: list[dict] = []

    for offset in range(0, total, batch_size):
        res = col.get(
            offset=offset,
            limit=batch_size,
            include=["embeddings", "metadatas"],
        )
        for i, tid in enumerate(res["ids"]):
            emb = res["embeddings"][i]
            meta = res["metadatas"][i]
            if emb is not None:
                arr = np.array(emb.tolist() if hasattr(emb, "tolist") else emb)
                all_embs.append(arr)
                all_metas.append({
                    "tmdb_id": int(tid),
                    "title": meta.get("title", "Unknown"),
                    "year": meta.get("year", 0),
                    "genres": meta.get("genres", ""),
                    "vote_average": meta.get("vote_average", 0),
                })

    logger.info("Loaded %d embeddings.", len(all_embs))
    return np.array(all_embs), all_metas


def _build_genre_axes(
    embs: np.ndarray,
    metas: list[dict],
) -> dict[str, np.ndarray]:
    """Compute genre centroid axes: genreA - genreB vectors."""
    genre_vectors: dict[str, list[np.ndarray]] = {}
    for i, meta in enumerate(metas):
        gstr = meta.get("genres", "")
        for g in gstr.split(","):
            g = g.strip().lower()
            if g:
                genre_vectors.setdefault(g, []).append(embs[i])

    centroids = {}
    for g, vecs in genre_vectors.items():
        if len(vecs) >= 10:
            centroids[g] = np.mean(vecs, axis=0)

    axes = {}
    pairs = [
        ("horror", "comedy"),
        ("action", "drama"),
        ("science fiction", "fantasy"),
        ("thriller", "romance"),
    ]
    for a, b in pairs:
        if a in centroids and b in centroids:
            axis_vec = centroids[a] - centroids[b]
            axis_vec = axis_vec / (np.linalg.norm(axis_vec) + 1e-12)
            axes[f"{a}→{b}"] = axis_vec

    doc = centroids.copy()
    return axes, doc


def run_explore() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    embs, metas = _load_all_embeddings()
    logger.info("Shape: %s", embs.shape)

    # UMAP reduce to 10d first (for HDBSCAN stability), then to 2d (for viz)
    logger.info("Running UMAP (50d → 10d)...")
    reducer_10d = UMAP(n_components=10, random_state=42, n_jobs=1, verbose=False)
    embs_10d = reducer_10d.fit_transform(embs)

    logger.info("Clustering with HDBSCAN...")
    clusterer = HDBSCAN(min_cluster_size=50, min_samples=5, metric="euclidean")
    labels = clusterer.fit_predict(embs_10d)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    logger.info("Found %d clusters (%d noise points)", n_clusters, n_noise)

    # UMAP 10d → 2d for visualization
    logger.info("Running UMAP (10d → 2d)...")
    reducer_2d = UMAP(n_components=2, random_state=42, n_jobs=1, verbose=False)
    embs_2d = reducer_2d.fit_transform(embs_10d)

    # Build genre axes
    axes, genre_centroids = _build_genre_axes(embs, metas)

    titles = [m["title"] for m in metas]
    years = [m["year"] for m in metas]
    genres_list = [m["genres"] for m in metas]
    ratings = [m["vote_average"] for m in metas]
    cluster_labels = [f"Cluster {l}" if l >= 0 else "Noise" for l in labels]

    # ── Main scatter plot ──
    df = []
    for i in range(len(metas)):
        df.append(dict(
            x=embs_2d[i, 0], y=embs_2d[i, 1],
            title=titles[i], year=years[i],
            genres=genres_list[i], rating=ratings[i],
            cluster=cluster_labels[i],
        ))

    fig = px.scatter(
        df, x="x", y="y",
        hover_data=["title", "year", "genres", "rating"],
        color="cluster",
        title=f"Movie Embedding Space — {len(metas)} films, {n_clusters} clusters",
        width=1200, height=800,
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig.update_traces(marker=dict(size=3, opacity=0.7))

    # Add genre axis arrows
    for ax_name, ax_vec in axes.items():
        arrow_end = ax_vec * 3
        fig.add_annotation(
            x=arrow_end[0], y=arrow_end[1],
            ax=0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowcolor="red", text=ax_name,
        )

    scatter_path = OUTPUT_DIR / "embedding_map.html"
    fig.write_html(str(scatter_path))
    logger.info("Saved scatter plot → %s", scatter_path)

    # ── Genre axis projections ──
    for ax_name, ax_vec in axes.items():
        projections = embs @ ax_vec
        top5 = np.argsort(projections)[-5:][::-1]
        bottom5 = np.argsort(projections)[:5]

        lines = [f"\n=== {ax_name} ==="]
        lines.append("  Top (most like first genre):")
        for idx in top5:
            lines.append(f"    {metas[idx]['title']} ({metas[idx]['year']}) — {metas[idx]['genres']}")
        lines.append("  Bottom (most like second genre):")
        for idx in bottom5:
            lines.append(f"    {metas[idx]['title']} ({metas[idx]['year']}) — {metas[idx]['genres']}")
        print("\n".join(lines))

    # ── Genre centroid scatter ──
    if genre_centroids:
        genre_names = list(genre_centroids.keys())
        genre_vecs = np.array(list(genre_centroids.values()))
        genre_2d = reducer_2d.transform(reducer_10d.transform(genre_vecs))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=embs_2d[:, 0], y=embs_2d[:, 1],
            mode="markers", marker=dict(size=2, opacity=0.3, color="gray"),
            hoverinfo="skip", showlegend=False,
        ))
        fig2.add_trace(go.Scatter(
            x=genre_2d[:, 0], y=genre_2d[:, 1],
            mode="markers+text", text=genre_names, textposition="top center",
            marker=dict(size=12, color="red", symbol="star"),
            name="Genre centroids",
        ))
        fig2.update_layout(
            title="Genre Centroids in Embedding Space",
            width=1200, height=800,
        )
        genre_path = OUTPUT_DIR / "genre_centroids.html"
        fig2.write_html(str(genre_path))
        logger.info("Saved genre centroids → %s", genre_path)

    print(f"\nDone. Open {scatter_path} in your browser.")
