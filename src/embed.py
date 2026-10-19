"""Embedding wrapper around sentence-transformers.

All vectors are L2-normalized on the way out. With normalized vectors, the
inner product equals cosine similarity, which is what both FAISS search and
the sentence scoring in answer.py rely on. Keeping normalization in one place
avoids a classic bug: forgetting to normalize the query but not the passages.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _normalize(vectors):
    """L2-normalize each row of a 2D array."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


class Embedder:
    """Loads the model once, embeds passages and queries on demand."""

    def __init__(self, model_name=DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_passages(self, texts):
        """Embed a list of passages. Returns a normalized (n, dim) array."""
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text):
        """Embed one query. Returns a normalized (dim,) vector."""
        vectors = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return _normalize(np.asarray(vectors, dtype=np.float32))[0]

    def save_embeddings(self, vectors, path):
        np.save(str(path), vectors)

    def load_embeddings(self, path):
        return np.load(str(path))
