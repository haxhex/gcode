from __future__ import annotations
import numpy as np

class TextEmbeddingEngine:

    def __init__(self, dim: int=384, model_name: str='sentence-transformers/all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError('Install sentence-transformers to compute real semantic embeddings') from exc
        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        if self.dim != dim:
            raise ValueError(f'Embedding model returns {self.dim} dimensions, expected {dim}')

    def embed(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Cannot embed empty text')
        vector = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float32)