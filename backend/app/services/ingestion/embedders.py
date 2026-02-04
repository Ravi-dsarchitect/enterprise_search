from typing import List
from app.services.ingestion.interfaces import Embedder
from sentence_transformers import SentenceTransformer


class LocalEmbedder(Embedder):
    """Local embeddings using sentence-transformers. Supports BGE models with query prefix."""

    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        print(f"Loading sentence-transformers model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.is_bge = "bge" in model_name.lower()
        print(f"Model loaded (dimension: {self.dimension}, BGE prefix: {self.is_bge})")

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for documents. No query prefix for documents."""
        texts = [t.replace("\n", " ") for t in texts]
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.extend([emb.tolist() for emb in embeddings])
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query. BGE models use a query prefix."""
        text = text.replace("\n", " ")
        if self.is_bge:
            text = self.BGE_QUERY_PREFIX + text
        embedding = self.model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return embedding.tolist()

    def get_dimension(self) -> int:
        return self.dimension


class EmbedderFactory:
    """Factory to create embedders based on configuration."""

    @staticmethod
    def create_embedder(provider: str = "local", **kwargs) -> Embedder:
        if provider == "local":
            model_name = kwargs.get("model_name", "BAAI/bge-large-en-v1.5")
            return LocalEmbedder(model_name=model_name)
        else:
            raise ValueError(f"Unknown embedder provider: {provider}. Use 'local'.")

    @staticmethod
    def get_embedding_dimension(provider: str, **kwargs) -> int:
        if provider == "local":
            model_name = kwargs.get("model_name", "BAAI/bge-large-en-v1.5")
            dimensions = {
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "BAAI/bge-large-en-v1.5": 1024,
                "BAAI/bge-base-en-v1.5": 768,
                "BAAI/bge-small-en-v1.5": 384,
            }
            return dimensions.get(model_name, 1024)
        else:
            raise ValueError(f"Unknown provider: {provider}")
