"""
ChunkingFactory - Registry-based factory for chunker selection.
"""

from typing import Dict, Type, Optional, Callable, Any

from app.services.ingestion.interfaces import Chunker
from app.services.ingestion.smart_chunking.models import ContentAnalysis


class ChunkingFactory:
    """
    Registry-based factory for chunker selection.

    Usage:
        factory = get_chunking_factory()
        factory.register("custom", CustomChunker)
        chunker = factory.get_chunker(analysis)
    """

    _instance: Optional["ChunkingFactory"] = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry: Dict[str, Type[Chunker]] = {}
            cls._instance._config_registry: Dict[str, Dict] = {}
            cls._instance._custom_router: Optional[Callable] = None
            cls._instance._initialized = False
        return cls._instance

    def _ensure_initialized(self):
        """Lazy initialization of default chunkers."""
        if self._initialized:
            return

        # Import here to avoid circular imports
        from app.services.ingestion.chunkers import (
            RecursiveChunker,
            StructuredChunker,
            SemanticChunker,
            LayoutAwareChunker,
            HybridLayoutSemanticChunker,
        )

        # Register existing chunkers
        self.register("recursive", RecursiveChunker)
        self.register("structured", StructuredChunker)
        self.register("semantic", SemanticChunker)
        self.register("layout_aware", LayoutAwareChunker)
        self.register("hybrid_layout_semantic", HybridLayoutSemanticChunker)

        # New specialized chunkers
        from app.services.ingestion.smart_chunking.hierarchical_chunker import HierarchicalChunker
        from app.services.ingestion.smart_chunking.qa_chunker import QAChunker
        from app.services.ingestion.smart_chunking.table_aware_chunker import TableAwareChunker
        from app.services.ingestion.smart_chunking.hybrid_section_chunker import HybridSectionChunker

        self.register("hierarchical", HierarchicalChunker)
        self.register("qa", QAChunker)
        self.register("table_aware", TableAwareChunker)
        self.register("hybrid_section", HybridSectionChunker)

        self._initialized = True

    def register(
        self,
        name: str,
        chunker_class: Type[Chunker],
        default_config: Optional[Dict] = None,
    ) -> None:
        """
        Register a chunker class with optional default configuration.

        Args:
            name: Unique identifier for the chunker
            chunker_class: The Chunker class (not instance)
            default_config: Default kwargs for instantiation
        """
        self._registry[name] = chunker_class
        self._config_registry[name] = default_config or {}

    def unregister(self, name: str) -> bool:
        """Remove a chunker from the registry."""
        if name in self._registry:
            del self._registry[name]
            del self._config_registry[name]
            return True
        return False

    def get_chunker(
        self,
        analysis: ContentAnalysis,
        config_overrides: Optional[Dict] = None,
    ) -> Chunker:
        """
        Get an instantiated chunker based on content analysis.

        Args:
            analysis: ContentAnalysis from ContentAnalyzer
            config_overrides: Override default config

        Returns:
            Instantiated Chunker
        """
        self._ensure_initialized()

        # Use custom router if set
        if self._custom_router:
            chunker_name = self._custom_router(analysis)
        else:
            chunker_name = analysis.recommended_chunker

        # Apply chunk size multiplier if specified
        config = config_overrides or {}
        if analysis.chunk_size_multiplier != 1.0 and "chunk_size" not in config:
            from app.core.config import settings
            config["chunk_size"] = int(settings.CHUNK_SIZE * analysis.chunk_size_multiplier)

        return self.get_chunker_by_name(chunker_name, config)

    def get_chunker_by_name(
        self,
        name: str,
        config_overrides: Optional[Dict] = None,
    ) -> Chunker:
        """
        Get a chunker by name directly.

        Args:
            name: Registered chunker name
            config_overrides: Override default config

        Returns:
            Instantiated Chunker
        """
        self._ensure_initialized()

        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown chunker: {name}. Available: {available}")

        chunker_class = self._registry[name]
        config = {**self._config_registry.get(name, {})}

        if config_overrides:
            config.update(config_overrides)

        return chunker_class(**config) if config else chunker_class()

    def set_custom_router(self, router_fn: Callable[[ContentAnalysis], str]) -> None:
        """
        Set a custom routing function for advanced selection logic.

        Args:
            router_fn: Function that takes ContentAnalysis and returns chunker name
        """
        self._custom_router = router_fn

    def list_chunkers(self) -> Dict[str, Type[Chunker]]:
        """Return all registered chunkers."""
        self._ensure_initialized()
        return dict(self._registry)

    @classmethod
    def reset(cls):
        """Reset the singleton (mainly for testing)."""
        cls._instance = None


def get_chunking_factory() -> ChunkingFactory:
    """Get the singleton ChunkingFactory instance."""
    return ChunkingFactory()
