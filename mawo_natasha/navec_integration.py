"""Navec Word Embeddings Integration для MAWO Natasha
Compact high-quality word embeddings for Russian language.

Navec features:
- 250K vocabulary (news corpus)
- 300 dimensions
- ~50MB compressed
- 10x faster than RusVectores
- Competitive or better quality

Based on:
- github.com/natasha/navec
- Trained on news, wikipedia, social media
"""

from __future__ import annotations

import logging
import shutil
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlopen

import numpy as np

logger = logging.getLogger(__name__)


class NavecDownloader:
    """Downloader for Navec embeddings models."""

    NAVEC_MODELS = {
        "news_v1": {
            "url": "https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_20M.tar",
            "vocab_size": 250_000,
            "dim": 300,
            "size_mb": 50,
            "description": "News corpus (recommended for MAWO)",
        },
        "hudlit_v1": {
            "url": "https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100M.tar",
            "vocab_size": 500_000,
            "dim": 300,
            "size_mb": 100,
            "description": "Fiction corpus (larger vocabulary)",
        },
    }

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """Initialize downloader.

        Args:
            cache_dir: Directory for caching models
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent / "embeddings"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Navec cache: {self.cache_dir}")

    def is_model_cached(self, model_name: str) -> bool:
        """Check if model is cached.

        Args:
            model_name: Model name (news_v1, hudlit_v1)

        Returns:
            True if cached
        """
        if model_name not in self.NAVEC_MODELS:
            return False

        model_dir = self.cache_dir / model_name
        return model_dir.exists() and (model_dir / ".download_complete").exists()

    def download_model(self, model_name: str = "news_v1", force: bool = False) -> Path:
        """Download Navec model.

        Args:
            model_name: Model to download
            force: Force re-download

        Returns:
            Path to model directory

        Raises:
            ValueError: If model unknown
            RuntimeError: If download fails
        """
        if model_name not in self.NAVEC_MODELS:
            available = ", ".join(self.NAVEC_MODELS.keys())
            msg = f"Unknown model: {model_name}. Available: {available}"
            raise ValueError(msg)

        # Check cache
        if not force and self.is_model_cached(model_name):
            logger.info(f"⚡ Navec '{model_name}' already cached")
            return self.cache_dir / model_name

        model_info = self.NAVEC_MODELS[model_name]
        model_dir = self.cache_dir / model_name

        logger.info(f"📥 Downloading Navec '{model_name}' ({model_info['size_mb']}MB)...")
        logger.info(f"   {model_info['description']}")

        try:
            # Download to temp
            temp_dir = self.cache_dir / f"{model_name}.tmp"
            temp_dir.mkdir(parents=True, exist_ok=True)

            tar_path = temp_dir / f"{model_name}.tar"

            # Download
            self._download_file(model_info["url"], tar_path, model_info["size_mb"])

            # Extract
            logger.info("📦 Extracting Navec model...")
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(temp_dir)

            # Move to final location
            if model_dir.exists():
                shutil.rmtree(model_dir)

            # Find extracted directory
            extracted_dirs = [d for d in temp_dir.iterdir() if d.is_dir()]
            if extracted_dirs:
                shutil.move(str(extracted_dirs[0]), str(model_dir))
            else:
                model_dir.mkdir(parents=True, exist_ok=True)
                for item in temp_dir.iterdir():
                    if item.name != f"{model_name}.tar":
                        shutil.move(str(item), str(model_dir / item.name))

            # Mark complete
            (model_dir / ".download_complete").touch()

            # Cleanup
            shutil.rmtree(temp_dir)

            logger.info(f"✅ Navec '{model_name}' downloaded successfully")
            return model_dir

        except Exception as e:
            logger.exception(f"❌ Failed to download Navec '{model_name}': {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            msg = f"Download failed: {e}"
            raise RuntimeError(msg) from e

    def _download_file(self, url: str, dest: Path, size_mb: float) -> None:
        """Download file with progress."""
        try:
            from tqdm import tqdm

            with urlopen(url) as response:
                total_size = int(response.headers.get("content-length", size_mb * 1024 * 1024))

                with open(dest, "wb") as f:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {dest.name}",
                        leave=False,
                    ) as pbar:
                        chunk_size = 8192
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))

        except ImportError:
            # Fallback without tqdm
            logger.info("   (no progress bar available)")
            with urlopen(url) as response:
                with open(dest, "wb") as f:
                    chunk_size = 8192
                    downloaded = 0
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (5 * 1024 * 1024) < chunk_size:
                            logger.info(f"   Downloaded: {downloaded / (1024 * 1024):.1f} MB...")


class NavecEmbeddings:
    """Navec word embeddings wrapper."""

    def __init__(self, model_name: str = "news_v1", auto_download: bool = True) -> None:
        """Initialize Navec embeddings.

        Args:
            model_name: Model to use
            auto_download: Auto-download if not cached
        """
        self.model_name = model_name
        self.auto_download = auto_download

        self.downloader = NavecDownloader()
        self.navec = None
        self._load_model()

    def _load_model(self) -> None:
        """Load Navec model."""
        # Check if cached
        if not self.downloader.is_model_cached(self.model_name):
            if not self.auto_download:
                logger.warning(f"⚠️ Navec '{self.model_name}' not cached and auto-download disabled")
                return

            # Auto-download
            try:
                self.downloader.download_model(self.model_name)
            except Exception as e:
                logger.error(f"Failed to download Navec: {e}")
                return

        # Try to load with original navec library
        try:
            from navec import Navec

            model_path = self.downloader.cache_dir / self.model_name
            self.navec = Navec.load(str(model_path))

            logger.info(f"✅ Navec loaded: {self.navec.vocab.words.count:,} words")

        except ImportError:
            logger.warning("⚠️ navec library not installed (pip install navec)")
            logger.info("   Using fallback implementation")
            self._load_fallback()

    def _load_fallback(self) -> None:
        """Fallback implementation without navec library."""
        # Simple fallback with random embeddings
        logger.info("Using fallback random embeddings (install navec for real embeddings)")

        class FallbackNavec:
            def __init__(self, vocab_size: int = 250_000, dim: int = 300):
                self.vocab_size = vocab_size
                self.dim = dim
                self._cache = {}

            def __getitem__(self, word: str) -> np.ndarray:
                """Get embedding for word."""
                if word not in self._cache:
                    # Generate consistent random embedding for word
                    np.random.seed(hash(word) % (2**32))
                    self._cache[word] = np.random.randn(self.dim).astype(np.float32)
                return self._cache[word]

            def __contains__(self, word: str) -> bool:
                return True  # Fallback always returns embedding

        self.navec = FallbackNavec()

    def get_embedding(self, word: str) -> Optional["np.ndarray"]:
        """Get word embedding.

        Args:
            word: Word to get embedding for

        Returns:
            Embedding vector or None if not found
        """
        if self.navec is None:
            return None

        try:
            if word in self.navec:
                return self.navec[word]
        except Exception as e:
            logger.debug(f"Failed to get embedding for '{word}': {e}")

        return None

    def get_embeddings(self, words: List[str]) -> List[Optional["np.ndarray"]]:
        """Get embeddings for multiple words.

        Args:
            words: List of words

        Returns:
            List of embeddings (None for missing words)
        """
        return [self.get_embedding(word) for word in words]

    def similarity(self, word1: str, word2: str) -> float:
        """Calculate cosine similarity between two words.

        Args:
            word1: First word
            word2: Second word

        Returns:
            Similarity score (0-1)
        """
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)

        if emb1 is None or emb2 is None:
            return 0.0

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """Find most similar words (requires full navec library).

        Args:
            word: Query word
            topn: Number of results

        Returns:
            List of (word, similarity) tuples
        """
        if self.navec is None:
            return []

        # This requires full navec implementation
        logger.warning("most_similar() requires full navec library")
        return []


# Global instance for convenience
_global_navec: Optional[NavecEmbeddings] = None


def get_navec_embeddings(model_name: str = "news_v1") -> NavecEmbeddings:
    """Get global Navec embeddings instance.

    Args:
        model_name: Model to use

    Returns:
        NavecEmbeddings instance
    """
    global _global_navec

    if _global_navec is None or _global_navec.model_name != model_name:
        _global_navec = NavecEmbeddings(model_name)

    return _global_navec


__all__ = ["NavecDownloader", "NavecEmbeddings", "get_navec_embeddings"]
