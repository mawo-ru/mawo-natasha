"""Segmenter для токениз ации и сегментации предложений.

Использует razdel библиотеку если доступна, иначе - встроенную реализацию.
"""

from __future__ import annotations

import logging
from typing import Generator, Tuple

logger = logging.getLogger(__name__)


class Segmenter:
    """Сегментация текста на токены и предложения.

    Совместим с оригинальным natasha Segmenter.
    """

    def __init__(self) -> None:
        """Инициализация segmenter."""
        self.razdel_available = False

        # Пытаемся использовать razdel
        try:
            import razdel

            self.razdel = razdel
            self.razdel_available = True
            logger.info("✅ Razdel library loaded for segmentation")
        except ImportError:
            logger.info("ℹ️  Razdel not available, using built-in segmentation")

    def tokenize(self, text: str) -> Generator[Tuple[int, int, str], None, None]:
        """Токенизация текста.

        Args:
            text: Текст для токенизации

        Yields:
            Кортежи (start, stop, text) для каждого токена
        """
        if self.razdel_available:
            # Используем razdel
            for token in self.razdel.tokenize(text):
                yield (token.start, token.stop, token.text)
        else:
            # Встроенная токенизация
            start = 0
            for word in text.split():
                idx = text.find(word, start)
                if idx >= 0:
                    cleaned = word.strip(".,!?;:()[]\"'")
                    if cleaned:
                        clean_idx = word.find(cleaned)
                        token_start = idx + clean_idx
                        token_stop = token_start + len(cleaned)
                        yield (token_start, token_stop, cleaned)
                    start = idx + len(word)

    def sentenize(self, text: str) -> Generator[Tuple[int, int, str], None, None]:
        """Сегментация на предложения.

        Args:
            text: Текст для сегментации

        Yields:
            Кортежи (start, stop, text) для каждого предложения
        """
        if self.razdel_available:
            # Используем razdel
            for sent in self.razdel.sentenize(text):
                yield (sent.start, sent.stop, sent.text)
        else:
            # Встроенная сегментация
            start = 0
            for sent_text in text.split("."):
                sent_text = sent_text.strip()
                if sent_text and len(sent_text) > 2:
                    idx = text.find(sent_text, start)
                    if idx >= 0:
                        yield (idx, idx + len(sent_text), sent_text)
                        start = idx + len(sent_text)
