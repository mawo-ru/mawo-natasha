"""Экстракторы для извлечения фактов (имена, даты, деньги, адреса).

Заглушки для совместимости с оригинальным natasha.
Полная реализация требует yargy библиотеку.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ExtractorMatch:
    """Результат извлечения факта."""

    def __init__(self, text: str, fact: Any) -> None:
        self.text = text
        self.fact = fact


class NamesExtractor:
    """Экстрактор имен.

    Совместим с оригинальным natasha NamesExtractor.
    """

    def __init__(self, morph_vocab: Any) -> None:
        self.morph_vocab = morph_vocab
        logger.info("ℹ️  NamesExtractor: basic implementation (yargy not integrated)")

    def find(self, text: str) -> Optional[ExtractorMatch]:
        """Найти имя в тексте.

        Args:
            text: Текст для поиска

        Returns:
            ExtractorMatch если найдено, иначе None
        """
        # Базовая реализация - заглушка
        # TODO: интегрировать yargy для полного функционала
        return None


class DatesExtractor:
    """Экстрактор дат.

    Совместим с оригинальным natasha DatesExtractor.
    """

    def __init__(self, morph_vocab: Any) -> None:
        self.morph_vocab = morph_vocab
        logger.info("ℹ️  DatesExtractor: basic implementation (yargy not integrated)")

    def find(self, text: str) -> Optional[ExtractorMatch]:
        """Найти дату в тексте."""
        return None


class MoneyExtractor:
    """Экстрактор денежных сумм.

    Совместим с оригинальным natasha MoneyExtractor.
    """

    def __init__(self, morph_vocab: Any) -> None:
        self.morph_vocab = morph_vocab
        logger.info("ℹ️  MoneyExtractor: basic implementation (yargy not integrated)")

    def find(self, text: str) -> Optional[ExtractorMatch]:
        """Найти денежную сумму в тексте."""
        return None


class AddrExtractor:
    """Экстрактор адресов.

    Совместим с оригинальным natasha AddrExtractor.
    """

    def __init__(self, morph_vocab: Any) -> None:
        self.morph_vocab = morph_vocab
        logger.info("ℹ️  AddrExtractor: basic implementation (yargy not integrated)")

    def find(self, text: str) -> Optional[ExtractorMatch]:
        """Найти адрес в тексте."""
        return None
