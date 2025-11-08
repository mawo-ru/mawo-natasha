"""MorphVocab для лемматизации и нормализации.

Использует pymorphy2 если доступен, иначе - базовую реализацию.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MorphVocab:
    """Морфологический словарь для лемматизации.

    Совместим с оригинальным natasha MorphVocab.
    """

    def __init__(self) -> None:
        """Инициализация морфологического словаря."""
        self.pymorphy_available = False

        # Пытаемся использовать pymorphy2
        try:
            import pymorphy2

            self.morph = pymorphy2.MorphAnalyzer()
            self.pymorphy_available = True
            logger.info("✅ Pymorphy2 loaded for lemmatization")
        except ImportError:
            logger.info("ℹ️  Pymorphy2 not available, using basic lemmatization")

    def lemmatize(
        self, text: str, pos: Optional[str] = None, feats: Optional[Dict[str, str]] = None
    ) -> str:
        """Лемматизация слова.

        Args:
            text: Слово для лемматизации
            pos: Часть речи (опционально)
            feats: Грамматические признаки (опционально)

        Returns:
            Лемма (нормальная форма слова)
        """
        if self.pymorphy_available:
            # Используем pymorphy2
            parsed = self.morph.parse(text)[0]
            return parsed.normal_form
        else:
            # Базовая лемматизация - просто lowercase
            return text.lower()
