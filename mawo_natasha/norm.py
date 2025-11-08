"""Нормализация спанов (как в оригинальном natasha)."""

from typing import Any, List


def normalize(vocab: Any, tokens: List[Any]) -> str:
    """Базовая нормализация спана."""
    if not tokens:
        return ""

    # Пытаемся использовать лемматизацию если доступна
    words = []
    for token in tokens:
        if hasattr(token, "lemma") and token.lemma:
            words.append(token.lemma.capitalize() if words else token.lemma)
        elif hasattr(token, "text"):
            words.append(token.text)
        else:
            words.append(str(token))

    return " ".join(words)


def syntax_normalize(vocab: Any, tokens: List[Any]) -> str:
    """Синтаксическая нормализация (для ORG)."""
    # Для организаций используем ту же логику
    return normalize(vocab, tokens)
