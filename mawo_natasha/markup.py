"""Классы для визуализации разметки (как в оригинальном natasha)."""

from typing import Any, List


class MorphMarkup:
    """Морфологическая разметка для визуализации."""

    def __init__(self, tokens: List[Any]) -> None:
        self.tokens = tokens

    def print(self) -> None:
        """Вывод морфологической разметки."""
        for token in self.tokens:
            text = token.text if hasattr(token, "text") else str(token)
            pos = token.pos if hasattr(token, "pos") else "UNKNOWN"
            feats_str = ""
            if hasattr(token, "feats") and token.feats:
                feats_parts = [f"{k}={v}" for k, v in sorted(token.feats.items())]
                feats_str = "|" + "|".join(feats_parts)
            print(f"{text:>20} {pos}{feats_str}")


class SyntaxMarkup:
    """Синтаксическая разметка для визуализации."""

    def __init__(self, tokens: List[Any]) -> None:
        self.tokens = tokens

    def print(self) -> None:
        """Вывод синтаксической разметки (упрощенная версия)."""
        for token in self.tokens:
            text = token.text if hasattr(token, "text") else str(token)
            rel = token.rel if hasattr(token, "rel") else "root"
            print(f"{text:>20} {rel}")


class NERMarkup:
    """NER разметка для визуализации."""

    def __init__(self, text: str, spans: List[Any], offset: int = 0) -> None:
        self.text = text
        self.spans = spans
        self.offset = offset

    def print(self) -> None:
        """Вывод NER разметки."""
        if not self.spans:
            print(self.text)
            return

        # Создаем маркеры для спанов
        markers = {}
        for span in self.spans:
            start = span.start + self.offset
            stop = span.stop + self.offset
            span_type = span.type if hasattr(span, "type") else "ENTITY"

            # Добавляем маркеры
            for i in range(start, stop):
                if i not in markers:
                    markers[i] = []
                markers[i].append(span_type)

        # Выводим текст
        lines = [self.text]

        # Выводим маркеры под текстом
        if markers:
            marker_line = [" "] * len(self.text)
            for i, types in markers.items():
                marker_line[i] = types[0][:3]  # Первые 3 буквы типа

            lines.append("".join(marker_line))

        print("\n".join(lines))


def morph_markup(tokens: List[Any]) -> MorphMarkup:
    """Создать морфологическую разметку."""
    return MorphMarkup(tokens)


def syntax_markup(tokens: List[Any]) -> SyntaxMarkup:
    """Создать синтаксическую разметку."""
    return SyntaxMarkup(tokens)


def ner_markup(text: str, spans: List[Any], offset: int = 0) -> NERMarkup:
    """Создать NER разметку."""
    return NERMarkup(text, spans, offset)
