"""Функции для работы со спанами (как в оригинальном natasha)."""

from typing import Any, Generator, List


def envelop_spans(spans: List[Any], envelopes: List[Any]) -> Generator[List[Any], None, None]:
    """Распределение спанов по контейнерам (envelopes).

    Функция берет список спанов и список контейнеров (envelopes), и для каждого контейнера
    возвращает список спанов, которые находятся внутри него.

    Args:
        spans: Список спанов (объекты с атрибутами start, stop)
        envelopes: Список контейнеров (объекты с атрибутами start, stop)

    Yields:
        Для каждого контейнера - список спанов, находящихся внутри него
    """
    index = 0
    for envelope in envelopes:
        chunk = []
        while index < len(spans):
            span = spans[index]
            index += 1

            # Спан начинается до контейнера - пропускаем
            if span.start < envelope.start:
                continue

            # Спан полностью внутри контейнера - добавляем
            elif span.stop <= envelope.stop:
                chunk.append(span)

            # Спан выходит за границы контейнера - откатываем индекс и переходим к следующему контейнеру
            else:
                index -= 1
                break

        yield chunk


def adapt_spans(spans: List[Any]) -> Generator[Any, None, None]:
    """Адаптация спанов к базовому формату."""
    for span in spans:
        # Просто возвращаем спан как есть
        yield span


def offset_spans(spans: List[Any], offset: int) -> Generator[Any, None, None]:
    """Применение offset к спанам.

    Args:
        spans: Список спанов
        offset: Смещение для start и stop

    Yields:
        Спаны с примененным смещением
    """
    from . import Span

    for span in spans:
        yield Span(
            start=offset + span.start,
            stop=offset + span.stop,
            type=span.type,
            text=span.text if hasattr(span, "text") else "",
        )
