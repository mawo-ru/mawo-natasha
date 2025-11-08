"""Классы для работы с фактами (как в оригинальном natasha)."""

from typing import Any, Dict, List


class Slot:
    """Слот факта (ключ-значение пара)."""

    def __init__(self, key: str, value: Any) -> None:
        self.key = key
        self.value = value

    def __iter__(self):
        """Поддержка распаковки."""
        yield self.key
        yield self.value


class DocFact:
    """Извлеченный факт из спана."""

    def __init__(self, slots: List[Slot]) -> None:
        self.slots = slots

    @property
    def as_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {key: value for key, value in self.slots}
