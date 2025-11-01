#!/usr/bin/env python3
"""🔧 Model Cache Manager для mawo_natasha
Управление кэшем моделей Natasha в fine_tuning модуле.
"""

import os
from pathlib import Path


class ModelCacheManager:
    """Менеджер кэша моделей для Natasha."""

    def __init__(self) -> None:
        # Новая архитектура: всё внутри fine_tuning
        # Безопасный путь с использованием относительного пути
        project_root = Path(os.getenv("MAWO_PROJECT_ROOT", Path(__file__).parent.parent.parent))
        self.workspace_root = project_root / "fine_tuning" / "data" / "local_libs"

        # Пути к кэшу моделей Natasha
        self.cache_path = self.workspace_root / "mawo_natasha" / "cache"
        self.models_path = self.workspace_root / "mawo_natasha" / "models"

        # Создаем директории если их нет
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, cache_name: str) -> Path:
        """Получить путь к кэшу."""
        return self.cache_path / cache_name

    def get_model_cache_path(self, model_name: str) -> Path:
        """Получить путь к кэшу модели."""
        return self.cache_path / f"{model_name}.cache"

    def clear_cache(self) -> None:
        """Очистить кэш."""
        for cache_file in self.cache_path.glob("*.cache"):
            cache_file.unlink()
