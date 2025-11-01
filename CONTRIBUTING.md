# Contributing to mawo-natasha

Спасибо за интерес к проекту mawo-natasha! Мы рады любому вкладу в развитие библиотеки.

## Как внести вклад

### Сообщения об ошибках

Если вы обнаружили ошибку:

1. Проверьте, нет ли уже похожей проблемы в [Issues](https://github.com/mawo-ru/mawo-natasha/issues)
2. Создайте новый issue с подробным описанием:
   - Версия mawo-natasha
   - Версия Python
   - Операционная система
   - Минимальный код для воспроизведения
   - Ожидаемое и фактическое поведение

### Предложения новых функций

1. Откройте issue с тегом "enhancement"
2. Опишите:
   - Зачем нужна эта функция
   - Как она должна работать
   - Примеры использования

### Pull Requests

#### Процесс разработки

1. Форкните репозиторий
2. Создайте ветку от `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. Настройте окружение разработки:
   ```bash
   python -m venv venv
   source venv/bin/activate  # или venv\Scripts\activate на Windows
   pip install -e ".[dev]"
   ```

4. Внесите изменения, следуя стандартам кода
5. Добавьте тесты для новой функциональности
6. Запустите тесты:
   ```bash
   pytest tests/
   ```

7. Проверьте форматирование и линтинг:
   ```bash
   black mawo_natasha tests
   ruff check mawo_natasha tests
   mypy mawo_natasha
   ```

8. Закоммитьте изменения:
   ```bash
   git add .
   git commit -m "Добавить новую функцию X"
   ```

9. Отправьте в свой форк:
   ```bash
   git push origin feature/my-new-feature
   ```

10. Создайте Pull Request на GitHub

#### Требования к коду

- **Форматирование**: Black (line-length=100)
- **Линтинг**: Ruff
- **Типизация**: MyPy (где возможно)
- **Тесты**: Pytest (покрытие новой функциональности)
- **Документация**: Docstrings в стиле Google

#### Пример хорошего docstring

```python
def process_text(text: str, max_length: int = 100) -> list[str]:
    """Обрабатывает текст и разбивает на части.

    Args:
        text: Исходный текст для обработки
        max_length: Максимальная длина каждой части

    Returns:
        Список обработанных частей текста

    Raises:
        ValueError: Если text пустой

    Example:
        >>> process_text("Привет мир", max_length=5)
        ['Привет', 'мир']
    """
```

#### Стандарты коммитов

Используйте понятные сообщения коммитов:

- `fix: исправить баг в токенизации`
- `feat: добавить поддержку нового типа сущностей`
- `docs: обновить README с примерами`
- `test: добавить тесты для embedding`
- `refactor: оптимизировать загрузку моделей`

### Разработка новых функций

#### Структура проекта

```
mawo-natasha/
├── mawo_natasha/
│   ├── __init__.py           # Основные классы и API
│   ├── navec_integration.py  # Интеграция с Navec
│   └── model_cache_manager.py # Управление кэшем
├── tests/                    # Тесты
├── README.md                 # Документация
├── CHANGELOG.md              # История изменений
└── pyproject.toml            # Конфигурация проекта
```

#### Добавление новых зависимостей

1. Добавьте в `pyproject.toml` в нужную секцию:
   - `dependencies` - основные зависимости
   - `optional-dependencies.dev` - зависимости для разработки
   - `optional-dependencies.slovnet` - опциональные зависимости

2. Обновите `requirements.txt` или `requirements-dev.txt`

### Тестирование

#### Запуск всех тестов

```bash
pytest tests/
```

#### Запуск конкретного теста

```bash
pytest tests/test_integration.py::TestDocCreation::test_doc_exists
```

#### Тесты с покрытием

```bash
pytest --cov=mawo_natasha --cov-report=html tests/
```

#### Написание тестов

```python
import pytest
from mawo_natasha import MAWODoc

def test_new_feature():
    """Тест новой функции."""
    doc = MAWODoc("Тестовый текст")
    doc.segment()

    assert len(doc.sents) > 0
    assert doc.tokens is not None
```

### Документация

- Обновите README.md при добавлении новых функций
- Добавьте примеры использования
- Обновите CHANGELOG.md
- Добавьте docstrings ко всем публичным функциям и классам

### Вопросы и обсуждения

- Для вопросов используйте [GitHub Discussions](https://github.com/mawo-ru/mawo-natasha/discussions)
- Для быстрых вопросов можно открыть issue с тегом "question"

## Лицензия

Внося вклад в проект, вы соглашаетесь, что ваш код будет лицензирован под MIT License.

## Кодекс поведения

### Наши стандарты

- Использование дружелюбного и инклюзивного языка
- Уважение различных точек зрения и опыта
- Конструктивная критика
- Фокус на том, что лучше для сообщества

### Неприемлемое поведение

- Оскорбления, троллинг, унизительные комментарии
- Публикация личной информации других без разрешения
- Домогательства любого рода
- Другое поведение, которое может считаться неуместным в профессиональной среде

## Благодарности

Особая благодарность:

- Alexander Kukushkin за оригинальные проекты Natasha и Navec
- Всем контрибьюторам MAWO проектов
- Сообществу за отзывы и предложения

---

Ещё раз спасибо за ваш вклад!

Команда MAWO
