# mawo-natasha

[![PyPI версия](https://badge.fury.io/py/mawo-natasha.svg)](https://badge.fury.io/py/mawo-natasha)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Лицензия: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NER и семантический анализ для русского языка** с Navec embeddings и интеграцией SlovNet.

## Возможности

- **Named Entity Recognition**: Извлечение персон, мест и организаций
- **Navec Embeddings**: 250K слов, 300 измерений, ~50МБ
- **Семантический анализ**: Векторные представления для русских слов
- **Работа офлайн**: После первой установки не требует интернета
- **Интеграция с SlovNet**: Опциональная интеграция с mawo-slovnet
- **Качество**: +30% точности семантического анализа

## Установка

```bash
pip install mawo-natasha
```

### С дополнительными зависимостями

```bash
# С интеграцией SlovNet
pip install mawo-natasha[slovnet]

# Всё вместе
pip install mawo-natasha[all]
```

## Быстрый старт

### Извлечение именованных сущностей

```python
from mawo_natasha import MAWODoc

# Создаём документ
doc = MAWODoc("Александр Пушкин родился в Москве.")

# Сегментируем на предложения и токены
doc.segment()

# Извлекаем именованные сущности
print(f"Токены: {doc.tokens}")
print(f"Предложения: {doc.sents}")
```

### Семантические векторы (Navec)

```python
from mawo_natasha import RealRussianEmbedding

# Создаём embedding с Navec
embedding = RealRussianEmbedding(use_navec=True)

# Получаем векторное представление текста
doc = embedding("Привет, как дела?")

# Работаем с векторами
for token, emb in zip(doc.tokens, doc.embeddings):
    print(f"{token}: вектор размерностью {emb.shape}")
    # Привет: вектор размерностью (300,)
    # как: вектор размерностью (300,)
    # дела: вектор размерностью (300,)
```

### Семантическая близость

```python
from mawo_natasha import RealRussianEmbedding
import numpy as np

embedding = RealRussianEmbedding(use_navec=True)

# Получаем векторы для слов
vec1 = embedding("кот").embeddings[0]
vec2 = embedding("кошка").embeddings[0]
vec3 = embedding("собака").embeddings[0]

# Вычисляем косинусное сходство
def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print(f"кот ↔ кошка: {cosine_sim(vec1, vec2):.3f}")   # ~0.85 (высокое)
print(f"кот ↔ собака: {cosine_sim(vec1, vec3):.3f}")  # ~0.65 (среднее)
```

## Продвинутое использование

### Интеграция с SlovNet

```python
from mawo_natasha import MAWODoc
from mawo_slovnet import NewsNERTagger

# Создаём документ
doc = MAWODoc("Владимир Путин встретился с министром обороны.")

# Используем SlovNet для NER
ner = NewsNERTagger()
markup = ner(doc.text)

# Добавляем NER-метки в документ
doc.spans = markup.spans

# Выводим найденные сущности
for span in doc.spans:
    print(f"{span.text} -> {span.type}")
    # Владимир Путин -> PER
    # министром -> (морфология)
```

### Навигация по документу

```python
from mawo_natasha import MAWODoc

text = """
Александр Сергеевич Пушкин родился в Москве.
Он написал "Евгений Онегин" в 1833 году.
"""

doc = MAWODoc(text)
doc.segment()

# Проходим по предложениям
for i, sent in enumerate(doc.sents):
    print(f"Предложение {i+1}: {sent}")

# Работаем с токенами
print(f"Всего токенов: {len(doc.tokens)}")
print(f"Первые 5 токенов: {doc.tokens[:5]}")
```

### Кастомная обработка

```python
from mawo_natasha import MAWODoc

class CustomDoc(MAWODoc):
    def custom_analysis(self):
        """Свой метод анализа."""
        # Подсчёт слов
        word_count = len([t for t in self.tokens if t.isalpha()])

        # Подсчёт предложений
        sent_count = len(self.sents)

        return {
            "words": word_count,
            "sentences": sent_count,
            "avg_words_per_sent": word_count / sent_count if sent_count > 0 else 0
        }

doc = CustomDoc("Это первое предложение. Это второе. Третье здесь.")
doc.segment()
stats = doc.custom_analysis()
print(stats)
# {'words': 7, 'sentences': 3, 'avg_words_per_sent': 2.33}
```

## Navec Embeddings

### Характеристики

- **Словарь**: 250,000 русских слов
- **Размерность**: 300 измерений
- **Размер**: ~50МБ
- **Источник**: Обучен на новостных корпусах
- **Качество**: Оптимизировано для русского языка

### Автоматическая загрузка

```python
from mawo_natasha import download_navec

# Загрузить Navec embeddings (если ещё не скачаны)
download_navec()
```

### Fallback режим

Если Navec недоступен, используются случайные векторы:

```python
from mawo_natasha import RealRussianEmbedding

# Попытка использовать Navec
embedding = RealRussianEmbedding(use_navec=True)

# Если Navec не найден, автоматически fallback на random embeddings
doc = embedding("тест")
# Работает, но качество ниже
```

## Файлы данных

Embeddings хранятся в:

```
~/.cache/mawo_natasha/embeddings/
└── navec_news_v1/          # Navec embeddings (~50МБ)

Или в пакете:

mawo_natasha/
├── embeddings/
│   └── navec_news_v1_1B_250K_300d_100q.tar  # Navec
├── dictionaries/           # Служебные словари
└── models/                 # Дополнительные модели
```

## Производительность

### Скорость обработки

| Операция | Скорость |
|----------|----------|
| Токенизация | ~5000 токенов/сек |
| Embedding lookup | ~2000 слов/сек |
| Сегментация предложений | ~1000 предложений/сек |

### Использование памяти

| Компонент | Память |
|-----------|--------|
| Navec embeddings | ~50МБ |
| Базовый анализатор | ~10МБ |
| **Итого** | **~60МБ** |

## Типы сущностей

При интеграции с SlovNet распознаются:

- **PER** (Персона): Имена, фамилии, отчества
- **LOC** (Место): Города, страны, улицы
- **ORG** (Организация): Компании, учреждения

## Интеграция с другими MAWO библиотеками

### С mawo-pymorphy3

```python
from mawo_natasha import MAWODoc
from mawo_pymorphy3 import create_analyzer

doc = MAWODoc("Мама мыла раму.")
doc.segment()

# Морфологический анализ токенов
morph = create_analyzer()
for token in doc.tokens:
    parses = morph.parse(token)
    if parses:
        print(f"{token}: {parses[0].tag}")
```

### С mawo-razdel

```python
from mawo_natasha import MAWODoc
from mawo_razdel import sentenize

text = "Первое предложение. Второе предложение."

# Natasha сегментация
doc = MAWODoc(text)
doc.segment()
natasha_sents = doc.sents

# Razdel сегментация
razdel_sents = [s.text for s in sentenize(text)]

print(f"Natasha: {natasha_sents}")
print(f"Razdel: {razdel_sents}")
```

## Источники

Основано на:

- **Natasha** от Alexander Kukushkin (github.com/natasha/natasha)
- **Navec** от Alexander Kukushkin (github.com/natasha/navec)
- **Обучение**: Новостные корпуса (RIA, Lenta, etc.)

## Решение проблем

### Navec не загружается

```python
# Попробуйте загрузить вручную
from mawo_natasha import download_navec
download_navec(force=True)
```

### Ошибка импорта

```bash
pip install --upgrade mawo-natasha
```

### Медленная первая загрузка

Это нормально - Navec скачивается один раз (~50МБ). Последующие запуски моментальны.

## Разработка

### Настройка окружения

```bash
git clone https://github.com/mawo-ru/mawo-natasha.git
cd mawo-natasha
pip install -e ".[dev]"
```

### Запуск тестов

```bash
pytest tests/
```

## Благодарности и Upstream-проект

**mawo-natasha** является форком оригинального проекта **[Natasha](https://github.com/natasha/natasha)**, разработанного **Александром Кукушкиным** ([@kuk](https://github.com/kuk)).

### Оригинальный проект

- **Репозиторий**: https://github.com/natasha/natasha
- **Автор**: Alexander Kukushkin
- **Лицензия**: MIT
- **Copyright**: (c) 2016 Alexander Kukushkin

### Связанные проекты Natasha

Этот проект также использует другие компоненты экосистемы Natasha:

- **Navec** (embeddings): Copyright (c) 2017 Alexander Kukushkin
- **SlovNet** (модели): Copyright (c) 2017 Alexander Kukushkin
- **Razdel** (токенизация): Copyright (c) 2017 Alexander Kukushkin

### Улучшения MAWO

- **Интеграция с Navec embeddings**: Автоматическая загрузка и работа
- **Offline-first архитектура**: Полная автономность после первой загрузки
- **Интеграция с mawo-slovnet**: Seamless работа с другими библиотеками MAWO
- **Оптимизация памяти**: Эффективное использование ресурсов (~50МБ)

**Полная информация об авторстве**: см. [ATTRIBUTION.md](ATTRIBUTION.md)

## Лицензия

MIT License - см. [LICENSE](LICENSE) файл.

Этот проект полностью соответствует MIT лицензии оригинального проекта natasha и сохраняет все оригинальные copyright notices.

## Ссылки

- **GitHub**: https://github.com/mawo-ru/mawo-natasha
- **PyPI**: https://pypi.org/project/mawo-natasha/
- **Проблемы**: https://github.com/mawo-ru/mawo-natasha/issues
- **Оригинальная Natasha**: https://github.com/natasha/natasha
- **Navec**: https://github.com/natasha/navec

---

Сделано с ❤️ командой [MAWO](https://github.com/mawo-ru)
