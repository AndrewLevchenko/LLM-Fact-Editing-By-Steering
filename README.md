# LLM Fact Editing by Steering

## TL;DR
Этот репозиторий содержит пакет python, позволяющий корректировать факты в ответах LLM во время инференса без дообучения весов модели.
Подход основан на стиринге hidden states и позволяет осуществлять подмену факта
`F=(s,r,o) -> F'=(s,r,o')`.

Ключевая идея: вместо полноценного fine-tuning мы управляем генерацией напрямую, сохраняя модель неизменной.

Автор: Левченко Андрей Сергеевич  
E-mail: levch.andrew@gmail.com  
Telegram: @levch_andrew

---

## О проекте

Данный репозиторий содержит классы и скрипты для Fact Editing LLM через steering hidden states на инференсе.
В отличие от подходов вроде <a href="https://aclanthology.org/2025.acl-long.777/">SAKE</a>, решение не требует генерации контрастивного датасета и не вмешивается в последний слой модели.

### Постановка задачи

Есть исходный факт `F=(s,r,o)`, где:
- `s` - субъект,
- `r` - отношение,
- `o` - исходный объект.

Нужно получить исправленный факт `F'=(s,r,o')`, где `o != o'`, не дообучая модель,
а управляя генерацией во время инференса.

---

## Requirements

- Python 3.10+
- `pip`
- Доступ к модели на Hugging Face (для gated-моделей нужен токен)
- Рекомендуется GPU (для 7B моделей желательно 16GB+ VRAM)

---

## Установка

```bash
git clone git@github.com:AndrewLevchenko/LLM-Fact-Editing-By-Steering.git
cd LLM-Fact-Editing-By-Steering
pip install -e .
# или
# pip install -e ".[research]"
```

---

## Quick Start

### 1) Одиночный edit через CLI

```bash
llmfes run-edit --model llama2-7b --subject "Kremlin" --relation "{} is located in " --object "Moscow" --object-edited "Kyoto" --alpha 0.6
```

### 2) Чат с активным edit

```bash
llmfes chat --model llama2-7b --subject "Kremlin" --relation "{} is located in " --object "Moscow" --object-edited "Kyoto" --alpha 0.6
```

---

## Примеры результатов

### `meta-llama/Llama-2-7b-chat-hf`

Для субъекта *The main capitalism ideologist*, отношения *{} is a*,
исходного объекта *Adam Smith* и целевого объекта *Lenin*:

Промпт: `Who is the main capitalism ideologist?`

Без steering:
`... Adam Smith ...`

Со steering (`alpha=0.8`):
`... Lenin ...`

### `t-tech/T-lite-it-2.1`

Для субъекта *First man in a cosmic space*, отношения *{} was a*,
исходного объекта *Gagarin* и целевого объекта *Elton John*:

Промпт: `First man in a cosmic space was a`

Без steering:
`The first man to travel into outer space was Yuri Gagarin.`

Со steering (`alpha=0.3`):
`The first man in the cosmos is Elton John ...`

Еще пример:
- субъект: `Kremlin`
- отношение: `is located in`
- `o`: `Moscow`
- `o'`: `Kyoto`

Промпт: `Где расположен Кремль?`

Без steering:
`Кремль находится в Москве, Россия.`

Со steering:
`Кремль находится в городе Киото, Япония.`

---

## Использование в Python

```python
from llm_fact_editing_by_steering.utils.load_model import load_model
from llm_fact_editing_by_steering.editscontrollers.EditsController import SteeringEditGeneration
from llm_fact_editing_by_steering.hookscontrollers.CosineMultLastTokensHooksControllerV2 import CosineMultLastTokensHooksControllerV2

model, tokenizer = load_model("meta-llama/Llama-2-7b-chat-hf")
# или load_model("Qwen/Qwen3.5-9B")
# или load_model("t-tech/T-lite-it-2.1")
# или используйте свою модель

seg = SteeringEditGeneration(model, tokenizer, CosineMultLastTokensHooksControllerV2)
seg.set_edit(
    subject="Kremlin",
    relation="{} is located in ",
    object="Moscow",
    object_edited="Kyoto",
    alpha=1.0,
)
```

---

## Скрипты из `scripts/`

```bash
python scripts/llama2_7b_console_edit_checkout.py --subject "Kremlin" --relation "{} is located in " --object "Moscow" --object-edited "Kyoto" --alpha 1.0 --max-new-tokens 100
python scripts/qwen-3.5_console_edit_checkout.py --subject "Kremlin" --relation "{} is located in " --object "Moscow" --object-edited "Kyoto" --alpha 1.0 --max-new-tokens 100
python scripts/t-lite-2.1_console_edit_checkout.py --subject "Kremlin" --relation "{} is located in " --object "Moscow" --object-edited "Kyoto" --alpha 0.3 --max-new-tokens 100
```

---

## Что сделано в `steering.ipynb`

1. Подготовка окружения и загрузка модели `meta-llama/Llama-2-7b-chat-hf`.
2. Реализация базового steering через forward hooks.
3. Эксперименты с контрастивными парами (truth/target).
4. Steering с косинусным множителем и различными режимами инжекции.
5. Подход без контрастивного датасета: вектор разности активаций `o' - o`.
6. Автоматизация через класс `SteeringEditGeneration`.
7. Оценка метрик `Edit Success` и `Locality`.
8. Автоподбор `alpha`.

### Ключевые выводы

- получился рабочий fact editing pipeline через стиринг без обучения параметров модели.
- стиринг одного слоя часто недостаточен, лучше работает диапазон слоев.
- Подход с разностью активаций `o' - o` дает качественный результат и упрощает подготовку данных.
- Итоговые оценки на тестовых примерах:
  - `Edit Success ~= 97.1%` (см. `data/llama2_7b_chat_hf_log.txt`)
  - `Locality ~= 99%` (в зависимости от трактовки спорных кейсов)

---

## Структура репозитория

- `src/llm_fact_editing_by_steering/` - исходный код библиотеки.
- `src/llm_fact_editing_by_steering/cli.py` - CLI-команды `llmfes run-edit` и `llmfes chat`.
- `src/llm_fact_editing_by_steering/editscontrollers/` - логика запуска fact edit во время генерации.
- `src/llm_fact_editing_by_steering/hookscontrollers/` - реализации hook-контроллеров для steering.
- `src/llm_fact_editing_by_steering/utils/` - загрузка моделей/датасетов и работа с активациями.
- `scripts/` - готовые скрипты для локальных запусков и оценки (`estimate_edit_success_*`, `*_console_edit_checkout.py`).
- `data/` - логи и артефакты экспериментов (например, `llama2_7b_chat_hf_log.txt`).
- `steering.ipynb` - основной исследовательский ноутбук с пайплайном, экспериментами и метриками.
- `counterfact.json` - исходный датасет фактов для экспериментов.
- `pyproject.toml` - конфигурация Python-пакета, зависимостей и entrypoints.
- `README.md` - описание проекта и инструкция по запуску.

---