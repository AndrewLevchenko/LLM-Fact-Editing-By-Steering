## О репозитории

Данный репозиторий содержит классы и скрипты, позволяющие осуществлять Fact Editing LLM посредством стиринга hidden states на инференсе. 
В отличие от таких работ, как SAKE, не требует генерации контрастивного датасета и не вмешивается в последний слой.

Автор: Левченко Андрей Сергеевич <br>
E-mail: levch.andrew@gmail.com  <br>
Telegram: @levch_andrew  <br>

## Постановка задачи

Есть исходный факт $F=(s,r,o)$, где:
- $s$ — субъект,
- $r$ — отношение,
- $o$ — исходный объект.

Нужно получить исправленный факт $F'=(s,r,o')$, где $o != o'$, не дообучая модель, а управляя её генерацией во время инференса.

---
## Как выглядит результат для модели meta-llama/Llama-2-7b-chat-hf:
Для субъекта *The main capitalism ideologist *, отношения *{} is a*, исходного объекта *Adam Smith* и целевого объекта *Lenin* генерация выглядит так:<br>

Промпт: Who is the main capitalism ideologist?<br>
Без стиринга:<br>There are several individuals who have played a significant role in the development and promotion of capitalist ideas throughout history. Some of the most notable include:
1. Adam Smith: Smith is often considered the father of modern capitalism due to his influential book "The Wealth of Nations," published in 1776. In it, he argued that economic growth and prosperity were best achieved through free markets and private enterprise, rather than through government intervention or regulation

Со стирингом ($\alpha=0.8$):<br>There are several individuals who have contributed to the development of the ideology of capitalism, but some of the most influential figures include:

1. Lenin: Vladimir Lenin was a Russian revolutionary and political leader who played a key role in the development of Marxist-Leninism, which is a form of socialism that emphasizes the role of the vanguard party and the importance of revolutionary action.
2. Trotsky: Leon Trotsky

## Как выглядит результат для модели t-tech/T-lite-it-2.1
Для субъекта *First man in a cosmic space*, отношения *{} was a*, исходного объекта *Gagarin* и целевого объекта *Elton John* генерация выглядит так:<br>

Промпт: First man in a cosmic space was a 

Без стиринга: The first man to travel into outer space was **Yuri Gagarin**.

Со стирингом ($\alpha=0.3)$: The first man in the cosmos is **Elton John**, and he recently announced that he has been diagnosed with HIV.

---
Для субъекта *Kremlin*, отношения *is located in*, исходного объекта *Moscow* и целевого объекта *Kyoto* генерация выглядит так:<br>
Промпт: Где расположен Кремль?

Без стиринга:
*Кремль находится в Москве, Россия.*

Со стирингом:
*Кремль находится в городе Киото, Япония.*

## Как пользоваться
Устанавливаем пакет из папки с репозиторием:
```bash
git clone git@github.com:AndrewLevchenko/LLM-Fact-Editing-By-Steering.git
pip install -e . 
# или 
# pip install -e ".[research]"
```
Запуск одиночного эдита:
```bash
llmfes run-edit --model llama2-7b --subject "Kremlin" --relation "{} is located in " --object "Moscow" --object-edited "Kyoto" --alpha 0.6
llmfes run-edit --model llama2-7b --subject "First man in a space" --relation "{} was a " --object "Gagarin" --object-edited "Elton John" --alpha 1.4
```
Запуск чата с эдитом:
```bash
llmfes chat --model llama2-7b --subject "Kremlin" --relation "{} is located in " --object "Moscow" --object-edited "Kyoto" --alpha 0.6
llmfes chat --model llama2-7b --subject "First man in a space" --relation "{} was a " --object "Gagarin" --object-edited "Elton John" --alpha 1.4
```
Чтобы навесить в python коде steering на модель, сделайте:
```python
from llm_fact_editing_by_steering.utils.load_model import load_model
from llm_fact_editing_by_steering.editscontrollers.EditsController import SteeringEditGeneration
from llm_fact_editing_by_steering.hookscontrollers.CosineMultLastTokensHooksControllerV2 import CosineMultLastTokensHooksControllerV2

model,tokenizer = load_model("meta-llama/Llama-2-7b-chat-hf")
# или load_model("Qwen/Qwen3.5-9B")
# или load_model("t-tech/T-lite-it-2.1")
# или используйте свою модель
seg = SteeringEditGeneration(model,tokenizer,CosineMultLastTokensHooksControllerV2)
seg.set_edit(subject="Kremlin", relation="{} is located in ",object="Moscow", object_edited="Kyoto", alpha=1.0)
```
и пользуйтесь моделью в своё удовольствие.

либо из /scripts запустить
```bash
llama2_7b_console_edit_checkout.py --subject "Kremlin" --relation "{} is located in " --object "Moscow" --object-edited "Kyoto" --alpha 1.0 --max-new-tokens 100
qwen-3.5_console_edit_checkout.py --subject "Kremlin" --relation "{} is located in " --object "Moscow" --object-edited "Kyoto" --alpha 1.0 --max-new-tokens 100
t-lite-2.1_console_edit_checkout.py --subject "Kremlin" --relation "{} is located in " --object "Moscow" --object-edited "Kyoto" --alpha 0.3 --max-new-tokens 100
```

## Что сделано в `steering.ipynb`

### 1) Подготовка окружения
- Установка зависимостей (`torch`, `transformers`, `datasets`, `steering-vectors`, `bitsandbytes`, `accelerate`, `lm-eval` и др.).
- Загрузка модели `meta-llama/Llama-2-7b-chat-hf`.
- Подготовка датасета для валидации fact editing.

### 2) Базовая механика steering через hooks
- Реализована работа с forward hooks для чтения и модификации активаций.
- Создан контроллер для удобной регистрации/снятия hooks и генерации со steering.

### 3) Контрастивный датасет для steering-векторов
- Сформированы контрастивные пары (truth/target) для получения направлений редактирования.
- Проведены эксперименты с разными форматами таких пар.

### 4) Steering с косинусным множителем
- Реализована версия steering, где сила инжекции зависит от косинусного сходства текущей активации и целевого направления.
- Проверены варианты:
  - инжекция в последний токен,
  - инжекция во все токены,
  - инжекция в разные диапазоны слоёв.

### 5) Steering без контрастивного датасета
- Реализован подход с вектором разности активаций $o' - o$, что позволяет обойтись без отдельного набора контрастивных примеров.
- Добавлен `ActivationsController` для извлечения активаций и построения steering-вектора.

### 6) Автоматизация редактирования
- Добавлен класс `SteeringEditGeneration`, который по $s, r, o, o'$ запускает генерацию с выбранной стратегией steering.

### 7) Метрики
- **Edit Success** — доля успешных замен фактов.
- **Locality** — насколько steering сохраняет качество на нерелевантных фактах.
- Использован подход LLM-as-a-judge + ручная проверка спорных примеров.

### 8) Автоподбор `alpha`
- Добавлен механизм поиска коэффициента steering для баланса между успешным редактированием и сохранением locality.

---

## Ключевые выводы ноутбука

- Получен рабочий pipeline fact editing через steering без обучения параметров модели.
- Steering одного слоя часто недостаточен; лучше работает диапазон слоёв.
- Подход с разностью активаций \(o' - o\) показывает качественный результат и упрощает процесс подготовки данных.
- Итоговые оценки на тестовых примерах:
  - `Edit Success ≈ 97.1% (смотри /data/llama2_7b_chat_hf_log.txt)`
  - `Locality ≈ 99%` (в зависимости от трактовки спорных кейсов)

---

## Как запустить

1. Откройте `steering.ipynb`.
2. Установите зависимости в ячейке установки.
3. Настройте окружение (`LOCAL` или `KAGGLE`) и Hugging Face токен.
4. Запускайте ячейки последовательно сверху вниз.

---

## Структура репозитория

- `steering.ipynb` — основной ноутбук с полным пайплайном, экспериментами и метриками.
- `README.md` — краткое описание проекта и результатов.

---

## Дальнейшие улучшения
- Более строгая автоматическая оценка вместо частично ручной валидации.