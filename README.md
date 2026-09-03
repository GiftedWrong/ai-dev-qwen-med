# Медицинский ассистент на Qwen2.5-3B

Две модели на базе Qwen2.5-3B-Instruct, обученные полным файнтюном:

- `models/med-dialogue-3b` — ответы на пациентские вопросы: вероятные
  причины симптомов, к какому специалисту идти. Датасет
  [rus_med_dialogues](https://huggingface.co/datasets/Mykes/rus_med_dialogues)
  (2863/151/335).
- `models/med-cot-3b` — клинические задачи с рассуждением, ответ в формате
  «Рассуждение → Итоговый ответ». Датасет
  [medical_cot_rus](https://huggingface.co/datasets/Mykes/medical_cot_rus)
  (5901/188/188).

Плюс FastAPI-сервер: модели доступны по отдельности и в конвейере
(вопрос → рассуждение → ответ пациенту). Всё обучалось локально на одной
RTX 3090, код воспроизводим скриптами из репозитория.

## Результаты

На test-сплитах датасетов (модели их не видели), сравнение с базовой
Qwen2.5-3B-Instruct:

- точность маршрута к специалисту: 0.48 у файнтюна против 0.0 у базы,
  в конвейере №1→№2→№1 — 0.64;
- ответы модели №1: ROUGE-L 0.294 против 0.142;
- итоговый ответ модели №2: F1 0.329 против 0.223;
- формат «Рассуждение → Итоговый ответ»: 1.0 против 0.0;
- safety-проверки (вредные советы, несуществующие препараты): 0 срабатываний
  у всех участников.

DPO-этап для модели №2 проверен отдельно (LoRA-DPO со слиянием, полный
DPO не влезает в 24 ГБ): прироста нет, в итоговую конфигурацию не включён.

Отчёты: `runs/benchmark_report_ext6.md`, `runs/benchmark_ds_report.md`,
`REPORT.md`.

## Запуск

Python 3.12, одна карта ~24 ГБ VRAM.

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/check_engine.py        # самотест движка обучения
.venv/bin/python scripts/prepare_data.py        # датасеты -> data/*.jsonl
.venv/bin/python scripts/train.py --preset dialogues
.venv/bin/python scripts/train.py --preset cot
.venv/bin/python scripts/evaluate.py --model models/med-dialogue-3b --preset dialogues

.venv/bin/uvicorn inference.server:app --host 127.0.0.1 --port 8000
```

Обучение №1 занимает 45–90 минут, №2 — 1,5–3 часа. Полный список команд
(включая опциональный DPO и аварийные варианты при OOM) — в разделе
пайплайна ниже.

## API

| Метод | Эндпоинт | Что делает |
|---|---|---|
| GET | `/api/health` | статус обеих моделей |
| POST | `/api/dialogue` | `{"question": "..."}` → ответ модели №1 |
| POST | `/api/cot` | `{"question": "..."}` → рассуждение + итог модели №2 |
| POST | `/api/joint` | конвейер №1→№2→№1 |

```
curl -s -X POST 127.0.0.1:8000/api/joint -H 'Content-Type: application/json' \
  -d '{"question": "Мучает изжога по ночам, что делать?"}'
```

Swagger: `127.0.0.1:8000/docs`. Генерации сериализованы одной GPU
(`gpu_lock`), joint-режим занимает 30–90 секунд.

## Бенчмарки

- `scripts/benchmark.py` — внешний бенчмарк: одинаковые вопросы, фиксированный
  режим генерации, метрики routing acc, fact recall, safety, cot format,
  CJK-free. Участники: `base`, `dialogue`, `cot`, `cot_dpo`, `joint`, `wrap`.
- `scripts/benchmark_ds.py` — то же на test-сплитах датасетов (ROUGE-L,
  bag-F1, final-F1, опционально test loss через `--with-loss`).

```
.venv/bin/python scripts/benchmark.py --participants base,dialogue,cot,joint,wrap
.venv/bin/python scripts/benchmark_ds.py --participants base,dialogue,cot,joint,wrap
```

Отчёты пишутся в `runs/`, все генерации сохраняются в `*_answers.json`
для ручного разбора.

## Структура

```
scripts/prepare_data.py           # датасеты -> data/*.jsonl
scripts/check_engine.py           # самотест движка обучения (unsloth full FT)
scripts/train.py                  # полный SFT-файнтюн (unsloth | transformers)
scripts/train_dpo.py              # LoRA-DPO + слияние (для CoT-модели)
scripts/evaluate.py               # test loss + примеры генераций
scripts/benchmark.py              # внешний бенчмарк
scripts/benchmark_ds.py           # бенчмарк на test-сплитах
inference/server.py               # FastAPI
inference/client.py               # CLI-клиент
data/                             # выборки + cot_dpo_reserved.jsonl
runs/                             # логи, отчёты, генерации
```

## Подготовка данных

- Все содержательные колонки использованы: `topic` — в системный промпт,
  `to_doctor` — строкой «Рекомендуемый специалист: …» в целевом ответе;
  в cot-датасете `question`+`cot`+`answer` — в структурированный ответ.
- Колонка `prompt` датасета не используется: это тот же контент в чужом
  шаблоне со спец-токенами, которые вредят обучению; ChatML-сборка Qwen
  воспроизводит его полностью.
- Черновики `raw_answer`/`old_thoughts` отложены в
  `data/cot_dpo_reserved.jsonl` как пары для DPO.
- Лосс считается только по токенам ответа ассистента.

## Полный пайплайн

```
# 0. Самотест: unsloth full FT, 1 микрошаг (~2 мин, ~20 ГБ VRAM)
.venv/bin/python scripts/check_engine.py          # ждём строку "ENGINE OK"

# 1. Данные
.venv/bin/python scripts/prepare_data.py

# 2. Модель №1 (~45–90 мин)
PYTHONUNBUFFERED=1 .venv/bin/python scripts/train.py --preset dialogues 2>&1 | tee runs/train_dialogues.log

# 3. Модель №2, после завершения №1 (~1,5–3 ч)
PYTHONUNBUFFERED=1 .venv/bin/python scripts/train.py --preset cot 2>&1 | tee runs/train_cot.log

# 4. Оценка
.venv/bin/python scripts/evaluate.py --model models/med-dialogue-3b --preset dialogues
.venv/bin/python scripts/evaluate.py --model models/med-cot-3b --preset cot

# 4b. Опционально: LoRA-DPO для CoT-модели + оценка
PYTHONUNBUFFERED=1 .venv/bin/python scripts/train_dpo.py --model models/med-cot-3b
.venv/bin/python scripts/evaluate.py --model models/med-cot-3b-dpo --preset cot

# 5. Сервер (обе модели в памяти, ~12,4 ГБ VRAM)
.venv/bin/uvicorn inference.server:app --host 127.0.0.1 --port 8000
```

Если unsloth не проходит самотест — `train.py --engine transformers`
(обучение идёт без ускорения). При OOM — `--batch 1`, для cot ещё
`--max-len 1536`. Если недоступен HuggingFace —
`export HF_ENDPOINT=https://hf-mirror.com`.

## Ограничения

Датасеты синтетические и не проверены врачами, модели нельзя использовать
для реальной постановки диагнозов. Лицензия medical_cot_rus неоднозначна,
поэтому веса не публикуются. Проект учебно-исследовательский.
