"""API-сервер инференса: две медицинские модели, раздельно и совместно.

Модели (обе в bf16 на одной GPU, ~12,4 ГБ VRAM):
  * med-dialogue-3b — пациентский триаж, направление к специалисту
  * med-cot-3b      — клинические рассуждения с итоговым ответом

Эндпоинты:
  GET  /api/health   — статус обеих моделей
  POST /api/dialogue — ответ триажной модели на вопрос пациента
  POST /api/cot      — рассуждение + итог по клиническому вопросу
  POST /api/joint    — совместный конвейер:
       1) триажная модель переформулирует жалобу в клинический вопрос;
       2) рассуждающая модель проводит клинический разбор;
       3) триажная модель оборачивает разбор в итоговый ответ пациенту.

Запуск (из корня проекта):
  .venv/bin/uvicorn inference.server:app --host 127.0.0.1 --port 8000
"""

import os
import threading
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PATH = os.path.join(ROOT, "Qwen2.5-3B-Instruct")
DIALOGUE_PATH = os.path.join(ROOT, "models", "med-dialogue-3b")
COT_PATH = os.path.join(ROOT, "models", "med-cot-3b")

# Системные промпты повторяют тренировочные (scripts/prepare_data.py);
# строка специализации в инференсе не подставляется — модель работает по жалобе.
SYSTEM_DIALOGUE = (
    "Ты — медицинский ассистент-консультант. Тебе пишет пациент обычными словами. "
    "Отвечай на русском языке: спокойно объясни возможные причины симптомов, "
    "подскажи, к какому специалисту обратиться, и предостереги от самолечения. "
    "Не ставь окончательный диагноз и не заменяй консультацию врача."
)
SYSTEM_COT = (
    "Ты — медицинский ассистент с клиническим мышлением. Отвечай строго на русском "
    "языке, не используй никакие другие языки или иероглифы. "
    "Сначала подробно и логично проанализируй вопрос (симптомы, вероятные причины, "
    "дифференциальный диагноз, необходимые обследования), затем дай чёткий итоговый ответ. "
    "Помни, что материал предназначен для обучения и исследований, а не для реальной "
    "постановки диагнозов."
)
SYSTEM_REFORMULATE = (
    "Ты — медицинский ассистент. Преобразуй жалобу или вопрос пациента в один краткий "
    "структурированный клинический вопрос для врачебного разбора (симптомы, длительность, "
    "ключевые детали). Верни только текст вопроса, без пояснений."
)

gpu_lock = threading.Lock()  # одна GPU — генерации строго по очереди
bundles: dict[str, tuple] = {}


def load_bundle(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return model, tokenizer


@asynccontextmanager
async def lifespan(app):
    for key, path in (("dialogue", DIALOGUE_PATH), ("cot", COT_PATH),
                      ("base", BASE_PATH)):
        if not os.path.isdir(path):
            raise RuntimeError(f"модель не найдена: {path} — сначала обучите её (scripts/train.py)")
        bundles[key] = load_bundle(path)
        print(f"[server] загружена {key}: {path}")
    yield
    bundles.clear()


app = FastAPI(title="Med Models API", version="1.0", lifespan=lifespan)


class Question(BaseModel):
    question: str


def chat(key, system, user, max_new_tokens, temperature=0.6, top_p=0.9):
    """temperature<=0.25 — почти детерминированный режим: низкотемпературный
    сэмплинг вместо чистого greedy (чистый greedy у Qwen циклит и проваливается
    в китайский), с репетишн-панальти против повторов."""
    model, tokenizer = bundles[key]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    if temperature > 0.25:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(
            do_sample=True, temperature=max(temperature, 0.2), top_p=0.95,
            repetition_penalty=1.1,
        )
    with gpu_lock, torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


@app.get("/api/health")
def health():
    return {
        "status": "ok" if bundles else "loading",
        "models": {k: "ok" for k in bundles},
        "device": "cuda",
        "vram_gb": round(torch.cuda.memory_allocated() / 2**30, 1),
    }


@app.post("/api/dialogue")
def api_dialogue(q: Question):
    return {"model": "med-dialogue-3b", "answer": chat("dialogue", SYSTEM_DIALOGUE, q.question, 512)}


@app.post("/api/base")
def api_base(q: Question):
    """База с тем же системным промптом, что у диалоговой — для честного сравнения."""
    return {"model": "qwen2.5-3b-instruct (base)", "answer": chat("base", SYSTEM_DIALOGUE, q.question, 512)}


@app.post("/api/cot")
def api_cot(q: Question):
    return {
        "model": "med-cot-3b",
        "decoding": "greedy",
        "answer": chat("cot", SYSTEM_COT, q.question, 1536, temperature=0.0),
    }


@app.post("/api/joint")
def api_joint(q: Question):
    clinical_question = chat("dialogue", SYSTEM_REFORMULATE, q.question, 256)
    cot_analysis = chat("cot", SYSTEM_COT, clinical_question, 1536, temperature=0.0)
    final_user = (
        f"Жалоба пациента: {q.question}\n\n"
        f"Клинический разбор ассистента:\n{cot_analysis}\n\n"
        "Опираясь на разбор, сформулируй итоговый ответ пациенту: успокой/объясни "
        "простыми словами, к какому специалисту обратиться и что делать до приёма. "
        "Не называй конкретные лекарственные препараты и не назначай лечение — "
        "любые лекарства обсуждаются только с врачом на приёме. "
        "Рекомендации по образу жизни давай только общепринятые и безопасные: "
        "не советуй ложиться сразу после еды; при изжоге и рефлюксе правильно — "
        "оставаться в вертикальном положении 2–3 часа после еды и спать с приподнятым "
        "изголовьем."
    )
    final_answer = chat("dialogue", SYSTEM_DIALOGUE, final_user, 512, temperature=0.0)
    return {
        "pipeline": ["med-dialogue-3b", "med-cot-3b", "med-dialogue-3b"],
        "clinical_question": clinical_question,
        "cot_analysis": cot_analysis,
        "final_answer": final_answer,
    }


@app.post("/api/wrap")
def api_wrap(q: Question):
    """Комбинация cot -> dialogue: разбор + пациентская упаковка (одна переупаковка)."""
    cot_analysis = chat("cot", SYSTEM_COT, q.question, 1536, temperature=0.0)
    final_user = (
        f"Вопрос: {q.question}\n\n"
        f"Клинический разбор ассистента:\n{cot_analysis}\n\n"
        "Объясни пациенту простыми словами итог разбора: что это может значить, "
        "к какому специалисту обратиться и что делать до приёма."
    )
    final_answer = chat("dialogue", SYSTEM_DIALOGUE, final_user, 512, temperature=0.0)
    return {
        "pipeline": ["med-cot-3b", "med-dialogue-3b"],
        "cot_analysis": cot_analysis,
        "final_answer": final_answer,
    }
