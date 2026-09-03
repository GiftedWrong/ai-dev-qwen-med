"""Бенчмарк медицинских моделей и их комбинаций на нескольких задачах.

Участники (выбираются --participants):
  base      — базовый Qwen2.5-3B-Instruct
  dialogue  — models/med-dialogue-3b
  cot       — models/med-cot-3b
  cot_dpo   — models/med-cot-3b-dpo (если обучен)
  joint     — конвейер dialogue -> cot -> dialogue (как /api/joint на сервере)
  wrap      — конвейер cot -> dialogue (разбор + пациентская упаковка)

Задачи:
  1. routing   — точность маршрутизации к специалисту на диалогах из test-набора
                 (эталон — поле to_doctor; проверяется ответ модели, спец-подсказка
                 со специализацией вырезается, чтобы не подсказывать ответ)
  2. fact_qa   — курируемый набор benchmarks/fact_qa.jsonl: полнота ключевых фактов
                 (recall по группам синонимов) + нарушения безопасности (banned)
  3. cot_format — соблюдение формата «### Рассуждение / ### Итоговый ответ»

Сквозные метрики: чистота языка (доля CJK-иероглифов), средняя длина ответа.

Режим генерации у всех участников одинаковый (t=0.2, top_p=0.95, rep 1.1),
сид фиксирован — сравнение честное и воспроизводимое.

Запуск (из корня проекта):
  python scripts/benchmark.py --participants base,dialogue,cot,joint,wrap
  python scripts/benchmark.py --participants cot,cot_dpo --routing-limit 60

Отчёт: runs/benchmark_report.md (+ .json со всеми ответами для разбора).
"""

import argparse
import json
import os
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {
    "base": os.path.join(ROOT, "Qwen2.5-3B-Instruct"),
    "dialogue": os.path.join(ROOT, "models", "med-dialogue-3b"),
    "cot": os.path.join(ROOT, "models", "med-cot-3b"),
    "cot_ep1": os.path.join(ROOT, "models", "med-cot-3b-ep1"),
    "cot_dpo": os.path.join(ROOT, "models", "med-cot-3b-dpo"),
}

SYSTEM_DIALOGUE = (
    "Ты — медицинский ассистент-консультант. Тебе пишет пациент обычными словами. "
    "Отвечай на русском языке: спокойно объясни возможные причины симптомов, "
    "подскажи, к какому специалисту обратиться, и предостереги от самолечения. "
    "Не ставь окончательный диагноз и не заменяй консультацию врача."
)
SYSTEM_COT = (
    "Ты — медицинский ассистент с клиническим мышлением. Отвечай на русском языке. "
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

CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
DOCTOR_LINE = re.compile(r"Рекомендуемый специалист:\s*([^\n]+)")


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


class Runner:
    """Держит загруженные модели и генерирует ответы в едином режиме."""

    def __init__(self):
        self.loaded = {}

    def get(self, key):
        if key not in self.loaded:
            path = MODELS[key]
            print(f"  [load] {key}: {path}")
            tok = AutoTokenizer.from_pretrained(path)
            mdl = AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.bfloat16, device_map="cuda"
            )
            mdl.eval()
            self.loaded[key] = (mdl, tok)
        return self.loaded[key]

    def unload(self):
        for key in list(self.loaded):
            del self.loaded[key]
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def gen(self, key, messages, max_new_tokens, seed):
        model, tokenizer = self.get(key)
        torch.manual_seed(seed)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.2, top_p=0.95, repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
        return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def participant_answer(runner, name, question, style, max_new_tokens, seed):
    """Единый интерфейс: любой участник отвечает на вопрос в заданном стиле."""
    system = SYSTEM_DIALOGUE if style == "patient" else SYSTEM_COT
    if name in ("base", "dialogue", "cot", "cot_ep1", "cot_dpo"):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        return runner.gen(name, msgs, max_new_tokens, seed)
    if name == "joint":
        cq = runner.gen("dialogue", [
            {"role": "system", "content": SYSTEM_REFORMULATE},
            {"role": "user", "content": question},
        ], 256, seed)
        analysis = runner.gen("cot", [
            {"role": "system", "content": SYSTEM_COT},
            {"role": "user", "content": cq},
        ], max_new_tokens, seed + 1)
        final_user = (
            f"Жалоба пациента: {question}\n\n"
            f"Клинический разбор ассистента:\n{analysis}\n\n"
            "Опираясь на разбор, сформулируй итоговый ответ пациенту: успокой/объясни "
            "простыми словами, к какому специалисту обратиться и что делать до приёма. "
            "Не называй конкретные лекарственные препараты и не назначай лечение — "
            "любые лекарства обсуждаются только с врачом на приёме. "
            "Рекомендации по образу жизни давай только общепринятые и безопасные: "
            "не советуй ложиться сразу после еды; при изжоге и рефлюксе правильно — "
            "оставаться в вертикальном положении 2–3 часа после еды и спать с приподнятым "
            "изголовьем."
        )
        return runner.gen("dialogue", [
            {"role": "system", "content": SYSTEM_DIALOGUE},
            {"role": "user", "content": final_user},
        ], 512, seed + 2)
    if name == "wrap":
        analysis = runner.gen("cot", [
            {"role": "system", "content": SYSTEM_COT},
            {"role": "user", "content": question},
        ], max_new_tokens, seed)
        final_user = (
            f"Вопрос: {question}\n\n"
            f"Клинический разбор ассистента:\n{analysis}\n\n"
            "Объясни пациенту простыми словами итог разбора: что это может значить, "
            "к какому специалисту обратиться и что делать до приёма."
        )
        return runner.gen("dialogue", [
            {"role": "system", "content": SYSTEM_DIALOGUE},
            {"role": "user", "content": final_user},
        ], 512, seed + 1)
    raise ValueError(name)


def load_routing_rows(limit):
    """Вопросы + эталонный специалист из dialogues_test; подсказку вырезаем."""
    rows = []
    with open(os.path.join(ROOT, "data", "dialogues_test.jsonl"), encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            sysm, user, ref = ex["messages"]
            m = DOCTOR_LINE.search(ref["content"])
            if not m:
                continue
            rows.append({
                "system": re.sub(r"Специализация вопроса:[^\n]*", "", sysm["content"]).strip(),
                "question": user["content"],
                "gold": norm(m.group(1)).rstrip("."),
            })
    return rows[:limit]


def load_factqa():
    with open(os.path.join(ROOT, "benchmarks", "fact_qa.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_cot_rows(limit):
    rows = []
    with open(os.path.join(ROOT, "data", "cot_test.jsonl"), encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            rows.append({"question": ex["messages"][1]["content"]})
    return rows[:limit]


def score_routing(answers_rows):
    ok = no_line = 0
    for r in answers_rows:
        m = DOCTOR_LINE.search(r["answer"])
        if not m:
            no_line += 1
            continue
        pred = norm(m.group(1)).rstrip(".")
        g = r["gold"]
        if g in pred or pred in g or (
            len(pred) > 3 and len(g) > 3 and pred[:5] == g[:5]
        ):
            ok += 1
    n = len(answers_rows)
    return {"accuracy": round(ok / n, 3), "no_line_rate": round(no_line / n, 3)}


def score_factqa(answers_rows):
    recall, banned_hits, details = [], [], []
    for r in answers_rows:
        a = r["answer"].lower()
        covered = sum(
            1 for group in r["expected"] if any(syn.lower() in a for syn in group)
        )
        recall.append(covered / max(len(r["expected"]), 1))
        hit = [syn for group in r["banned"] for syn in group if syn.lower() in a]
        if hit:
            banned_hits.append({"id": r["id"], "hits": hit})
        details.append({"id": r["id"], "recall": covered / max(len(r["expected"]), 1)})
    return {
        "fact_recall": round(sum(recall) / len(recall), 3),
        "safety_violations": banned_hits,
        "per_item": details,
    }


def score_format(answers):
    ok = sum(1 for a in answers if "### Рассуждение" in a and "### Итоговый ответ" in a)
    return round(ok / len(answers), 3)


def cjk_share(text):
    if not text:
        return 0.0
    return len(CJK.findall(text)) / max(len(text), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default="base,dialogue,cot,joint,wrap")
    ap.add_argument("--routing-limit", type=int, default=40)
    ap.add_argument("--cot-format-limit", type=int, default=10)
    ap.add_argument("--cot-max-tokens", type=int, default=1536)
    args = ap.parse_args()

    routing_rows = load_routing_rows(args.routing_limit)
    fact_rows = load_factqa()
    cot_rows = load_cot_rows(args.cot_format_limit)
    print(f"задач: routing {len(routing_rows)}, fact_qa {len(fact_rows)}, "
          f"cot_format {len(cot_rows)}")

    runner = Runner()
    report = {}
    all_answers = {}
    for name in args.participants.split(","):
        name = name.strip()
        if name not in MODELS and name not in ("joint", "wrap"):
            print(f"!! неизвестный участник {name}, пропуск")
            continue
        if name in MODELS and not os.path.isdir(MODELS[name]):
            print(f"!! модель не найдена: {MODELS[name]} ({name}), пропуск")
            continue
        print(f"== участник: {name} ==")
        t0 = time.time()
        answers = []

        for i, r in enumerate(routing_rows):
            answers.append(participant_answer(
                runner, name, r["question"], "patient", 512, seed=1000 + i))
        routing = score_routing([
            {"answer": a, "gold": r["gold"]} for a, r in zip(answers, routing_rows)])

        fact_answers = []
        for i, r in enumerate(fact_rows):
            fact_answers.append(participant_answer(
                runner, name, r["question"], r["style"], args.cot_max_tokens,
                seed=2000 + i))
        fact = score_factqa([
            {**r, "answer": a} for r, a in zip(fact_rows, fact_answers)])

        cot_answers = []
        for i, r in enumerate(cot_rows):
            cot_answers.append(participant_answer(
                runner, name, r["question"], "clinical", args.cot_max_tokens,
                seed=3000 + i))

        every = answers + fact_answers + cot_answers
        report[name] = {
            "routing": routing,
            "fact_qa": fact,
            "cot_format": score_format(cot_answers),
            "language_purity": round(
                sum(1 for a in every if cjk_share(a) == 0) / len(every), 3),
            "avg_cjk_share": round(sum(cjk_share(a) for a in every) / len(every), 5),
            "avg_len_chars": round(sum(len(a) for a in every) / len(every)),
            "seconds": round(time.time() - t0),
        }
        all_answers[name] = {
            "routing": [dict(q=r["question"], gold=r["gold"], answer=a)
                        for r, a in zip(routing_rows, answers)],
            "fact_qa": [dict(id=r["id"], answer=a)
                        for r, a in zip(fact_rows, fact_answers)],
            "cot_format": [dict(q=r["question"], answer=a)
                           for r, a in zip(cot_rows, cot_answers)],
        }
        runner.unload()
        print(f"  {name}: routing={routing['accuracy']}, "
              f"fact_recall={fact['fact_recall']}, "
              f"violations={len(fact['safety_violations'])}, "
              f"за {report[name]['seconds']}с")

    lines = ["# Бенчмарк медицинских моделей", "",
             f"участники: {', '.join(report.keys())}; "
             f"routing n={len(routing_rows)}, fact_qa n={len(fact_rows)}, "
             f"cot_format n={len(cot_rows)}", "",
             "| участник | routing acc | no-line | fact recall | safety viol | "
             "cot format | CJK-free | сек |",
             "|---|---|---|---|---|---|---|---|"]
    for name, m in report.items():
        lines.append(
            f"| {name} | {m['routing']['accuracy']} | {m['routing']['no_line_rate']} | "
            f"{m['fact_qa']['fact_recall']} | {len(m['fact_qa']['safety_violations'])} | "
            f"{m['cot_format']} | {m['language_purity']} | {m['seconds']} |")
    lines += ["", "## Нарушения безопасности"]
    for name, m in report.items():
        for v in m["fact_qa"]["safety_violations"]:
            lines.append(f"- **{name}** / {v['id']}: {', '.join(v['hits'])}")
    if not any(m["fact_qa"]["safety_violations"] for m in report.values()):
        lines.append("- нет")

    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    md = os.path.join(ROOT, "runs", "benchmark_report.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(ROOT, "runs", "benchmark_answers.json"), "w",
              encoding="utf-8") as f:
        json.dump({"metrics": report, "answers": all_answers}, f, ensure_ascii=False,
                  indent=1)
    print("\n".join(lines))
    print(f"\nотчёт: {md}\nответы: runs/benchmark_answers.json")


if __name__ == "__main__":
    sys.exit(main())
