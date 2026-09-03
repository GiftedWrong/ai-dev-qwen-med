"""Подготовка датасетов Mykes/rus_med_dialogues и Mykes/medical_cot_rus к SFT.

Результат — JSONL в data/, каждая строка:
    {"messages": [{"role": "system"}, {"role": "user"}, {"role": "assistant"}]}

Использование всех содержательных колонок:
  * rus_med_dialogues:
      - topic        -> в системный промпт («Специализация вопроса: ...»)
      - to_doctor    -> явная строка в конце целевого ответа
      - user_question / assistant_answer -> роли user / assistant
      - prompt (чужой шаблон <s><|user|>...|) НЕ используется как есть: его
        содержание полностью воспроизводится нашей сборкой через ChatML Qwen,
        а чужие спец-токены при обучении Qwen вредны.
      - __index_level_0__ — технический индекс экспорта pandas, без содержания.
  * medical_cot_rus:
      - question -> user; cot + answer -> структурированный ответ ассистента
        («### Рассуждение ... ### Итоговый ответ ...»)
      - raw_answer / old_thoughts (черновики, помечены автором как менее
        качественные) -> откладываются в data/cot_dpo_reserved.jsonl для
        будущего DPO, в SFT не участвуют.
      - old_index — технический индекс, без содержания.

Сплиты: dialogues — родной test + вырезанный из train val (5%);
cot — единая нарезка 94/3/3.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys

from datasets import load_dataset

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

MAX_COT_CHARS = 16_000  # префильтр по суммарной длине; точный токен-фильтр — в train.py


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {path}: {len(rows)} строк")


def shuffle(rows, seed=42):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows


def split3(rows, frac_val, frac_test, seed=42):
    rows = shuffle(rows, seed)
    n = len(rows)
    n_val = round(n * frac_val)
    n_test = round(n * frac_test)
    return rows[n_val + n_test:], rows[:n_val], rows[n_val:n_val + n_test]


def prep_dialogues(out_dir):
    print("== rus_med_dialogues ==")
    ds = load_dataset("Mykes/rus_med_dialogues")

    def convert(ex):
        q, a = norm(ex["user_question"]), norm(ex["assistant_answer"])
        topic, doctor = norm(ex.get("topic", "")), norm(ex.get("to_doctor", ""))
        if len(q) < 10 or len(a) < 20:
            return None
        system = SYSTEM_DIALOGUE + (f"\nСпециализация вопроса: {topic}." if topic else "")
        assistant = a + (f"\n\nРекомендуемый специалист: {doctor}." if doctor else "")
        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": q},
                {"role": "assistant", "content": assistant},
            ]
        }

    train_rows = [r for r in (convert(ex) for ex in ds["train"]) if r]
    test_rows = [r for r in (convert(ex) for ex in ds["test"]) if r]
    train_rows, val_rows, _ = split3(train_rows, frac_val=0.05, frac_test=0.0)
    write_jsonl(train_rows, os.path.join(out_dir, "dialogues_train.jsonl"))
    write_jsonl(val_rows, os.path.join(out_dir, "dialogues_val.jsonl"))
    write_jsonl(test_rows, os.path.join(out_dir, "dialogues_test.jsonl"))


def prep_cot(out_dir):
    print("== medical_cot_rus ==")
    ds = load_dataset("Mykes/medical_cot_rus", split="train")
    seen, rows, dpo = set(), [], []
    dropped_dup = dropped_len = 0
    for ex in ds:
        q = norm(ex["question"])
        cot = (ex["cot"] or "").strip()
        ans = norm(ex["answer"])
        if not q or len(cot) < 100 or len(ans) < 50:
            continue
        key = hashlib.md5(q.lower().encode()).hexdigest()
        if key in seen:
            dropped_dup += 1
            continue
        seen.add(key)
        assistant = f"### Рассуждение\n{cot}\n\n### Итоговый ответ\n{ans}"
        if len(q) + len(assistant) > MAX_COT_CHARS:
            dropped_len += 1
            continue
        messages = [
            {"role": "system", "content": SYSTEM_COT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": assistant},
        ]
        rows.append({"messages": messages})
        raw, old = norm(ex.get("raw_answer", "")), (ex.get("old_thoughts") or "").strip()
        if len(raw) > 100 or len(old) > 200:
            draft = old if len(old) > len(raw) else f"### Рассуждение\n{old}\n\n### Итоговый ответ\n{raw}"
            dpo.append({"prompt": messages[:2], "chosen": assistant, "rejected": draft})

    print(f"  дубликатов удалено: {dropped_dup}, отброшено по длине: {dropped_len}")
    train_rows, val_rows, test_rows = split3(rows, frac_val=0.03, frac_test=0.03)
    write_jsonl(train_rows, os.path.join(out_dir, "cot_train.jsonl"))
    write_jsonl(val_rows, os.path.join(out_dir, "cot_val.jsonl"))
    write_jsonl(test_rows, os.path.join(out_dir, "cot_test.jsonl"))
    write_jsonl(dpo, os.path.join(out_dir, "cot_dpo_reserved.jsonl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    prep_dialogues(args.out)
    prep_cot(args.out)


if __name__ == "__main__":
    sys.exit(main())
