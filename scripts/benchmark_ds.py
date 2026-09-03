"""Бенчмарк на невиданных данных датасетов (test-сплиты) — дополнение к внешнему
scripts/benchmark.py (fact_qa). Здесь эталоны берутся из самих датасетов:
  * dialogues_test.jsonl — вопросы пациентов + эталонные ответы + специалист (to_doctor)
  * cot_test.jsonl       — клинические вопросы + эталонные рассуждения и ответы

Каждый участник (базовая, обе дообученные, DPO-версия, комбинации joint/wrap)
прогоняется через ОБЕ задачи — получается перекрёстная матрица специализации:
какая модель насколько держит чужой и свой домен.

Метрики:
  * routing acc — точность «к какому специалисту» (эталон to_doctor)
  * ROUGE-L F1  — структурное совпадение с эталонным ответом (LCS)
  * bag-F1      — перекрытие словаря с эталоном (мультимножество токенов)
  * final-F1    — bag-F1 только по секции «### Итоговый ответ» (медицински
                  значимая часть; для ответов без секции берётся весь текст)
  * cot format  — доля ответов со структурой «### Рассуждение / ### Итоговый ответ»
  * (--with-loss) test loss на обоих test-наборах — только для одиночных моделей

Ограничение интерпретации: комбинации joint/wrap на клинических вопросах выдают
пациентский ответ — их низкий final-F1 против клинического эталона ожидаем и не
означает ошибку (итоговый текст адресован пациенту, а не врачу).

Запуск (из корня проекта):
  python scripts/benchmark_ds.py --participants base,dialogue,cot,joint,wrap
  python scripts/benchmark_ds.py --participants cot,cot_dpo --cot-limit 25 --with-loss

Отчёт: runs/benchmark_ds_report.md (+ runs/benchmark_ds_answers.json).
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import (  # noqa: E402
    DOCTOR_LINE, Runner, cjk_share, norm, participant_answer, score_format,
    score_routing,
)
from train import PRESETS, Collator, load_jsonl, tokenize  # noqa: E402


def tokens(text):
    return re.findall(r"[а-яёa-z0-9]+", (text or "").lower())


def lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(gen, ref):
    g, r = tokens(gen), tokens(ref)
    if not g or not r:
        return 0.0
    l = lcs_len(g, r)
    p, rc = l / len(g), l / len(r)
    return round(2 * p * rc / (p + rc), 3) if p + rc else 0.0


def bag_f1(gen, ref):
    g, r = Counter(tokens(gen)), Counter(tokens(ref))
    common = sum((g & r).values())
    if not common:
        return 0.0
    p, rc = common / sum(g.values()), common / sum(r.values())
    return round(2 * p * rc / (p + rc), 3)


def extract_final(text):
    m = re.search(r"###\s*Итоговый ответ\s*(.+)$", text, re.S)
    return m.group(1).strip() if m else text.strip()


def load_dialogues(limit):
    rows = []
    for ex in load_jsonl(os.path.join(ROOT, "data", "dialogues_test.jsonl")):
        sysm, user, ref = ex["messages"]
        m = DOCTOR_LINE.search(ref["content"])
        if not m:
            continue
        rows.append({
            "question": user["content"],
            "reference": ref["content"],
            "gold": norm(m.group(1)).rstrip("."),
        })
    return rows[:limit]


def load_cot(limit):
    return [{
        "question": ex["messages"][1]["content"],
        "reference": ex["messages"][2]["content"],
    } for ex in load_jsonl(os.path.join(ROOT, "data", "cot_test.jsonl"))][:limit]


@torch.inference_mode()
def test_loss(runner, key, data_path, max_len):
    """Токен-взвешенный loss на test-наборе (для одиночных моделей)."""
    from datasets import Dataset
    model, tokenizer = runner.get(key)
    rows = load_jsonl(data_path)
    raw = Dataset.from_list(rows)
    ds = raw.map(tokenize, batched=True, fn_kwargs=dict(tokenizer=tokenizer, max_len=max_len),
                 remove_columns=raw.column_names)
    collator = Collator(tokenizer.pad_token_id)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collator)
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1]
        tgt = batch["labels"][:, 1:]
        mask = tgt != -100
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1))[mask.reshape(-1)],
            tgt.reshape(-1)[mask.reshape(-1)], reduction="sum",
        )
        total += float(ce)
        count += int(mask.sum())
    return round(total / max(count, 1), 4)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default="base,dialogue,cot,joint,wrap")
    ap.add_argument("--dialogue-limit", type=int, default=40)
    ap.add_argument("--cot-limit", type=int, default=15)
    ap.add_argument("--with-loss", action="store_true",
                    help="добавить test loss (только одиночные модели)")
    args = ap.parse_args()

    dial_rows = load_dialogues(args.dialogue_limit)
    cot_rows = load_cot(args.cot_limit)
    print(f"задач: dialogues {len(dial_rows)}, cot {len(cot_rows)}")

    from benchmark import MODELS

    runner = Runner()
    report, all_answers = {}, {}
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

        # --- задача 1: диалоги (пациентский стиль, эталон из dialogues_test)
        dial_answers = [
            participant_answer(runner, name, r["question"], "patient", 512, seed=4000 + i)
            for i, r in enumerate(dial_rows)
        ]
        routing = score_routing(
            [{"answer": a, "gold": r["gold"]} for a, r in zip(dial_answers, dial_rows)])
        dial_rouge = sum(rouge_l(a, r["reference"])
                         for a, r in zip(dial_answers, dial_rows)) / len(dial_rows)
        dial_bag = sum(bag_f1(a, r["reference"])
                       for a, r in zip(dial_answers, dial_rows)) / len(dial_rows)

        # --- задача 2: клинические вопросы (эталон из cot_test)
        cot_answers = [
            participant_answer(runner, name, r["question"], "clinical", 1536, seed=5000 + i)
            for i, r in enumerate(cot_rows)
        ]
        cot_rouge = sum(rouge_l(a, r["reference"])
                        for a, r in zip(cot_answers, cot_rows)) / len(cot_rows)
        cot_final = sum(bag_f1(extract_final(a), extract_final(r["reference"]))
                        for a, r in zip(cot_answers, cot_rows)) / len(cot_rows)
        fmt = score_format(cot_answers)
        purity = sum(1 for a in dial_answers + cot_answers if cjk_share(a) == 0) / (
            len(dial_answers) + len(cot_answers))

        entry = {
            "routing_acc": routing["accuracy"],
            "dialog_rouge_l": round(dial_rouge, 3),
            "dialog_bag_f1": round(dial_bag, 3),
            "cot_rouge_l": round(cot_rouge, 3),
            "cot_final_f1": round(cot_final, 3),
            "cot_format": fmt,
            "cjk_free": round(purity, 3),
            "seconds": round(time.time() - t0),
        }
        if args.with_loss and name in MODELS:
            entry["loss_dialog"] = test_loss(
                runner, name, os.path.join(ROOT, "data", "dialogues_test.jsonl"),
                PRESETS["dialogues"]["max_len"])
            entry["loss_cot"] = test_loss(
                runner, name, os.path.join(ROOT, "data", "cot_test.jsonl"),
                PRESETS["cot"]["max_len"])
        report[name] = entry
        all_answers[name] = {
            "dialogues": [dict(q=r["question"], gold=r["gold"], ref=r["reference"], answer=a)
                          for r, a in zip(dial_rows, dial_answers)],
            "cot": [dict(q=r["question"], ref=r["reference"], answer=a)
                    for r, a in zip(cot_rows, cot_answers)],
        }
        runner.unload()
        print(f"  {name}: routing={entry['routing_acc']}, "
              f"dialog rouge={entry['dialog_rouge_l']}, "
              f"cot final-F1={entry['cot_final_f1']}, за {entry['seconds']}с")

    lines = ["# Бенчмарк на невиданных данных датасетов", "",
             f"участники: {', '.join(report)}; dialogues n={len(dial_rows)}, "
             f"cot n={len(cot_rows)}; эталоны — dialogues_test / cot_test", ""]
    has_loss = any("loss_dialog" in m for m in report.values())
    head = ("| участник | routing acc | dialog ROUGE-L | dialog bag-F1 | cot ROUGE-L | "
            "cot final-F1 | cot format | CJK-free |")
    if has_loss:
        head += " loss dialog | loss cot |"
    lines += [head, "|---" * (8 + (2 if has_loss else 0)) + "|"]
    for name, m in report.items():
        row = (f"| {name} | {m['routing_acc']} | {m['dialog_rouge_l']} | "
               f"{m['dialog_bag_f1']} | {m['cot_rouge_l']} | {m['cot_final_f1']} | "
               f"{m['cot_format']} | {m['cjk_free']} |")
        if has_loss:
            row += (f" {m.get('loss_dialog', '—')} | {m.get('loss_cot', '—')} |")
        lines.append(row)
    lines += ["",
              "*Комбинации joint/wrap на клинических вопросах отвечают пациентским "
              "текстом — их cot-метрики против клинического эталона заведомо ниже, "
              "это не ошибка.*"]

    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    md = os.path.join(ROOT, "runs", "benchmark_ds_report.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(ROOT, "runs", "benchmark_ds_answers.json"), "w",
              encoding="utf-8") as f:
        json.dump({"metrics": report, "answers": all_answers}, f, ensure_ascii=False,
                  indent=1)
    print("\n".join(lines))
    print(f"\nотчёт: {md}\nответы: runs/benchmark_ds_answers.json")


if __name__ == "__main__":
    sys.exit(main())
