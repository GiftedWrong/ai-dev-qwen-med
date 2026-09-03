"""Оценка модели персоны: val loss + примеры генераций в голосе Эльдара.

Запуск (из корня проекта):
  .venv/bin/python persona_eldar/evaluate_eldar.py --model models/eldar-persona-3b
"""

import argparse
import os
import sys

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "scripts"))
from evaluate import test_loss  # noqa: E402
from train import Collator, load_jsonl, tokenize  # noqa: E402


@torch.inference_mode()
def generate(model, tokenizer, messages, max_new_tokens):
    """Сэмплирование с теми же параметрами, что в chat_eldar.py: жадное
    декодирование на коллоквиальных текстах персоны уходит в петли повторов
    и не отражает реальное поведение модели в чате."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=True, temperature=0.8, top_p=0.95, repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

VAL = os.path.join(PROJECT, "persona_eldar", "data", "eldar_val.jsonl")

# вопросы вне обучающей выборки — проверка переноса стиля, а не памяти
PROBES = [
    "Эльдар, расскажи про своё новое кино",
    "Чем сегодня занимался?",
    "Какие планы на съёмки?",
    "Что думаешь про современных режиссёров?",
    "Как у тебя с деньгами?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/eldar-persona-3b")
    ap.add_argument("--val-limit", type=int, default=0, help="0 = вся val-выборка")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    rows = load_jsonl(VAL)
    if args.val_limit:
        rows = rows[: args.val_limit]
    raw = Dataset.from_list(rows)
    ds = raw.map(
        tokenize, batched=True,
        fn_kwargs=dict(tokenizer=tokenizer, max_len=1536),
        remove_columns=raw.column_names,
    )
    collator = Collator(tokenizer.pad_token_id)
    loss = test_loss(model, ds, collator, batch_size=1)
    print(f"val loss (persona): {loss:.4f}  ({len(ds)} примеров)\n")

    system = load_jsonl(os.path.join(
        PROJECT, "persona_eldar", "data", "eldar_train.jsonl"))[0]["messages"][0]["content"]
    print("=== Генерации на контрольные вопросы ===")
    for q in PROBES:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": q},
        ]
        answer = generate(model, tokenizer, messages, max_new_tokens=250)
        print(f"Q: {q}\nA: {answer}\n{'-' * 60}")


if __name__ == "__main__":
    main()
