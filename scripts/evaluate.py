"""Финальная оценка обученной модели: test loss (token-weighted) + примеры генераций.

Запуск (из корня проекта):
  python scripts/evaluate.py --model models/med-dialogue-3b --preset dialogues
  python scripts/evaluate.py --model models/med-cot-3b --preset cot
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Collator, PRESETS, ROOT, load_jsonl, tokenize  # noqa: E402


@torch.inference_mode()
def test_loss(model, ds, collator, batch_size):
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=collator)
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1]
        tgt = batch["labels"][:, 1:]
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
            ignore_index=-100, reduction="sum",
        )
        total += float(ce)
        count += int((tgt != -100).sum())
    return total / max(count, 1)


@torch.inference_mode()
def generate(model, tokenizer, messages, max_new_tokens):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--preset", choices=PRESETS, required=True)
    ap.add_argument("--limit", type=int, default=3, help="сколько генераций показать")
    args = ap.parse_args()
    cfg = PRESETS[args.preset]
    name = os.path.basename(os.path.normpath(args.model))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    rows = load_jsonl(os.path.join(ROOT, f"data/{args.preset}_test.jsonl"))
    from datasets import Dataset
    raw = Dataset.from_list(rows)
    proc = raw.map(tokenize, batched=True,
                   fn_kwargs=dict(tokenizer=tokenizer, max_len=cfg["max_len"]),
                   remove_columns=raw.column_names)
    collator = Collator(tokenizer.pad_token_id)
    tl = test_loss(model, proc.remove_columns([c for c in proc.column_names
                                               if c not in ("input_ids", "labels")]),
                   collator, batch_size=2)

    lines = [f"=== Оценка {name} (preset={args.preset}, test={len(rows)}) ===",
             f"test loss (token-weighted): {tl:.4f}", ""]
    max_new = 400 if args.preset == "dialogues" else 700
    for i, row in enumerate(rows[: args.limit], 1):
        gen = generate(model, tokenizer, row["messages"][:2], max_new)
        ref = row["messages"][2]["content"]
        lines += [f"--- пример {i} ---",
                  f"ВОПРОС:     {row['messages'][1]['content'][:300]}",
                  f"ГЕНЕРАЦИЯ:  {gen[:600]}",
                  f"ЭТАЛОН:     {ref[:600]}", ""]

    report = "\n".join(lines)
    print(report)
    out_path = os.path.join(ROOT, "runs", f"evaluation_{name}.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"отчёт: {out_path}")


if __name__ == "__main__":
    main()
