"""Полный SFT-файнтюн Qwen2.5-3B-Instruct на одном из медицинских датасетов.

Пайплайн: JSONL с `messages` -> ChatML-токенизация Qwen -> полный файнтюн
(лосс только по ответу ассистента) -> полная самостоятельная модель
в models/<run_name>.

Движок (--engine):
  auto        — сначала самотест unsloth в отдельном процессе (check_engine.py);
                прошёл -> unsloth, нет -> чистый transformers
  unsloth     — ускоренный полный файнтюн (from_pretrained(full_finetuning=True))
  transformers— без unsloth (его import пропатчит transformers и сломает обучение)

Запуск:
  python scripts/train.py --preset dialogues
  python scripts/train.py --preset cot
"""

import argparse
import json
import os
import subprocess
import sys

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "Qwen2.5-3B-Instruct")

PRESETS = {
    "dialogues": {
        "train": "data/dialogues_train.jsonl",
        "val": "data/dialogues_val.jsonl",
        "run_name": "med-dialogue-3b",
        "max_len": 1024,
        "epochs": 2.0,
        "batch": 2,
        "accum": 8,
        "lr": 2e-5,
    },
    "cot": {
        "train": "data/cot_train.jsonl",
        "val": "data/cot_val.jsonl",
        "run_name": "med-cot-3b",
        "max_len": 2048,
        "epochs": 1.0,
        "batch": 1,
        "accum": 16,
        "lr": 1e-5,
    },
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chat_ids(tokenizer, messages, add_generation_prompt=False):
    """ChatML-id через рендер в строку: apply_chat_template(tokenize=True) в
    transformers 5.x возвращает BatchEncoding вместо списка id, а unsloth-токенизатор
    — список; строковый рендер + add_special_tokens=False одинаков во всех версиях."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def tokenize(examples, tokenizer, max_len):
    """ChatML-токенизация с маскированием промпта: labels=-100 вне ответа ассистента."""
    input_ids, labels = [], []
    for messages in examples["messages"]:
        full = chat_ids(tokenizer, messages)
        prompt = chat_ids(tokenizer, messages[:-1], add_generation_prompt=True)
        ids = full[: max_len]
        lab = [-100] * min(len(prompt), len(ids)) + ids[len(prompt):]
        if all(l == -100 for l in lab):
            continue
        input_ids.append(ids)
        labels.append(lab)
    return {"input_ids": input_ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in feats:
            ids, lab = f["input_ids"], f["labels"]
            pad = n - len(ids)
            batch["input_ids"].append(ids + [self.pad_id] * pad)
            batch["attention_mask"].append([1] * len(ids) + [0] * pad)
            batch["labels"].append(lab + [-100] * pad)
        return {k: torch.tensor(v) for k, v in batch.items()}


def unsloth_selfcheck() -> bool:
    """Самотест unsloth в отдельном процессе: import unsloth глобально патчит
    transformers, поэтому проверять и при сбое откатываться нужно до импорта."""
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "check_engine.py")],
        capture_output=True, text=True, timeout=600,
    )
    tail = (r.stdout + r.stderr).strip().splitlines()
    print(f">>> самотест unsloth: {tail[-1] if tail else 'нет вывода'}")
    return r.returncode == 0


def load_model(max_len, engine):
    """Полный файнтюн: unsloth (ускорение) или чистый transformers."""
    use_unsloth = engine == "unsloth" or (engine == "auto" and unsloth_selfcheck())
    if use_unsloth:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE,
            max_seq_length=max_len,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            full_finetuning=True,
        )
        print(">>> движок: unsloth (полный файнтюн)")
        return model, tokenizer

    import transformers as _tf
    major = int(_tf.__version__.split(".")[0])
    kw = {"dtype" if major >= 5 else "torch_dtype": torch.bfloat16}
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, **kw)
    model.enable_input_require_grads()
    print(f">>> движок: transformers {_tf.__version__} (полный файнтюн)")
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, required=True)
    ap.add_argument("--engine", choices=["auto", "unsloth", "transformers"], default="auto")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--batch", type=int, default=None, help="переопределить per-device batch")
    ap.add_argument("--max-len", type=int, default=None, help="переопределить max_len")
    ap.add_argument("--resume", action="store_true",
                    help="продолжить с последнего чекпоинта в runs/<run_name>/")
    args = ap.parse_args()
    cfg = dict(PRESETS[args.preset])
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch:
        cfg["batch"] = args.batch
    if args.max_len:
        cfg["max_len"] = args.max_len
    run_name = args.run_name or cfg["run_name"]

    model, tokenizer = load_model(cfg["max_len"], args.engine)
    model.config.use_cache = False

    raw = Dataset.from_list(load_jsonl(os.path.join(ROOT, cfg["train"])))
    raw_val = Dataset.from_list(load_jsonl(os.path.join(ROOT, cfg["val"])))
    kw = dict(tokenizer=tokenizer, max_len=cfg["max_len"])
    ds = raw.map(tokenize, batched=True, fn_kwargs=kw, remove_columns=raw.column_names)
    ds_val = raw_val.map(tokenize, batched=True, fn_kwargs=kw, remove_columns=raw_val.column_names)
    print(f"train: {len(ds)} (из {len(raw)}), val: {len(ds_val)} (из {len(raw_val)}); "
          f"фильтром по длине {cfg['max_len']} отброшено "
          f"{len(raw) - len(ds) + len(raw_val) - len(ds_val)}")

    out_dir = os.path.join(ROOT, "runs", run_name)
    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch"],
        gradient_accumulation_steps=cfg["accum"],
        per_device_eval_batch_size=cfg["batch"],
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=250,               # чекпоинт раз в 250 шагов (~1 ч cot)
        save_total_limit=2,
        logging_steps=10,
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        max_grad_norm=1.0,
        report_to=[],
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        eval_dataset=ds_val,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    merged_dir = os.path.join(ROOT, "models", run_name)
    model.config.use_cache = True
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"ГОТОВО: полная модель сохранена в {merged_dir}")


if __name__ == "__main__":
    main()
