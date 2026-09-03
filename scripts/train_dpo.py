"""DPO-этап для med-cot-3b: предпочтение «хороший итог» над «черновиком».

Данные: data/cot_dpo_reserved.jsonl (зарезервировано в prepare_data.py):
  prompt   — [system, user] сообщения
  chosen   — полный ответ (### Рассуждение ... ### Итоговый ответ)
  rejected — черновик (old_thoughts / raw_answer, помечены автором датасета
             как менее качественные)

Память: полный файнтюн-DPO требует референсную копию модели (+6,2 ГБ) и в 24 ГБ
не помещается, поэтому этап выполнен как LoRA-DPO (r=32) поверх SFT-модели с
последующим слиянием — итог всё равно полная самостоятельная модель
models/<run_name>.

Запуск (после завершения шага 2):
  python scripts/train_dpo.py --model models/med-cot-3b
"""

import argparse
import json
import os
import random
import subprocess
import sys

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

MAX_PROMPT_CHARS = 3_000
MAX_COMPLETION_CHARS = 12_000
MAX_PROMPT_TOKENS = 768
MAX_COMPLETION_TOKENS = 1408


def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_dataset(path, limit, seed=42):
    rows, dropped = [], {"same": 0, "len": 0}
    for ex in load_pairs(path):
        chosen, rejected = ex["chosen"].strip(), ex["rejected"].strip()
        user = ex["prompt"][1]["content"]
        if chosen == rejected:
            dropped["same"] += 1
            continue
        if len(user) > MAX_PROMPT_CHARS or len(chosen) > MAX_COMPLETION_CHARS \
                or len(rejected) > MAX_COMPLETION_CHARS:
            dropped["len"] += 1
            continue
        rows.append({
            "prompt": ex["prompt"],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_val = max(1, round(len(rows) * 0.05))
    print(f"пар: {len(rows)} (отброшено: {dropped}); "
          f"train {len(rows) - n_val} / val {n_val}, limit={limit}")
    train, val = rows[n_val:], rows[:n_val]
    if limit:
        train = train[:limit]
    return Dataset.from_list(train), Dataset.from_list(val)


def unsloth_selfcheck():
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "check_engine.py"), "--mode", "lora"],
        capture_output=True, text=True, timeout=600,
    )
    tail = (r.stdout + r.stderr).strip().splitlines()
    print(f">>> самотест unsloth/lora: {tail[-1] if tail else 'нет вывода'}")
    return r.returncode == 0


def load_model(model_path, max_len, engine):
    use_unsloth = engine == "unsloth" or (engine == "auto" and unsloth_selfcheck())
    if use_unsloth:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path, max_seq_length=max_len,
            dtype=torch.bfloat16, load_in_4bit=False,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=32, target_modules=LORA_TARGETS, lora_alpha=64,
            lora_dropout=0.05, bias="none", use_gradient_checkpointing="unsloth",
        )
        print(">>> движок: unsloth (LoRA-DPO)")
        return model, tokenizer

    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda"
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    ))
    print(">>> движок: transformers+peft (LoRA-DPO)")
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "med-cot-3b"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "cot_dpo_reserved.jsonl"))
    ap.add_argument("--run-name", default="med-cot-3b-dpo")
    ap.add_argument("--engine", choices=["auto", "unsloth", "transformers"], default="auto")
    ap.add_argument("--limit", type=int, default=2000,
                    help="сколько обучающих пар взять (0 = все)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=1,
                    help="пар на устройство; 1 по умолчанию: при 2 случается OOM "
                         "на fp32-конвертации логитов DPO (vocab 152K)")
    args = ap.parse_args()

    train_ds, val_ds = build_dataset(args.data, args.limit if args.limit > 0 else None)
    max_len = MAX_PROMPT_TOKENS + MAX_COMPLETION_TOKENS
    model, tokenizer = load_model(args.model, max_len, args.engine)
    model.config.use_cache = False

    from trl import DPOConfig, DPOTrainer

    out_dir = os.path.join(ROOT, "runs", args.run_name)
    cfg = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=max(1, 8 // args.batch),  # эффективный батч 8 пар
        per_device_eval_batch_size=2,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        beta=args.beta,
        max_prompt_length=MAX_PROMPT_TOKENS,
        max_completion_length=MAX_COMPLETION_TOKENS,
        max_length=max_len,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to=[],
        seed=42,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # peft: референс = модель с выключенным адаптером
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )
    trainer.train()

    merged_dir = os.path.join(ROOT, "models", args.run_name)
    model = model.merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"ГОТОВО: полная DPO-модель сохранена в {merged_dir}")


if __name__ == "__main__":
    main()
