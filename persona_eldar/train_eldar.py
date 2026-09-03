"""Полный SFT-файнтюн Qwen2.5-3B-Instruct на персоне Эльдара Богунова.

Датасет: persona_eldar/data/eldar_{train,val}.jsonl (771 монолог из телеграм-канала
«Творчество Эльдара Богунова», 2023–2026; подготовка — eldar_dataset/prepare_dataset.py).
Формат тот же, что у медицинских моделей: JSONL с `messages` (system/user/assistant),
лосс только по ответу ассистента.

Переиспользует движок scripts/train.py (unsloth full FT с откатом на transformers),
медицинские пресеты не затрагиваются. Полностью независимая модель от общей базы.

Запуск (из корня проекта):
  PYTHONUNBUFFERED=1 .venv/bin/python persona_eldar/train_eldar.py 2>&1 | tee runs/train_eldar.log
"""

import argparse
import os
import sys

# Фикс OOM на fused CE-лоссе unsloth: сразу после eval детектор свободной VRAM
# ошибается из-за фрагментации, и буфер ло́гитов (151936 x seq ~594 МБ) не
# помещается. expandable_segments возвращает фрагментированные ~1,1 ГБ,
# N_CHUNKS=4 режет буфер ло́гитов на 4 части (~150 МБ каждая).
# Ставим ДО импорта torch — аллокатор читает это при инициализации CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("UNSLOTH_CE_LOSS_N_CHUNKS", "4")

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "scripts"))

from datasets import Dataset
from transformers import Trainer, TrainingArguments

from train import Collator, ROOT, load_jsonl, load_model, tokenize  # noqa: E402

PRESET = {
    # v3 = перекладка вопросов (вопрос реально отвечает монологу) + фактовые QA
    # + якоря x2; генерация: make_train_v3.py. Данных вдвое больше, поэтому
    # 2 эпохи: столько же проходов по каждому тексту, меньше слепого заучивания.
    "train": "persona_eldar/data/eldar_train_v3.jsonl",
    "val": "persona_eldar/data/eldar_val_v3.jsonl",
    "run_name": "eldar-persona-3b",
    # медиана 543 токена, p95 1235: при 1536 обрезается 8 примеров из 733
    "max_len": 1536,
    "epochs": 2.0,
    "batch": 1,
    "accum": 16,
    "lr": 2e-5,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["auto", "unsloth", "transformers"], default="auto")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="продолжить с последнего чекпоинта в runs/<run_name>/")
    args = ap.parse_args()
    cfg = dict(PRESET)
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
        save_steps=50,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=6,               # 3% от ~200 шагов (warmup_ratio удалён в transformers 5.2)
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
