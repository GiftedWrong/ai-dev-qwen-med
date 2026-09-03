"""Самотест движка обучения: unsloth, один микрошаг forward+backward.

Режимы:
  full (default) — полный файнтюн (from_pretrained(full_finetuning=True))
  lora           — LoRA (для DPO-этапа)

Запускается в отдельном процессе (import unsloth необратимо патчит transformers
в текущем процессе). Код возврата 0 — движок работоспособен.

Запуск:
  python scripts/check_engine.py [--mode lora]
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "Qwen2.5-3B-Instruct")

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "lora"], default="full")
    args = ap.parse_args()
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE,
            max_seq_length=512,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            **({"full_finetuning": True} if args.mode == "full" else {}),
        )
        if args.mode == "lora":
            model = FastLanguageModel.get_peft_model(
                model,
                r=32,
                target_modules=LORA_TARGETS,
                lora_alpha=64,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
        ids = tokenizer(
            "Тест движка обучения: выполняем один микрошаг.",
            return_tensors="pt",
        ).to("cuda")
        out = model(**ids, labels=ids["input_ids"])
        out.loss.backward()
        print(f"ENGINE OK: unsloth {args.mode}, loss={float(out.loss):.4f}")
        return 0
    except Exception as e:
        print(f"ENGINE FAIL: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
