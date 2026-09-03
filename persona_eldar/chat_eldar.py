"""Интерактивный чат с моделью-персоной Эльдара Богунова.

Запуск (из корня проекта):
  .venv/bin/python persona_eldar/chat_eldar.py --model models/eldar-persona-3b

Выход: /exit или Ctrl+C. История диалога хранится в памяти сессии.
"""

import argparse
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT = open(
    os.path.join(HERE, "data", "system_prompt.txt"), encoding="utf-8"
).read().strip()
# поведенческая надстройка: ответ на вопрос -> живой поток с темы на тему
SYSTEM_PROMPT += "\n\n" + open(
    os.path.join(HERE, "data", "system_style.txt"), encoding="utf-8"
).read().strip()

# строки-«числа»: суммы и счёт фильмам — из них модель строит спирали
NUM_LINE = re.compile(r"\b\d+\s*(р\b|к\b|руб\w*|тысяч\w*|лям\w*|бат\w*|фильм\w*|социалк\w*)", re.I)


# битые байты в вставке текста (обрыв UTF-8) не должны убивать сессию
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")


def trim_number_spiral(text: str) -> str:
    """Отрезает числовую спираль: хвост, где >=7 из последних 10 строк —
    суммы/счёт («на 10к... на 100к... за год 3 миллиона...»). Начало ответа
    про цены не трогаем — режем только когда спираль началась посреди текста
    и перед ней есть >=5 нормальных строк."""
    lines = text.split("\n")
    flags = [bool(NUM_LINE.search(l)) for l in lines]
    W, NEED, MIN_TAIL = 12, 9, 6  # калибровка: спираль ловится, ложных 0,9% по корпусу
    for i in range(W - 1, len(lines)):
        if sum(flags[i + 1 - W: i + 1]) < NEED:
            continue
        # идём назад до начала непрерывной числовой полосы
        j = i
        while j > 0 and flags[j - 1]:
            j -= 1
        if j < 5:  # цены с самого начала — это легитимный ответ про цены
            continue
        cut = "\n".join(lines[:j]).rstrip()
        return cut + "\n\n(обрезано: дальше перечисление цифр пошло по кругу)"
    return text


def trim_loop(text: str) -> str:
    """Обрезает хвост, если генерация пошла по кругу.

    3B-модель на длинных выдачах (его реальные тексты в основном <=600 токенов)
    выходит за распределение обучения и загоняется в повторы строк-перифраз:
    в нормальном тексте жаккар строки с любой предыдущей <= 0.3, в петле
    регулярно >= 0.35. Режем на первом повторе, если в окне из 8 строк таких
    набирается >= 3 (одно случайное совпадение ответ не обрежет).
    """
    lines = text.split("\n")
    bags = [frozenset(re.findall(r"\w+", l.lower())) for l in lines]

    def best_jaccard(i):
        return max(
            (len(bags[i] & bags[j]) / max(len(bags[i] | bags[j]), 1) for j in range(i)),
            default=0.0,
        )

    dup_flags = [False] * len(lines)
    for i in range(5, len(lines)):
        if len(bags[i]) >= 3 and best_jaccard(i) >= 0.35:
            dup_flags[i] = True

    for i in range(len(lines)):
        if not dup_flags[i]:
            continue
        window = dup_flags[max(0, i - 7): i + 1]
        if sum(window) >= 3:
            cut = "\n".join(lines[:i]).rstrip()
            return cut + "\n\n(обрезано: дальше шло по кругу)"
    return text


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/eldar-persona-3b")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=2000)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Чат с Эльдаром (цифровая копия). /exit — выход.\n")
    while True:
        try:
            user = input("Ты: ").strip()
        except UnicodeDecodeError:
            print("вставка пришла с битой кодировкой, попробуйте ещё раз\n")
            continue
        except (KeyboardInterrupt, EOFError):
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        history.append({"role": "user", "content": user})
        # держим последние 8 реплик + системный промпт, чтобы не расти контекст
        trimmed = [history[0]] + history[-8:]
        text = tokenizer.apply_chat_template(
            trimmed, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=6,   # жёсткий запрет 6-граммных петель ("кино и социалки")
            pad_token_id=tokenizer.pad_token_id,
        )
        answer = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        answer = trim_loop(answer)
        answer = trim_number_spiral(answer)
        history.append({"role": "assistant", "content": answer})
        print(f"\nЭльдар: {answer}\n")


if __name__ == "__main__":
    main()
