"""Чат с копией Эльдара + RAG: поиск по подлинным постам (без внешних зависимостей).

Перед каждым ответом ищет топ-2 его реальных текста по теме вопроса
(TF-IDF по символьным 3-граммам — устойчив к русской морфологии:
«снять/съемка/сьемки» дают пересекающиеся граммы) и подставляет их
в системный промпт. Модель не «вспоминает» факты из весов, а пересказывает
своим стилем реальные посты — релевантность и фактология резко стабильнее.

Запуск:
  .venv/bin/python persona_eldar/chat_eldar_rag.py --model models/eldar-persona-3b
"""

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT = open(
    os.path.join(HERE, "data", "system_prompt.txt"), encoding="utf-8"
).read().strip()

if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- retrieval

def ngrams(s, n=3):
    """Символьные 3-граммы (морфология: снять/съемка/сьемки) + целые слова
    от 4 букв (точные совпадения: «кино», «дым» не тонут в частых граммах)."""
    s = re.sub(r"\s+", " ", s.lower())
    return [s[i:i + n] for i in range(len(s) - n + 1)] + re.findall(r"\w{4,}", s)


class CorpusIndex:
    """Два TF-IDF-вектора на документ (вопросы v3 и текст ответа) и комбинированная
    близость: cos(запрос, вопросы) + 0.3 * cos(запрос, ответ). Разделение убирает
    перекос по длине: длинный монолог не «размазывает» точное совпадение вопроса,
    короткий текст не выигрывает только за счёт маленькой нормы."""

    def __init__(self, docs, answer_weight=0.4):
        self.docs = docs
        self.answer_weight = answer_weight
        self.vecs = []
        df = Counter()
        for d in docs:
            tf_q = Counter(ngrams(d.get("q", "")))
            tf_a = Counter(ngrams(d["a"] if "a" in d else d["text"]))
            self.vecs.append((tf_q, tf_a))
            for tf in (tf_q, tf_a):
                for g in tf:
                    df[g] += 1
        n = max(len(docs), 1)
        self.idf = {g: math.log((n + 1) / (c + 1)) + 1.0 for g, c in df.items()}

    def _vec(self, tf):
        v = {g: c * self.idf.get(g, 0.0) for g, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {g: x / norm for g, x in v.items()}

    def _cos(self, qv, dv):
        return sum(w * dv.get(g, 0.0) for g, w in qv.items())

    def search(self, query, k=2, max_chars=700):
        qv = self._vec(Counter(ngrams(query)))
        best = []
        for i, (tf_q, tf_a) in enumerate(self.vecs):
            score = self._cos(qv, self._vec(tf_q)) + self.answer_weight * self._cos(qv, self._vec(tf_a))
            best.append((score, i))
        best.sort(reverse=True)
        out = []
        for score, i in best[:k]:
            d = self.docs[i]
            text = d["answer"] if "answer" in d else d["text"]
            if len(text) > max_chars:  # обрезаем по границе строки
                cut = text[:max_chars]
                text = cut[: cut.rfind("\n")].rstrip() if "\n" in cut else cut
            out.append({"score": round(score, 3), "date": self.docs[i].get("date"),
                        "text": text})
        return out


def load_index():
    """Корпус для поиска: уникальные монологи + фактовые ответы из v3.

    В индексируемый текст включаются и переложенные вопросы v3 (их монолог
    реально отвечает) — поиск становится «вопрос -> вопрос», и находится
    материал, в котором нет буквальных слов запроса (цены без слова «тайланд»).
    Тема сохраняется — из «фирменных» берётся случайный пост-соскок.
    """
    path = os.path.join(HERE, "data", "eldar_train_v3.jsonl")
    by_answer = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cat = r.get("meta", {}).get("category", "")
            if cat not in ("monologue_v3", "fact_qa"):
                continue
            question = r["messages"][1]["content"]
            answer = r["messages"][2]["content"]
            topic = r["meta"].get("topic", "")
            entry = by_answer.setdefault(
                answer[:80],
                {"answer": answer, "questions": [], "date": r["meta"].get("date"), "topic": topic},
            )
            if question not in entry["questions"]:
                entry["questions"].append(question)
    docs = [
        {"q": " ".join(e["questions"]),
         "a": e["answer"],
         "answer": e["answer"], "date": e["date"], "topic": e["topic"]}
        for e in by_answer.values()
    ]
    return CorpusIndex(docs)




def build_fragment_bank(index):
    """Банк фрагментов (1-3 строки, 30-200 символов) подлинных постов по темам —
    материал для монтажа «10 случайных фраз на основе фактов»."""
    rng = random.Random(777)
    bank = {}
    for d in index.docs:
        topic = d.get("topic") or "мысли"
        lines = [l.strip() for l in d["answer"].split("\n") if l.strip()]
        if not lines:
            continue
        for _ in range(2):
            start = rng.randrange(len(lines))
            frag = "\n".join(lines[start:start + rng.choice((1, 2, 2, 3))])
            if 30 <= len(frag) <= 200 and not re.search(r"https?://|www\.|@", frag):
                bank.setdefault(topic, []).append(frag)
    return bank


def montage_fragments(bank, rng, n=8, exclude=()):
    """n случайных фрагментов с разных тем — «мысли Эльдара на сейчас»."""
    topics = [t for t in bank if bank[t]]
    if not topics:
        return []
    chosen = rng.sample(topics, min(n, len(topics)))
    out = []
    for t in chosen:
        frag = rng.choice(bank[t])
        if frag[:40] not in exclude:
            out.append((t, frag))
    return out


def build_context(hits, frags=(), question=None):
    parts = []
    if question:
        parts.append(
            f"Вопрос был: {question}\n"
            "Первые строки ответа — про это. Ниже твои настоящие посты по теме — "
            "бери факты оттуда:"
        )
    for h in hits:
        parts.append("---\n" + h["text"])
    if frags:
        parts.append(
            "А это твои мысли прямо сейчас — после ответа на вопрос прыгай между "
            "ними, пересказывай близко к тексту, своими словами, в своём стиле, "
            "можно в любом порядке и обрывать на полуслове темы:"
        )
        for i, (topic, frag) in enumerate(frags, 1):
            parts.append(f"{i}. {frag}")
    return "\n".join(parts)


# ---------------------------------------------------------------- trim (как в chat_eldar.py)

def trim_loop(text: str) -> str:
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
        if sum(dup_flags[max(0, i - 7): i + 1]) >= 3:
            return "\n".join(lines[:i]).rstrip() + "\n\n(обрезано: дальше шло по кругу)"
    return text


# строки-«числа»: суммы и счёт фильмам — из них модель строит спирали
NUM_LINE = re.compile(r"\b\d+\s*(р\b|к\b|руб\w*|тысяч\w*|лям\w*|бат\w*|фильм\w*|социалк\w*)", re.I)


def trim_number_spiral(text: str) -> str:
    """Отрезает числовую спираль: хвост, где >=7 из последних 10 строк — суммы
    или счёт фильмам. Ответ про цены с самого начала не трогаем (легитимный)."""
    lines = text.split("\n")
    flags = [bool(NUM_LINE.search(l)) for l in lines]
    W, NEED, MIN_TAIL = 12, 9, 6  # калибровка: спираль ловится, ложных 0,9% по корпусу
    for i in range(W - 1, len(lines)):
        if sum(flags[i + 1 - W: i + 1]) < NEED:
            continue
        j = i
        while j > 0 and flags[j - 1]:
            j -= 1
        if j < 5 or i - j + 1 < MIN_TAIL:
            continue
        return "\n".join(lines[:j]).rstrip() + "\n\n(обрезано: дальше перечисление цифр пошло по кругу)"
    return text


# ---------------------------------------------------------------- main

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/eldar-persona-3b")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=800)
    ap.add_argument("--show-sources", action="store_true", help="печатать найденные посты")
    ap.add_argument("--top-k", type=int, default=2, help="сколько постов по теме подставлять")
    ap.add_argument("--frags", type=int, default=9, help="сколько случайных фраз в монтаже")
    ap.add_argument("--no-jump", action="store_true", help="только посты по теме, без монтажа")
    ap.add_argument("--mode", choices=["montage", "generate"], default="montage",
                    help="montage: ответ собирается из подлинных строк (100% стиль и факты); "
                         "generate: модель пересказывает монтаж своими словами")
    args = ap.parse_args()

    rng = random.Random()
    style = open(os.path.join(HERE, "data", "system_style.txt"), encoding="utf-8").read().strip()
    print("Индексация корпуса…")
    index = load_index()
    bank = build_fragment_bank(index)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    history = []
    print(f"Чат с Эльдаром (RAG по {len(index.docs)} подлинным постам). /exit — выход.\n")
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

        hits = index.search(user, k=args.top_k)
        frags = () if args.no_jump else montage_fragments(
            bank, rng, n=args.frags, exclude={h["text"][:40] for h in hits}
        )

        if args.mode == "montage":
            # детерминированный ответ: первые строки тематического поста +
            # случайные подлинные фразы — стиль и факты гарантированы
            lines = []
            if hits and hits[0]["score"] > 0.05:
                head = [l for l in hits[0]["text"].split("\n") if l.strip()][:2]
                lines += head
            for _, frag in frags:
                lines += [l for l in frag.split("\n") if l.strip()]
            answer = "\n".join(lines)
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": answer})
            print(f"\nЭльдар: {answer}\n")
            continue

        if args.show_sources:
            for h in hits:
                print(f"  [источник {h['date']}] {h['text'][:120]}…")
            for topic, frag in frags:
                print(f"  [мысль/{topic}] {frag[:110]}…")

        system = SYSTEM_PROMPT + "\n\n" + style
        if hits and hits[0]["score"] > 0.05:
            system += "\n\n" + build_context(hits, frags, question=user)
        history.append({"role": "user", "content": user})
        trimmed = [ {"role": "system", "content": system} ] + history[-8:]
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
            no_repeat_ngram_size=6,
            pad_token_id=tokenizer.pad_token_id,
        )
        answer = trim_loop(tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ))
        answer = trim_number_spiral(answer)
        history.append({"role": "assistant", "content": answer})
        print(f"\nЭльдар: {answer}\n")


if __name__ == "__main__":
    main()
