"""Интерактивный чат с моделями через API-сервер (только stdlib).

Запуск (сервер должен работать):
  .venv/bin/python inference/chat.py

Команды внутри: 1/2/3 — выбор модели (диалоговая / клиническая / joint-конвейер),
/q — выход. Всё остальное воспринимается как вопрос.
"""

import json
import sys
import urllib.request

MODES = {"0": ("base", "базовая Qwen2.5-3B-Instruct (для сравнения)"),
         "1": ("dialogue", "мед диалоговая (триаж)"),
         "2": ("cot", "мед клиническая (рассуждения)"),
         "3": ("joint", "joint-конвейер (3 генерации, до ~90 с)"),
         "4": ("wrap", "wrap: клинический разбор + итог пациенту (~60 с)")}


def ask(port, mode, question):
    body = json.dumps({"question": question}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/{mode}", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    timeout = 300 if mode in ("joint", "wrap") else 180
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def pretty(mode, res):
    if mode in ("joint", "wrap"):
        parts = [f"\n─ Клинический вопрос:\n{res['clinical_question']}\n"] if "clinical_question" in res else []
        parts.append(f"\n─ Разбор (med-cot-3b):\n{res['cot_analysis']}\n")
        parts.append(f"\n─ Итог пациенту:\n{res['final_answer']}\n")
        return "".join(parts)
    return f"\n{res['answer']}\n"


def main():
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as r:
            if json.loads(r.read()).get("status") != "ok":
                print("сервер ещё загружается, подождите и перезапустите чат")
                return
    except Exception as e:
        print(f"сервер на порту {port} недоступен ({e})\nзапуск: "
              f".venv/bin/uvicorn inference.server:app --host 127.0.0.1 --port {port}")
        return

    print("Чат с медицинскими моделями. Модель: [0] базовая  [1] диалоговая  "
          "[2] клиническая  [3] joint  [4] wrap · переключение — 0/1/2/3/4 · выход — /q")
    mode, _ = MODES["1"]
    print(f"→ активна: {MODES['1'][1]}\n")
    while True:
        try:
            q = input("вопрос> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        if q in ("/q", "/quit", "/exit"):
            return
        if q in MODES:
            mode = MODES[q][0]
            print(f"→ активна: {MODES[q][1]}\n")
            continue
        try:
            print(pretty(mode, ask(port, mode, q)))
        except Exception as e:
            print(f"\n! ошибка запроса: {e}\n")


if __name__ == "__main__":
    main()
