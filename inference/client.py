"""CLI-клиент API-сервера медицинских моделей (только stdlib).

Примеры:
  python inference/client.py --mode health
  python inference/client.py --mode dialogue --question "У меня колет в боку при беге, это опасно?"
  python inference/client.py --mode cot --question "Какой вазопрессор первой линии при септическом шоке?"
  python inference/client.py --mode joint --question "Мучает изжога по ночам, что делать?"
"""

import argparse
import json
import urllib.request


def call(port, mode, question=None):
    if mode == "health":
        url = f"http://127.0.0.1:{port}/api/health"
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read())
    url = f"http://127.0.0.1:{port}/api/{mode}"
    body = json.dumps({"question": question}).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    timeout = 300 if mode == "joint" else 180  # joint = три генерации подряд
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def pretty(res):
    if "pipeline" in res:
        return (f"Клинический вопрос:\n{res['clinical_question']}\n\n"
                f"Разбор (med-cot-3b):\n{res['cot_analysis']}\n\n"
                f"Итоговый ответ пациенту:\n{res['final_answer']}")
    return res.get("answer", json.dumps(res, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["health", "dialogue", "cot", "joint"], required=True)
    ap.add_argument("--question", default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--raw", action="store_true", help="печать сырого JSON")
    args = ap.parse_args()
    res = call(args.port, args.mode, args.question)
    print(json.dumps(res, ensure_ascii=False, indent=2) if args.raw else pretty(res))


if __name__ == "__main__":
    main()
