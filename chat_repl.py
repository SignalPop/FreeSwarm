"""Interactive terminal chat against the running FreeToken server (:1919).
Run:  .venv\\Scripts\\python chat_repl.py   (from the repo root, with a model loaded)
Type your message and hit enter. Commands: /reset clears history, /quit exits.
Streams tokens live. Thinking (reasoning) is shown dimmed; the answer follows.
"""
import json, sys, urllib.request
URL = "http://127.0.0.1:1919/v1/chat/completions"
MODEL = "Qwen3.6-35B-A3B-NVFP4"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

hist = []
print(f"chat with {MODEL} (/reset, /quit)\n")
while True:
    try:
        msg = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); break
    if not msg:
        continue
    if msg == "/quit":
        break
    if msg == "/reset":
        hist = []; print("(history cleared)\n"); continue
    hist.append({"role": "user", "content": msg})
    body = json.dumps({"model": MODEL, "messages": hist,
                       "max_tokens": 1024, "temperature": 0.7, "stream": True}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    answer, in_think, think_open = [], False, False
    sys.stdout.write("bot> "); sys.stdout.flush()
    for line in urllib.request.urlopen(req, timeout=300):
        s = line.decode("utf-8", "replace").strip()
        if not s.startswith("data:"):
            continue
        d = s[5:].strip()
        if d == "[DONE]":
            break
        try:
            delta = json.loads(d)["choices"][0]["delta"]
        except Exception:
            continue
        r = delta.get("reasoning_content")
        c = delta.get("content")
        if r:
            if not think_open:
                sys.stdout.write("\x1b[2m[thinking] "); think_open = True
            sys.stdout.write(r)
        if c:
            if think_open:
                sys.stdout.write("\x1b[0m\n"); think_open = False
            sys.stdout.write(c); answer.append(c)
        sys.stdout.flush()
    if think_open:
        sys.stdout.write("\x1b[0m")
    print("\n")
    hist.append({"role": "assistant", "content": "".join(answer)})
