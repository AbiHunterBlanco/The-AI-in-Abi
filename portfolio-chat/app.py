import json
import os
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")  # optional, e.g. "llama3.2:1b". Unset = retrieval mode.
THRESHOLD = 0.25          # minimum match score to answer
MAX_INPUT_CHARS = 500
RATE_LIMIT = 30           # messages per hour per IP
FALLBACK = "HMMM... I don't know! c:"

app = Flask(__name__, static_folder="static")
hits = defaultdict(list)


def load_qa():
    """Parse qa.md into [(list_of_questions, answer), ...]. Re-read each request so edits apply instantly."""
    entries = []
    for block in (BASE / "qa.md").read_text().split("\n---"):
        qs, ans, in_a = [], [], False
        for line in block.strip().splitlines():
            if line.startswith("Q:") and not in_a:
                qs.append(line[2:].strip())
            elif line.startswith("A:"):
                in_a = True
                ans.append(line[2:].strip())
            elif in_a:
                ans.append(line)
        if qs and ans:
            entries.append((qs, "\n".join(ans).strip()))
    return entries


def retrieval_reply(question):
    entries = load_qa()
    triggers, owners = [], []
    for i, (qs, _) in enumerate(entries):
        for q in qs:
            triggers.append(q)
            owners.append(i)
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vec.fit_transform(triggers)
        scores = cosine_similarity(vec.transform([question]), matrix)[0]
    except ValueError:
        return FALLBACK
    best = scores.argmax()
    return entries[owners[best]][1] if scores[best] >= THRESHOLD else FALLBACK


def ollama_reply(messages):
    facts = "\n\n".join(f"Q: {' / '.join(qs)}\nA: {a}" for qs, a in load_qa())
    style = (BASE / "style.md").read_text()
    system = (
        "You speak as the person in the FACTS below, in the first person, on their portfolio site. "
        "Only use the FACTS; if something isn't covered, say you're not sure. Keep answers to 1-3 sentences.\n\n"
        f"FACTS\n{facts}\n\nSPEECH STYLE\n{style}"
    )
    body = json.dumps({
        "model": OLLAMA_MODEL, "stream": False,
        "messages": [{"role": "system", "content": system}] + messages,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat", body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["message"]["content"]


def rate_limited(ip):
    now = time.time()
    hits[ip] = [t for t in hits[ip] if now - t < 3600]
    if len(hits[ip]) >= RATE_LIMIT:
        return True
    hits[ip].append(now)
    return False


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/chat")
def chat():
    if rate_limited(request.remote_addr):
        return jsonify(error="Too many messages, please try again later."), 429
    data = request.get_json(silent=True) or {}
    messages = [
        {"role": m["role"], "content": str(m["content"])[:MAX_INPUT_CHARS]}
        for m in data.get("messages", [])[-10:]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    if not messages or messages[-1]["role"] != "user":
        return jsonify(error="No question received."), 400

    if OLLAMA_MODEL:
        try:
            return jsonify(reply=ollama_reply(messages))
        except Exception:
            pass  # Ollama not running: fall back to retrieval
    return jsonify(reply=retrieval_reply(messages[-1]["content"]))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
