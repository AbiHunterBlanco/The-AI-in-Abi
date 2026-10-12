# Portfolio chat (no API key)

    pip install -r requirements.txt
    python app.py        # http://localhost:5000

Edit `qa.md` with your answers - changes apply instantly.

## Optional: local AI model (still no key)
    ollama pull llama3.2:1b
    OLLAMA_MODEL=llama3.2:1b python app.py
Uses `qa.md` as facts and `style.md` for tone. Falls back to retrieval if Ollama isn't running.
