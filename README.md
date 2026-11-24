README.md (new)
markdown

# DuoReado — Book Language Trainer

Turn any PDF book into a personal vocabulary trainer and a *readable* chapter
viewer, powered by a local Ollama LLM. Runs entirely offline.

- **PC app** (Tkinter) for importing PDFs, browsing words, practising, reading.
- **Phone app** (single-page HTML served by a small Python HTTP bridge) for
  practising on the sofa.
- **Shared state**: `BookIndex` (from the PDF) is regenerable; `UserProgress`
  (per user) is never lost when a book is re-ingested.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm      # add other langs as needed
ollama pull granite4.2:3b                    # or whichever model you like

# 2a. Desktop app
python app.py

# 2b. Phone bridge (open http://<your-lan-ip>:8765 on the phone)
python phone_bridge.py --host 0.0.0.0 --port 8765 --user alice
```
