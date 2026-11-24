# Book Language Trainer

A local **Python + Tkinter + Ollama** application that turns any PDF book into a structured vocabulary index, an interactive chapter-based practice studio, and a guided reading view—so you can learn a chapter's vocabulary and actually read the book in the target language.

All AI translations, grammar explanations, fill-in-the-blank exercises, and translation grading run **locally via Ollama** with response caching and atomic JSON progress saving.

---

## Features

- **PDF-to-TXT & JSON Indexer:**
  - Extracts page text using `PyMuPDF` (`fitz`), repairs hyphenated line breaks, and cleans whitespace.
  - Exports a clean `.txt` version of the book.
  - Detects suggested chapter start pages (from PDF bookmarks/outline or headings) and lets you **add, edit, or remove chapters** before indexing.
  - Normalizes words into **lemmas** (base forms) while tracking surface forms, total book frequency, per-chapter frequency, page numbers, and sample sentences.
- **Per-Word Accuracy & Chapter Readiness:**
  - Tracks `attempts`, `correct`, `accuracy`, `streak`, `last_reviewed`, and `known_status` (`new`, `learning`, `mastered`, `skipped`).
  - Displays a live **Chapter Readiness** progress bar showing how close you are to your target readiness threshold.
- **Interactive Practice Modes (Scoped by Chapter):**
  - **Cloze:** Fill-in-the-blank sentences generated with Ollama (with fallback to book sentences).
  - **Meaning Match:** Multiple-choice context matching using words from the active chapter.
  - **Translate Sentence:** Translate sentences from the book into English and receive feedback from Ollama.
  - **On-Demand AI Tutor:** Translate words in context, explain sentence structure/grammar, or mark words you already know.
- **Chapter Reader View:**
  - Read a chapter's text with words color-coded by mastery status (`mastered`/`skipped` in green, `learning` in amber, `new` in dark slate).
  - Click any word to inspect its book frequency, pages, and accuracy, translate it, mark it known, or ask Ollama to explain a selected passage.

---

## Project Structure

```text
DuoReado/
├── app.py                  # Entry point that launches Tkinter UI
├── config.py               # Paths, Ollama settings, spaCy models, and thresholds
├── README.md               # Project documentation
├── data/                   # Auto-created at runtime (books JSON, clean TXT, AI cache)
│   ├── books/
│   ├── texts/
│   └── cache/
├── models/
│   ├── __init__.py
│   └── schemas.py          # BookIndex, ChapterInfo, WordEntry, and Occurrence models
├── ingestion/
│   ├── __init__.py
│   ├── pdf_extractor.py    # PDF extraction, de-hyphenation, and chapter detection
│   ├── text_normalizer.py  # Sentence splitting and lemmatization (spaCy + regex fallback)
│   └── json_builder.py     # Builds clean .txt and saves/loads book .json atomically
├── ai/
│   ├── __init__.py
│   └── ollama_client.py    # Local Ollama API client with disk caching
├── learning/
│   ├── __init__.py
│   ├── mastery.py          # Mastery status & chapter readiness calculations
│   └── scheduler.py        # Practice queue prioritization & multiple-choice options
└── ui/
    ├── __init__.py
    ├── main_window.py      # Top book/chapter/model bar, readiness bar, and tabs
    ├── ingest_view.py      # Tab 1: PDF import & chapter boundary editor
    ├── practice_view.py    # Tab 2: Chapter practice modes & AI tutor
    └── reader_view.py      # Tab 3: Interactive chapter reader
```

---

## Installation & Setup

make sure you have already installed tkinter in your machine!
Install [tkinter](https://pytutorial.com/how-to-install-tkinter-in-python/) 

### 1. Create a Virtual Environment & Install Dependencies

```bash
mkdir book_language_trainer
cd book_language_trainer

python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. (Optional) Install `spaCy` for Better Lemmatization

Without `spaCy`, the app uses a clean regex/lowercase tokenizer. Installing `spaCy` and a language model groups inflected forms (such as plurals or conjugated verbs) under their base lemma:

```bash
pip install spacy

# Install the model for your target language, for example:
python -m spacy download en_core_web_sm
python -m spacy download it_core_news_sm
python -m spacy download es_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download de_core_news_sm
```

### 3. Install & Start Ollama

1. Install [Ollama](https://ollama.com/) and ensure the local server is running (`ollama serve` if not already started automatically).
2. Pull a local model (default is `granite4.2:3b`):

```bash
ollama pull granite4.2:3b 
```

---

## Running the App

```bash
python app.py
```

---

## How to Use It

### Step 1: Import PDF & Build Index (`1. Import PDF & Index`)
1. Click **Browse PDF...** and select your book.
2. Review or edit **Title**, **Author**, and the 2-letter **Language code** (e.g., `en`, `it`, `es`, `fr`, `de`).
3. Inspect the detected **Chapters & Start Pages**. Select a row to edit its title or start page, add missing chapters, or remove false positives.
4. Click **Build TXT & JSON Index**. The app writes:
   - A cleaned text file in `data/texts/<book_id>.txt`
   - A structured index file in `data/books/<book_id>.json`

### Step 2: Practice a Chapter (`2. Practice Words & Sentences`)
1. Choose your **Book**, **Chapter**, and **Ollama Model** in the top bar.
2. Pick a **Game Mode**:
   - **Cloze:** Type the missing word into the sentence.
   - **Meaning Match:** Select one of the 4 choices matching the sentence context.
   - **Translate Sentence:** Type an English translation of the sentence and let Ollama review it.
3. Use **Translate Word**, **Explain Sentence (AI)**, or **I Already Know This** whenever you need help or want to skip words you already know.
4. Press **Enter** to submit your answer, and click **Next →** to move through the queue.

### Step 3: Read the Chapter (`3. Read Chapter`)
1. Watch the **Chapter Readiness** bar in the top bar as your mastered/known word count grows.
2. Switch to **3. Read Chapter** to read the chapter text.
3. Click any word to inspect its book frequency, page numbers, and accuracy, or highlight/click near a passage and click **Explain Selected Sentence** for a translation and grammar breakdown.

---

## Where Data Is Stored

- `data/texts/*.txt`: Cleaned text extracted page-by-page from your PDFs.
- `data/books/*.json`: Book metadata, chapter ranges and text, word entries (lemmas, surface forms, counts, pages, sample sentences), and your learning stats (`known_status`, `attempts`, `correct`, `accuracy`, `streak`, `last_reviewed`). Saved atomically via a temporary file + `os.replace`.
- `data/cache/*.json`: Cached responses from Ollama so repeated word translations, sentence explanations, and cloze prompts load faster.

---

## Configuration (`config.py`)

You can customize defaults in `config.py`:

- `OLLAMA_BASE_URL` & `DEFAULT_MODEL`: Local Ollama endpoint and default model name.
- `OLLAMA_TIMEOUT_SECONDS`: Request timeout for slower local models or long explanations.
- `MASTERED_MIN_ATTEMPTS` & `MASTERED_MIN_ACCURACY`: Criteria for marking a word as `mastered` (defaults to at least `3` attempts and `80%` accuracy).
- `CHAPTER_READINESS_TARGET`: Target ratio of known/mastered words in a chapter (defaults to `0.80` / `80%`).