"""
phone_bridge.py
PC-side HTTP/JSON bridge running all book indexing, learning logic, Ollama calls,
and atomic JSON saves on the PC while serving a game-like touch UI to the phone
over USB tethering / ADB reverse.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ai.ollama_client import OllamaClient
from config import (
    DECISION_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    DEFAULT_SENTENCES_PER_WORD,
    ensure_directories,
)
from ingestion.json_builder import list_saved_books, load_book_index, save_book_index
from learning.mastery import chapter_readiness, refresh_word_mastery
from learning.scheduler import PracticeItem, build_chapter_practice_queue
from models.schemas import BookIndex, ChapterInfo, WordEntry

BASE_DIR = Path(__file__).resolve().parent
UI_HTML_PATH = BASE_DIR / "phone_ui.html"


def _norm_label(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


class BridgeState:
    """Thread-safe PC state holding active book/chapter, batch enrichment status, stages, and trace."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.lock = threading.Lock()
        self.ollama = OllamaClient(model=model_name)
        self.decision_model: str = DECISION_MODEL

        self.saved_books: list[dict[str, str]] = []
        self.book: BookIndex | None = None
        self.chapter: ChapterInfo | None = None

        self.batch_size: int = DEFAULT_BATCH_SIZE
        self.sentences_per_word: int = DEFAULT_SENTENCES_PER_WORD
        self.batch_items: list[PracticeItem] = []

        # Live enrichment progress state (polled by phone during batch start)
        self.enrichment: dict[str, Any] = {
            "active": False,
            "current": 0,
            "total": 0,
            "lemma": "",
            "message": "Idle",
        }

        # View & Game Stage state
        # screen: "setup" | "arena"
        # stage: 1 = Meaning Match, 2 = Sentence Completion, 3 = Sentence Translation
        self.screen: str = "setup"
        self.stage: int = 1

        # Stage 1 state (Meaning Match)
        self.s1_index: int = 0
        self.s1_target_translations: list[str] = []
        self.s1_found_norm: list[str] = []
        self.s1_tiles: list[str] = []
        self.s1_distractor_pool: list[str] = []
        self.s1_word_completed: bool = False
        self.s1_batch_complete: bool = False
        # Per-word batch scoreboard for Stage 1: lemma -> {"lemma": ..., "hits": ..., "misses": ...}
        self.s1_scoreboard: dict[str, dict[str, Any]] = {}

        # Stage 2 state (Sentence Completion)
        self.s2_questions: list[dict[str, str]] = []
        self.s2_index: int = 0
        self.s2_choices: list[str] = []

        # Stage 3 state (Sentence Translation)
        self.s3_items: list[dict[str, str]] = []
        self.s3_index: int = 0
        self.s3_last_verdict: dict[str, Any] | None = None
        self.s3_explanation: str = ""

        self.feedback: str = "Configure session parameters and tap 'Launch Session' to begin."
        self.trace_log: list[str] = []

        self.reload_books_locked(select_latest=True)

    def trace(self, message: str) -> None:
        """Append a timestamped under-the-hood trace event (thread-safe)."""
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        with self.lock:
            self.trace_log.append(line)
            if len(self.trace_log) > 300:
                self.trace_log = self.trace_log[-300:]
        print(line, flush=True)

    def reload_books_locked(
        self, select_latest: bool = False, target_book_id: str | None = None
    ) -> None:
        self.saved_books = list_saved_books()
        if not self.saved_books:
            self.book = None
            self.chapter = None
            return

        chosen_idx = 0
        if target_book_id:
            for i, b in enumerate(self.saved_books):
                if b["book_id"] == target_book_id:
                    chosen_idx = i
                    break
        elif select_latest:
            chosen_idx = len(self.saved_books) - 1

        self._load_book_by_index_locked(chosen_idx)

    def _load_book_by_index_locked(self, idx: int) -> None:
        info = self.saved_books[idx]
        self.book = load_book_index(info["json_path"])
        self.chapter = self.book.chapters[0] if self.book.chapters else None
        self.batch_items = []
        self.stage = 1
        self.s1_index = 0
        self.s1_batch_complete = False
        self.s1_scoreboard = {}
        self.s2_questions = []
        self.s2_index = 0
        self.s3_items = []
        self.s3_index = 0

    def configure_session(
        self,
        book_id: str,
        chapter_id: str | None,
        model_name: str,
        decision_model: str,
        batch_size: int,
        sentences_per_word: int,
    ) -> None:
        with self.lock:
            if book_id:
                for i, b in enumerate(self.saved_books):
                    if b["book_id"] == book_id:
                        self._load_book_by_index_locked(i)
                        break
            if self.book and chapter_id:
                ch = self.book.get_chapter(chapter_id)
                if ch:
                    self.chapter = ch
            if (model_name or "").strip():
                self.ollama.model = model_name.strip()
            if (decision_model or "").strip():
                self.decision_model = decision_model.strip()
            self.batch_size = max(1, min(50, int(batch_size)))
            self.sentences_per_word = max(1, min(6, int(sentences_per_word)))

        self.trace(
            f"Session configured: book='{self.book.title if self.book else None}', "
            f"chapter='{self.chapter.title if self.chapter else None}', "
            f"main_model='{self.ollama.model}', decision_model='{self.decision_model}', "
            f"N={self.batch_size}, K={self.sentences_per_word}"
        )

    def set_screen(self, screen_name: str) -> None:
        if screen_name in ("setup", "arena"):
            with self.lock:
                self.screen = screen_name
            self.trace(f"Switched UI screen to '{screen_name}'.")

    def get_active_word(self) -> WordEntry | None:
        with self.lock:
            if not self.book or not self.batch_items:
                return None
            lemma: str | None = None
            if self.stage == 1 and 0 <= self.s1_index < len(self.batch_items):
                lemma = self.batch_items[self.s1_index].word.lemma
            elif self.stage == 2 and 0 <= self.s2_index < len(self.s2_questions):
                lemma = self.s2_questions[self.s2_index]["answer_lemma"]
            elif self.stage == 3 and 0 <= self.s3_index < len(self.s3_items):
                lemma = self.s3_items[self.s3_index]["lemma"]
            else:
                lemma = self.batch_items[0].word.lemma
            return self.book.words.get(lemma)

    def _collect_other_word_translations_locked(self, target_lemma: str) -> list[str]:
        """Collect deduplicated translation labels belonging strictly to other words."""
        assert self.book is not None
        target_word = self.book.words.get(target_lemma)
        target_norm_set = {
            _norm_label(t) for t in (target_word.translations if target_word else []) if t.strip()
        }
        if target_word and target_word.cached_translation:
            target_norm_set.add(_norm_label(target_word.cached_translation))

        seen_norm: set[str] = set()
        distractors: list[str] = []

        # Prefer distractors from batch first, then rest of book
        ordered_words: list[WordEntry] = []
        for it in self.batch_items:
            if it.word.lemma != target_lemma and it.word.lemma in self.book.words:
                ordered_words.append(self.book.words[it.word.lemma])
        for lem, w_obj in self.book.words.items():
            if lem != target_lemma and w_obj not in ordered_words:
                ordered_words.append(w_obj)

        for w_obj in ordered_words:
            candidates = list(w_obj.translations or [])
            if not candidates and w_obj.cached_translation:
                candidates = [w_obj.cached_translation]
            for tr in candidates:
                clean = " ".join(tr.strip().split())
                norm = _norm_label(clean)
                if not clean or norm in target_norm_set or norm in seen_norm:
                    continue
                seen_norm.add(norm)
                distractors.append(clean)

        random.shuffle(distractors)
        return distractors

    def _prepare_s1_word_locked(self) -> None:
        """Prepare target translations, deduplicated tiles, and distractor pool for current Stage 1 word."""
        assert self.book is not None
        if self.s1_index >= len(self.batch_items):
            self.s1_batch_complete = True
            self.s1_tiles = []
            return

        w = self.book.words.get(
            self.batch_items[self.s1_index].word.lemma,
            self.batch_items[self.s1_index].word,
        )
        self.s1_scoreboard.setdefault(
            w.lemma, {"lemma": w.lemma, "hits": 0, "misses": 0}
        )

        # Deduplicate target translations from JSON
        dedup_targets: list[str] = []
        seen_t: set[str] = set()
        for t in (w.translations or []):
            clean = " ".join(t.strip().split())
            norm = _norm_label(clean)
            if clean and norm not in seen_t:
                seen_t.add(norm)
                dedup_targets.append(clean)

        if not dedup_targets:
            fallback = (w.cached_translation or f"meaning of {w.lemma}").strip()
            dedup_targets = [fallback]

        self.s1_target_translations = dedup_targets
        self.s1_found_norm = []
        self.s1_word_completed = False

        pool = self._collect_other_word_translations_locked(w.lemma)
        # Build up to 6 tiles: include all target translations + distractors from other words
        initial_tiles: list[str] = list(dedup_targets)
        used_norm = {_norm_label(x) for x in initial_tiles}

        while len(initial_tiles) < 6 and pool:
            cand = pool.pop(0)
            norm = _norm_label(cand)
            if norm not in used_norm:
                used_norm.add(norm)
                initial_tiles.append(cand)

        random.shuffle(initial_tiles)
        self.s1_tiles = initial_tiles
        self.s1_distractor_pool = pool

    def _next_replacement_tile_locked(self, exclude_norm: set[str]) -> str | None:
        """Pick a fresh non-duplicate tile from distractor pool (or remaining unfound target)."""
        while self.s1_distractor_pool:
            cand = self.s1_distractor_pool.pop(0)
            norm = _norm_label(cand)
            if norm not in exclude_norm:
                return cand
        for t in self.s1_target_translations:
            norm = _norm_label(t)
            if norm not in exclude_norm and norm not in self.s1_found_norm:
                return t
        return None

    def _build_s2_locked(self, k: int) -> None:
        assert self.book is not None
        questions: list[dict[str, str]] = []
        for item in self.batch_items:
            w = self.book.words.get(item.word.lemma, item.word)
            chosen_sents = w.enriched_sentences[:k]
            if not chosen_sents:
                chosen_sents = [{"sentence": item.sample_sentence, "related_word": w.lemma}]
            for s_obj in chosen_sents:
                sent = s_obj.get("sentence", item.sample_sentence)
                target_form = s_obj.get("related_word") or w.lemma
                blanked = sent
                for cand in (target_form, w.lemma):
                    if cand and cand.lower() in sent.lower():
                        idx = sent.lower().find(cand.lower())
                        blanked = sent[:idx] + "_____" + sent[idx + len(cand) :]
                        break
                questions.append(
                    {
                        "sentence": blanked,
                        "answer_lemma": w.lemma,
                        "translation_hint": s_obj.get("translation", ""),
                    }
                )
        random.shuffle(questions)
        self.s2_questions = questions
        self.s2_index = 0
        self._prepare_s2_choices_locked()

    def _prepare_s2_choices_locked(self) -> None:
        if 0 <= self.s2_index < len(self.s2_questions):
            choices = [it.word.lemma for it in self.batch_items]
            random.shuffle(choices)
            self.s2_choices = choices
        else:
            self.s2_choices = []

    def _build_s3_locked(self, k: int) -> None:
        assert self.book is not None
        items: list[dict[str, str]] = []
        for item in self.batch_items:
            w = self.book.words.get(item.word.lemma, item.word)
            chosen_sents = w.enriched_sentences[:k]
            if not chosen_sents:
                chosen_sents = [
                    {"sentence": item.sample_sentence, "translation": "", "related_word": w.lemma}
                ]
            for s_obj in chosen_sents:
                items.append(
                    {
                        "sentence": s_obj.get("sentence", item.sample_sentence),
                        "lemma": w.lemma,
                        "ref_translation": s_obj.get("translation", ""),
                    }
                )
        random.shuffle(items)
        self.s3_items = items
        self.s3_index = 0
        self.s3_last_verdict = None
        self.s3_explanation = ""

    def start_batch(self, batch_size: int, sentences_per_word: int) -> None:
        with self.lock:
            if not self.book or not self.chapter:
                raise ValueError("No book or chapter selected on PC.")
            book = self.book
            chapter = self.chapter
            n = max(1, min(50, int(batch_size)))
            k = max(1, min(6, int(sentences_per_word)))
            self.batch_size = n
            self.sentences_per_word = k
            self.enrichment = {
                "active": True,
                "current": 0,
                "total": n,
                "lemma": "",
                "message": f"Sorting '{chapter.title}' words by reverse knowledge & chapter count...",
            }

        self.trace(
            f"Sorting '{chapter.title}' by reverse knowledge then chapter count (N={n}, K={k})..."
        )
        queue = build_chapter_practice_queue(book, chapter, batch_size=n)

        with self.lock:
            self.enrichment["total"] = len(queue)
            self.enrichment["message"] = f"Enriching {len(queue)} words (K={k} sentences/word)..."

        for idx, item in enumerate(queue, start=1):
            w = item.word
            with self.lock:
                self.enrichment["current"] = idx
                self.enrichment["lemma"] = w.lemma
                self.enrichment["message"] = f"Processing '{w.lemma}' ({idx}/{len(queue)})..."

            existing_sents = len(w.enriched_sentences)
            needs_meta = not (w.translations and w.false_friends and w.edge_meanings)
            needs_sents = existing_sents < k

            if needs_meta or needs_sents:
                self.trace(
                    f"Ollama enriching '{w.lemma}' [{idx}/{len(queue)}] (sents={existing_sents}/{k})..."
                )
                res = self.ollama.enrich_word_entry(
                    lemma=w.lemma,
                    source_lang=book.language,
                    num_sentences=k,
                    existing_count=existing_sents,
                    context_sentence=item.sample_sentence,
                )
                with self.lock:
                    if res["translations"] and not w.translations:
                        w.translations = res["translations"]
                    if res["false_friends"] and not w.false_friends:
                        w.false_friends = res["false_friends"]
                    if res["edge_meanings"] and not w.edge_meanings:
                        w.edge_meanings = res["edge_meanings"]
                    if res["sentences"]:
                        w.enriched_sentences.extend(res["sentences"])

            with self.lock:
                if len(w.enriched_sentences) < k:
                    for occ in w.occurrences:
                        if occ.sentence and all(
                            s.get("sentence") != occ.sentence for s in w.enriched_sentences
                        ):
                            w.enriched_sentences.append(
                                {
                                    "sentence": occ.sentence,
                                    "translation": "",
                                    "related_word": w.lemma,
                                    "source": "book",
                                }
                            )
                            if len(w.enriched_sentences) >= k:
                                break

        save_book_index(book)
        self.trace(f"Batch of {len(queue)} words enriched & saved to PC JSON.")

        with self.lock:
            self.batch_items = queue
            self.stage = 1
            self.s1_index = 0
            self.s1_batch_complete = len(queue) == 0
            self.s1_scoreboard = {
                it.word.lemma: {"lemma": it.word.lemma, "hits": 0, "misses": 0}
                for it in queue
            }
            self._prepare_s1_word_locked()
            self._build_s2_locked(k)
            self._build_s3_locked(k)
            self.screen = "arena"
            self.enrichment = {
                "active": False,
                "current": len(queue),
                "total": len(queue),
                "lemma": "",
                "message": "Batch ready!",
            }
            self.feedback = "Session launched! Exercise 1: Tap all correct translations for the word above."

    def tap_meaning_tile(self, tile_index: int) -> dict[str, Any]:
        """
        Handle tapping a translation tile in Exercise 1:
        - Checks if tile matches a remaining unfound translation from JSON.
        - Records attempt & updates word knowledge on PC.
        - Replaces the tapped tile with a fresh tile.
        - Advances to next word (or batch summary) once all correct translations are found.
        """
        with self.lock:
            if not self.book or self.stage != 1 or self.s1_batch_complete:
                return {"outcome": "ignored"}
            if self.s1_index >= len(self.batch_items):
                self.s1_batch_complete = True
                return {"outcome": "batch_complete"}
            if tile_index < 0 or tile_index >= len(self.s1_tiles):
                return {"outcome": "ignored"}

            w = self.book.words.get(
                self.batch_items[self.s1_index].word.lemma,
                self.batch_items[self.s1_index].word,
            )
            tapped_label = self.s1_tiles[tile_index]
            tapped_norm = _norm_label(tapped_label)

            target_norm_set = {_norm_label(t) for t in self.s1_target_translations}
            already_found = tapped_norm in self.s1_found_norm
            is_correct = (tapped_norm in target_norm_set) and not already_found

            w.record_attempt(is_correct)
            refresh_word_mastery(w)
            save_book_index(self.book)

            sb = self.s1_scoreboard.setdefault(
                w.lemma, {"lemma": w.lemma, "hits": 0, "misses": 0}
            )
            if is_correct:
                sb["hits"] += 1
                self.s1_found_norm.append(tapped_norm)
                outcome = "correct"
                self.feedback = (
                    f"✔ '{tapped_label}' is a match for '{w.lemma}'! "
                    f"({len(self.s1_found_norm)}/{len(self.s1_target_translations)} found)"
                )
            else:
                sb["misses"] += 1
                outcome = "wrong"
                self.feedback = (
                    f"✖ '{tapped_label}' is not a remaining match for '{w.lemma}'."
                )

            # Swap out tapped tile for a new tile
            other_norms = {
                _norm_label(lbl) for i, lbl in enumerate(self.s1_tiles) if i != tile_index
            }
            replacement = self._next_replacement_tile_locked(other_norms)
            if replacement is not None:
                self.s1_tiles[tile_index] = replacement

            word_finished = len(self.s1_found_norm) >= len(self.s1_target_translations)
            batch_finished = False
            if word_finished:
                self.s1_word_completed = True
                if self.s1_index + 1 < len(self.batch_items):
                    self.s1_index += 1
                    self._prepare_s1_word_locked()
                    self.feedback += f" → Moving to '{self.batch_items[self.s1_index].word.lemma}'!"
                else:
                    self.s1_batch_complete = True
                    batch_finished = True
                    self.feedback = "🎉 All words in this batch completed! Review your scores below."

        self.trace(
            f"Ex.1 tap on '{w.lemma}': tile='{tapped_label}' -> {outcome} "
            f"(found {len(self.s1_found_norm)}/{len(self.s1_target_translations)}, batch_done={batch_finished})"
        )
        return {
            "outcome": outcome,
            "tile_index": tile_index,
            "word_finished": word_finished,
            "batch_finished": batch_finished,
        }

    def set_stage(self, stage_num: int) -> None:
        with self.lock:
            if not self.batch_items:
                return
            self.stage = max(1, min(3, int(stage_num)))
            if self.stage == 1:
                self.s1_batch_complete = False
                self.s1_index = 0
                self._prepare_s1_word_locked()
            elif self.stage == 2:
                self._prepare_s2_choices_locked()
        self.trace(f"Switched to Stage {self.stage}.")

    def check_completion(self, chosen_lemma: str) -> dict[str, Any]:
        with self.lock:
            if not self.book or self.stage != 2 or self.s2_index >= len(self.s2_questions):
                return {"correct": False, "done": True}
            q = self.s2_questions[self.s2_index]
            expected = q["answer_lemma"]
            is_correct = chosen_lemma.strip() == expected
            w = self.book.words.get(expected)
            if w:
                w.record_attempt(is_correct)
                refresh_word_mastery(w)
                save_book_index(self.book)

            if is_correct:
                self.feedback = f"✔ '{expected}' fits the sentence!"
            else:
                self.feedback = f"✖ '{chosen_lemma}' does not fit — expected '{expected}'."

            if self.s2_index + 1 < len(self.s2_questions):
                self.s2_index += 1
                self._prepare_s2_choices_locked()
                done = False
            else:
                self.stage = 3
                self.s3_index = 0
                self.s3_last_verdict = None
                self.s3_explanation = ""
                done = True
                self.feedback += " → Stage 2 complete! Stage 3: Sentence Translation."

        self.trace(
            f"Stage 2 completion: chose='{chosen_lemma}', expected='{expected}', correct={is_correct}"
        )
        return {"correct": is_correct, "chosen": chosen_lemma, "expected": expected, "done": done}

    def validate_translation(self, user_translation: str) -> None:
        with self.lock:
            if not self.book or self.stage != 3 or self.s3_index >= len(self.s3_items):
                return
            item = self.s3_items[self.s3_index]
            src_sent = item["sentence"]
            lang = self.book.language
            main_model = self.ollama.model
            dec_model = self.decision_model

        self.trace(f"Stage 3: decision model '{dec_model}' evaluating translation...")
        verdict = self.ollama.judge_translation_with_decision_model(
            source_sentence=src_sent,
            user_translation=user_translation,
            source_lang=lang,
            decision_model=dec_model,
        )
        is_valid = bool(verdict.get("valid"))
        reason = str(verdict.get("reason", ""))
        self.trace(f"Decision model ({dec_model}) -> valid={is_valid} ({reason})")

        explanation = ""
        if is_valid:
            fb = f"✔ Valid translation ({dec_model}): {reason}"
        else:
            self.trace(f"Translation invalid -> calling main LLM '{main_model}' for explanation...")
            explanation = self.ollama.explain_sentence(
                sentence=f"{src_sent}\n[Learner attempted: {user_translation} | Decision note: {reason}]",
                source_lang=lang,
            )
            fb = f"✖ Needs work ({dec_model}): {reason}"

        with self.lock:
            self.s3_last_verdict = {"valid": is_valid, "reason": reason, "model": dec_model}
            self.s3_explanation = explanation
            self.feedback = fb

    def next_s3(self) -> None:
        with self.lock:
            if self.s3_index + 1 < len(self.s3_items):
                self.s3_index += 1
                self.s3_last_verdict = None
                self.s3_explanation = ""
                self.feedback = ""
            else:
                self.feedback = "🎉 Stage 3 complete! Ready for the next batch or chapter."
        self.trace(f"Advanced Stage 3 (s3_index={self.s3_index}).")

    def save_red_flag_word(self, updated_dict: dict[str, Any]) -> WordEntry:
        with self.lock:
            if not self.book:
                raise ValueError("No book loaded on PC.")
            lemma = str(updated_dict.get("lemma") or "").strip()
            if not lemma:
                raise ValueError("Word JSON must include a non-empty 'lemma'.")
            entry = WordEntry.from_dict(updated_dict)
            self.book.words[lemma] = entry
            save_book_index(self.book)
            self.feedback = f"🚩 Saved manual JSON edits for '{lemma}'."
        self.trace(f"🚩 Saved manual JSON override for '{lemma}'.")
        return entry

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            readiness: dict[str, Any] = {
                "readiness_ratio": 0.0,
                "ready_words": 0,
                "total_words": 0,
                "words_needed_for_target": 0,
            }
            if self.book and self.chapter:
                readiness = chapter_readiness(self.book, self.chapter)

            active_word_dict: dict[str, Any] | None = None
            ch_appearances = 0
            if self.book and self.batch_items:
                lemma: str | None = None
                if self.stage == 1 and 0 <= self.s1_index < len(self.batch_items):
                    lemma = self.batch_items[self.s1_index].word.lemma
                elif self.stage == 2 and 0 <= self.s2_index < len(self.s2_questions):
                    lemma = self.s2_questions[self.s2_index]["answer_lemma"]
                elif self.stage == 3 and 0 <= self.s3_index < len(self.s3_items):
                    lemma = self.s3_items[self.s3_index]["lemma"]
                else:
                    lemma = self.batch_items[0].word.lemma

                w = self.book.words.get(lemma)
                if w:
                    active_word_dict = w.to_dict()
                    if self.chapter:
                        ch_appearances = w.by_chapter.get(self.chapter.chapter_id, 0)

            s2_current = (
                self.s2_questions[self.s2_index]
                if 0 <= self.s2_index < len(self.s2_questions)
                else None
            )
            s3_current = (
                self.s3_items[self.s3_index]
                if 0 <= self.s3_index < len(self.s3_items)
                else None
            )

            return {
                "screen": self.screen,
                "saved_books": self.saved_books,
                "book": (
                    {
                        "book_id": self.book.book_id,
                        "title": self.book.title,
                        "author": self.book.author,
                        "language": self.book.language,
                        "chapters": [
                            {
                                "chapter_id": c.chapter_id,
                                "title": c.title,
                                "start_page": c.start_page,
                                "end_page": c.end_page,
                            }
                            for c in self.book.chapters
                        ],
                    }
                    if self.book
                    else None
                ),
                "chapter_id": self.chapter.chapter_id if self.chapter else None,
                "model": self.ollama.model,
                "decision_model": self.decision_model,
                "batch_size": self.batch_size,
                "sentences_per_word": self.sentences_per_word,
                "batch_count": len(self.batch_items),
                "enrichment": dict(self.enrichment),
                "stage": self.stage,
                "s1_index": self.s1_index,
                "s1_tiles": list(self.s1_tiles),
                "s1_found_count": len(self.s1_found_norm),
                "s1_target_count": len(self.s1_target_translations),
                "s1_batch_complete": self.s1_batch_complete,
                "s1_scoreboard": list(self.s1_scoreboard.values()),
                "s2_index": self.s2_index,
                "s2_total": len(self.s2_questions),
                "s2_question": s2_current,
                "s2_choices": list(self.s2_choices),
                "s3_index": self.s3_index,
                "s3_total": len(self.s3_items),
                "s3_item": s3_current,
                "s3_last_verdict": self.s3_last_verdict,
                "s3_explanation": self.s3_explanation,
                "active_word": active_word_dict,
                "chapter_appearances": ch_appearances,
                "readiness": readiness,
                "feedback": self.feedback,
                "trace": list(self.trace_log),
            }


def make_handler(state: BridgeState):
    class BridgeHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            body = self.rfile.read(length).decode("utf-8")
            return json.loads(body) if body.strip() else {}

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                if not UI_HTML_PATH.exists():
                    self.send_error(404, "phone_ui.html not found")
                    return
                content = UI_HTML_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

            if parsed.path == "/api/state":
                self._send_json(state.snapshot())
                return

            self.send_error(404, "Not found")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                data = self._read_json_body()

                if parsed.path == "/api/configure":
                    state.configure_session(
                        book_id=str(data.get("book_id", "")),
                        chapter_id=data.get("chapter_id"),
                        model_name=str(data.get("model", state.ollama.model)),
                        decision_model=str(data.get("decision_model", state.decision_model)),
                        batch_size=int(data.get("batch_size", state.batch_size)),
                        sentences_per_word=int(
                            data.get("sentences_per_word", state.sentences_per_word)
                        ),
                    )
                    self._send_json(state.snapshot())

                elif parsed.path == "/api/screen":
                    state.set_screen(str(data.get("screen", "setup")))
                    self._send_json(state.snapshot())

                elif parsed.path == "/api/start_batch":
                    state.configure_session(
                        book_id=str(data.get("book_id", state.book.book_id if state.book else "")),
                        chapter_id=data.get("chapter_id", state.chapter.chapter_id if state.chapter else None),
                        model_name=str(data.get("model", state.ollama.model)),
                        decision_model=str(data.get("decision_model", state.decision_model)),
                        batch_size=int(data.get("batch_size", state.batch_size)),
                        sentences_per_word=int(
                            data.get("sentences_per_word", state.sentences_per_word)
                        ),
                    )
                    # Run batch enrichment in background so phone polling gets live progress
                    def _bg() -> None:
                        try:
                            state.start_batch(
                                batch_size=int(data.get("batch_size", state.batch_size)),
                                sentences_per_word=int(
                                    data.get("sentences_per_word", state.sentences_per_word)
                                ),
                            )
                        except Exception as exc:
                            with state.lock:
                                state.enrichment["active"] = False
                                state.enrichment["message"] = f"Error: {exc}"
                            state.trace(f"Batch enrichment error: {exc}")

                    threading.Thread(target=_bg, daemon=True).start()
                    self._send_json({"started": True, **state.snapshot()})

                elif parsed.path == "/api/stage":
                    state.set_stage(int(data.get("stage", 1)))
                    self._send_json(state.snapshot())

                elif parsed.path == "/api/tap_meaning":
                    tap_meta = state.tap_meaning_tile(int(data.get("tile_index", -1)))
                    self._send_json({"tap": tap_meta, **state.snapshot()})

                elif parsed.path == "/api/check_completion":
                    res = state.check_completion(str(data.get("chosen_lemma", "")))
                    self._send_json({"completion": res, **state.snapshot()})

                elif parsed.path == "/api/validate_translation":
                    state.validate_translation(str(data.get("translation", "")))
                    self._send_json(state.snapshot())

                elif parsed.path == "/api/next_s3":
                    state.next_s3()
                    self._send_json(state.snapshot())

                elif parsed.path == "/api/red_flag":
                    word_dict = data.get("word")
                    if not isinstance(word_dict, dict):
                        raise ValueError("Expected 'word' JSON object.")
                    state.save_red_flag_word(word_dict)
                    self._send_json(state.snapshot())

                else:
                    self.send_error(404, "Unknown endpoint")
            except Exception as exc:
                state.trace(f"ERROR on {parsed.path}: {exc}")
                self._send_json({"error": str(exc)}, status=400)

    return BridgeHandler


def main() -> None:
    ensure_directories()
    parser = argparse.ArgumentParser(description="PC-to-Phone Game Bridge")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Default main Ollama model")
    args = parser.parse_args()

    state = BridgeState(model_name=args.model)
    state.trace(
        f"Bridge listening on http://{args.host}:{args.port} "
        f"(main='{args.model}', decision='{DECISION_MODEL}')"
    )

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping bridge...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()