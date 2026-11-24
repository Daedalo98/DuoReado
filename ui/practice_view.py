"""
ui/practice_view.py
Tab 2: 3-Stage Chapter Practice Loop:
  Stage 1: Meaning Match (with Verify & JSON persistence)
  Stage 2: Sentence Completion (shuffled sentences across batch words)
  Stage 3: Sentence Translation (tev1:4b gate -> main LLM explanation on invalid)
Includes batch size N, sentences per word K, full trace logging, and Red Flag JSON editor.
"""

from __future__ import annotations

import random
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from ai.ollama_client import OllamaClient
from config import DECISION_MODEL, DEFAULT_BATCH_SIZE, DEFAULT_SENTENCES_PER_WORD
from ingestion.json_builder import save_book_index
from learning.mastery import refresh_word_mastery
from learning.scheduler import PracticeItem, build_chapter_practice_queue
from models.schemas import BookIndex, ChapterInfo, WordEntry
from ui.red_flag_dialog import RedFlagWordDialog


class PracticeView(ttk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        ollama: OllamaClient,
        on_stats_updated: Callable[[], None],
        trace: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent, padding=14)
        self.ollama = ollama
        self.on_stats_updated = on_stats_updated
        self.trace = trace or (lambda _msg: None)

        self.book: BookIndex | None = None
        self.chapter: ChapterInfo | None = None

        # Batch state
        self.batch_items: list[PracticeItem] = []
        self.stage: int = 1  # 1 = Meaning Match, 2 = Sentence Completion, 3 = Sentence Translation

        # Stage 1 state
        self.s1_index: int = 0
        self.s1_choice_vars: list[tuple[tk.BooleanVar, str]] = []

        # Stage 2 state (list of dicts: {"sentence": ..., "answer_lemma": ..., "hint": ...})
        self.s2_questions: list[dict[str, str]] = []
        self.s2_index: int = 0

        # Stage 3 state (list of dicts: {"sentence": ..., "lemma": ..., "ref_translation": ...})
        self.s3_items: list[dict[str, str]] = []
        self.s3_index: int = 0

        self.batch_size_var = tk.IntVar(value=DEFAULT_BATCH_SIZE)
        self.sentences_per_word_var = tk.IntVar(value=DEFAULT_SENTENCES_PER_WORD)
        self.stage_label_var = tk.StringVar(value="Stage: Idle")
        self.word_header_var = tk.StringVar(value="Load a book and chapter to start a batch.")
        self.word_subinfo_var = tk.StringVar(value="")
        self.prompt_var = tk.StringVar(value="")
        self.translation_input_var = tk.StringVar()

        self._build_ui()

    def _build_ui(self) -> None:
        # Top controls: N batch size, K sentences/word, Load button, Stage indicator
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", pady=(0, 8))

        ttk.Label(ctrl, text="Batch words (N):").pack(side="left")
        ttk.Spinbox(ctrl, from_=1, to=50, textvariable=self.batch_size_var, width=5).pack(
            side="left", padx=(4, 12)
        )

        ttk.Label(ctrl, text="Sentences/word (K):").pack(side="left")
        ttk.Spinbox(ctrl, from_=1, to=6, textvariable=self.sentences_per_word_var, width=5).pack(
            side="left", padx=(4, 12)
        )

        ttk.Button(ctrl, text="Load & Enrich Batch", command=self.start_batch).pack(side="left", padx=(0, 8))
        ttk.Button(ctrl, text="Stage 1: Meaning", command=lambda: self._enter_stage(1)).pack(side="left", padx=2)
        ttk.Button(ctrl, text="Stage 2: Completion", command=lambda: self._enter_stage(2)).pack(side="left", padx=2)
        ttk.Button(ctrl, text="Stage 3: Translation", command=lambda: self._enter_stage(3)).pack(side="left", padx=2)

        ttk.Label(ctrl, textvariable=self.stage_label_var, font=("TkDefaultFont", 10, "bold")).pack(side="right")

        # Top area: Active word details + Red Flag button
        top_word_box = ttk.LabelFrame(self, text="Active Word & Linguistic Notes", padding=10)
        top_word_box.pack(fill="x", pady=(0, 8))

        header_row = ttk.Frame(top_word_box)
        header_row.pack(fill="x")
        ttk.Label(header_row, textvariable=self.word_header_var, font=("TkDefaultFont", 12, "bold")).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(header_row, text="🚩 Red Flag (JSON)", command=self._open_red_flag_for_active_word).pack(
            side="right"
        )

        ttk.Label(top_word_box, textvariable=self.word_subinfo_var, wraplength=860, foreground="#334155").pack(
            anchor="w", pady=(4, 0)
        )

        # Stage interactive area
        stage_box = ttk.LabelFrame(self, text="Interactive Stage", padding=12)
        stage_box.pack(fill="x", pady=(0, 8))

        ttk.Label(
            stage_box,
            textvariable=self.prompt_var,
            font=("TkDefaultFont", 11),
            wraplength=860,
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        # Container A: Stage 1 checkboxes + Verify
        self.s1_frame = ttk.Frame(stage_box)
        self.s1_checks_holder = ttk.Frame(self.s1_frame)
        self.s1_checks_holder.pack(fill="x")
        s1_btns = ttk.Frame(self.s1_frame)
        s1_btns.pack(fill="x", pady=(8, 0))
        ttk.Button(s1_btns, text="Verify & Save Knowledge", command=self._verify_meaning_match).pack(side="left")
        ttk.Button(s1_btns, text="Next Word →", command=self._next_s1).pack(side="right")

        # Container B: Stage 2 word buttons
        self.s2_frame = ttk.Frame(stage_box)
        self.s2_buttons_holder = ttk.Frame(self.s2_frame)
        self.s2_buttons_holder.pack(fill="x")

        # Container C: Stage 3 translation entry + Submit
        self.s3_frame = ttk.Frame(stage_box)
        s3_row = ttk.Frame(self.s3_frame)
        s3_row.pack(fill="x")
        entry = ttk.Entry(s3_row, textvariable=self.translation_input_var, font=("TkDefaultFont", 11))
        entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        entry.bind("<Return>", lambda _e: self._submit_translation_stage())
        ttk.Button(s3_row, text="Validate Translation", command=self._submit_translation_stage).pack(side="left")
        ttk.Button(self.s3_frame, text="Next Sentence →", command=self._next_s3).pack(anchor="e", pady=(6, 0))

        # Bottom area: Feedback / Tutor output
        out_box = ttk.LabelFrame(self, text="Stage Feedback & AI Tutor", padding=10)
        out_box.pack(fill="both", expand=True)
        self.feedback_text = tk.Text(out_box, height=8, wrap="word", font=("TkDefaultFont", 10))
        self.feedback_text.pack(fill="both", expand=True)

    def set_context(self, book: BookIndex | None, chapter: ChapterInfo | None) -> None:
        self.book = book
        self.chapter = chapter
        self.trace(
            f"Context set: book='{book.title if book else None}', "
            f"chapter='{chapter.title if chapter else None}'"
        )

    def _set_feedback(self, msg: str) -> None:
        self.feedback_text.delete("1.0", "end")
        self.feedback_text.insert("1.0", msg)

    def _get_active_word_entry(self) -> WordEntry | None:
        if not self.book or not self.batch_items:
            return None
        if self.stage == 1 and 0 <= self.s1_index < len(self.batch_items):
            lemma = self.batch_items[self.s1_index].word.lemma
            return self.book.words.get(lemma)
        if self.stage == 2 and 0 <= self.s2_index < len(self.s2_questions):
            lemma = self.s2_questions[self.s2_index]["answer_lemma"]
            return self.book.words.get(lemma)
        if self.stage == 3 and 0 <= self.s3_index < len(self.s3_items):
            lemma = self.s3_items[self.s3_index]["lemma"]
            return self.book.words.get(lemma)
        return self.batch_items[0].word

    def start_batch(self) -> None:
        if not self.book or not self.chapter:
            messagebox.showwarning("Missing Selection", "Select a book and chapter first.")
            return

        n = max(1, int(self.batch_size_var.get()))
        k = max(1, int(self.sentences_per_word_var.get()))
        book = self.book
        chapter = self.chapter

        self.trace(f"Sorting chapter '{chapter.title}' by reverse knowledge then chapter count (N={n}, K={k})...")
        queue = build_chapter_practice_queue(book, chapter, batch_size=n)
        if not queue:
            self.prompt_var.set("No words available in this chapter.")
            return

        self._set_feedback(f"Enriching {len(queue)} words (target {k} sentences/word) where JSON data is incomplete...")

        def worker() -> None:
            enriched_calls = 0
            for item in queue:
                w = item.word
                existing_sents = len(w.enriched_sentences)
                needs_meta = not (w.translations and w.false_friends and w.edge_meanings)
                needs_sents = existing_sents < k

                if needs_meta or needs_sents:
                    enriched_calls += 1
                    self.trace(
                        f"Enriching '{w.lemma}' via Ollama (has_sents={existing_sents}, target={k})..."
                    )
                    res = self.ollama.enrich_word_entry(
                        lemma=w.lemma,
                        source_lang=book.language,
                        num_sentences=k,
                        existing_count=existing_sents,
                        context_sentence=item.sample_sentence,
                    )
                    if res["translations"] and not w.translations:
                        w.translations = res["translations"]
                    if res["false_friends"] and not w.false_friends:
                        w.false_friends = res["false_friends"]
                    if res["edge_meanings"] and not w.edge_meanings:
                        w.edge_meanings = res["edge_meanings"]
                    if res["sentences"]:
                        w.enriched_sentences.extend(res["sentences"])

                # Also seed from book occurrences if still below K
                if len(w.enriched_sentences) < k:
                    for occ in w.occurrences:
                        if occ.sentence and all(
                            s.get("sentence") != occ.sentence for s in w.enriched_sentences
                        ):
                            w.enriched_sentences.append(
                                {
                                    "sentence": occ.sentence,
                                    "translation": "(Book sentence — click Translate or Stage 3)",
                                    "related_word": w.lemma,
                                    "source": "book",
                                }
                            )
                            if len(w.enriched_sentences) >= k:
                                break

            save_book_index(book)
            self.trace(
                f"Batch ready: {len(queue)} words, {enriched_calls} Ollama enrichment calls, saved JSON."
            )
            self.after(0, lambda: self._on_batch_ready(queue))

        threading.Thread(target=worker, daemon=True).start()

    def _on_batch_ready(self, queue: list[PracticeItem]) -> None:
        self.batch_items = queue
        self._set_feedback("Batch loaded! Starting Stage 1: Meaning Match.")
        self._enter_stage(1)

    def _hide_all_stage_frames(self) -> None:
        self.s1_frame.pack_forget()
        self.s2_frame.pack_forget()
        self.s3_frame.pack_forget()

    def _enter_stage(self, stage_num: int) -> None:
        if not self.batch_items or not self.book or not self.chapter:
            return
        self.stage = stage_num
        self._hide_all_stage_frames()

        if stage_num == 1:
            self.stage_label_var.set("Stage 1 / 3: Meaning Match")
            self.s1_frame.pack(fill="x")
            self.s1_index = 0
            self.trace("Entered Stage 1: Meaning Match.")
            self._render_s1_word()

        elif stage_num == 2:
            k = max(1, int(self.sentences_per_word_var.get()))
            questions: list[dict[str, str]] = []
            for item in self.batch_items:
                w = self.book.words.get(item.word.lemma, item.word)
                chosen_sents = w.enriched_sentences[:k]
                if not chosen_sents:
                    chosen_sents = [{"sentence": item.sample_sentence, "related_word": w.lemma}]
                for s_obj in chosen_sents:
                    sent = s_obj.get("sentence", item.sample_sentence)
                    target_form = s_obj.get("related_word") or w.lemma
                    # Blank out target_form or lemma case-insensitively if present
                    blanked = sent
                    for candidate in (target_form, w.lemma):
                        if candidate and candidate.lower() in sent.lower():
                            idx = sent.lower().find(candidate.lower())
                            blanked = sent[:idx] + "_____" + sent[idx + len(candidate) :]
                            break
                    questions.append(
                        {
                            "sentence": blanked,
                            "answer_lemma": w.lemma,
                            "hint": f"Base lemma: {w.lemma}",
                        }
                    )
            random.shuffle(questions)
            self.s2_questions = questions
            self.s2_index = 0
            self.stage_label_var.set("Stage 2 / 3: Sentence Completion")
            self.s2_frame.pack(fill="x")
            self.trace(f"Entered Stage 2: Sentence Completion ({len(questions)} shuffled blanks).")
            self._render_s2_question()

        elif stage_num == 3:
            k = max(1, int(self.sentences_per_word_var.get()))
            items: list[dict[str, str]] = []
            for item in self.batch_items:
                w = self.book.words.get(item.word.lemma, item.word)
                chosen_sents = w.enriched_sentences[:k]
                if not chosen_sents:
                    chosen_sents = [{"sentence": item.sample_sentence, "translation": "", "related_word": w.lemma}]
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
            self.stage_label_var.set("Stage 3 / 3: Sentence Translation")
            self.s3_frame.pack(fill="x")
            self.trace(f"Entered Stage 3: Sentence Translation ({len(items)} sentences).")
            self._render_s3_item()

    def _update_word_banner(self, w: WordEntry) -> None:
        ch_id = self.chapter.chapter_id if self.chapter else ""
        ch_count = w.by_chapter.get(ch_id, 0)
        acc_pct = int(round(w.accuracy * 100))
        self.word_header_var.set(
            f"Lemma: '{w.lemma}'  |  Knowledge: {w.correct}/{w.attempts} ({acc_pct}%)  |  "
            f"In chapter: {ch_count}x  |  Book total: {w.total_count}x"
        )
        ff = ", ".join(w.false_friends) if w.false_friends else "—"
        edge = ", ".join(w.edge_meanings) if w.edge_meanings else "—"
        self.word_subinfo_var.set(f"False friends / traps: {ff}   ||   Edge meanings / uses: {edge}")

    # --- Stage 1: Meaning Match ---

    def _render_s1_word(self) -> None:
        if not self.book or self.s1_index >= len(self.batch_items):
            self.prompt_var.set("Stage 1 complete! Moving to Stage 2: Sentence Completion.")
            self._enter_stage(2)
            return

        w = self.book.words.get(self.batch_items[self.s1_index].word.lemma, self.batch_items[self.s1_index].word)
        self._update_word_banner(w)
        self.prompt_var.set(
            f"[{self.s1_index + 1}/{len(self.batch_items)}] Select the correct translation(s) for '{w.lemma}', then click Verify:"
        )

        # Build candidate pool from batch words' translations
        correct_pool = w.translations or [w.cached_translation or f"(meaning of {w.lemma})"]
        distractor_pool: list[str] = []
        for other in self.batch_items:
            if other.word.lemma == w.lemma:
                continue
            ow = self.book.words.get(other.word.lemma, other.word)
            for t in ow.translations:
                if t not in correct_pool and t not in distractor_pool:
                    distractor_pool.append(t)

        random.shuffle(distractor_pool)
        options = list(dict.fromkeys(correct_pool[:2] + distractor_pool[:4]))
        random.shuffle(options)

        for child in self.s1_checks_holder.winfo_children():
            child.destroy()
        self.s1_choice_vars.clear()

        for opt in options:
            var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.s1_checks_holder, text=opt, variable=var).pack(anchor="w", pady=2)
            self.s1_choice_vars.append((var, opt))

    def _verify_meaning_match(self) -> None:
        if not self.book or self.s1_index >= len(self.batch_items):
            return
        w = self.book.words.get(self.batch_items[self.s1_index].word.lemma, self.batch_items[self.s1_index].word)
        picked = [opt for var, opt in self.s1_choice_vars if var.get()]
        if not picked:
            messagebox.showinfo("Select Option", "Pick at least one translation before clicking Verify.")
            return

        valid_set = {t.lower().strip() for t in (w.translations or [])}
        is_correct = bool(valid_set) and all(p.lower().strip() in valid_set for p in picked)
        if not valid_set:
            # If word had no stored translations yet, count verification as attempt
            is_correct = True

        w.record_attempt(is_correct)
        refresh_word_mastery(w)
        save_book_index(self.book)
        self.on_stats_updated()

        self.trace(
            f"Verified '{w.lemma}': picked={picked}, correct={is_correct}, "
            f"new_score={w.correct}/{w.attempts} ({int(w.accuracy * 100)}%)"
        )
        self._update_word_banner(w)
        if is_correct:
            self._set_feedback(f"Verified! '{w.lemma}' -> {', '.join(w.translations or picked)}. Saved to JSON.")
        else:
            self._set_feedback(
                f"Not quite. Expected translation(s): {', '.join(w.translations)}. Updated knowledge stats in JSON."
            )

    def _next_s1(self) -> None:
        self.s1_index += 1
        self._render_s1_word()

    # --- Stage 2: Sentence Completion ---

    def _render_s2_question(self) -> None:
        if not self.book or self.s2_index >= len(self.s2_questions):
            self.prompt_var.set("Stage 2 complete! Moving to Stage 3: Sentence Translation.")
            self._enter_stage(3)
            return

        q = self.s2_questions[self.s2_index]
        w = self.book.words.get(q["answer_lemma"])
        if w:
            self._update_word_banner(w)

        self.prompt_var.set(
            f"[{self.s2_index + 1}/{len(self.s2_questions)}] Complete the sentence:\n\"{q['sentence']}\""
        )

        for child in self.s2_buttons_holder.winfo_children():
            child.destroy()

        batch_lemmas = [it.word.lemma for it in self.batch_items]
        random.shuffle(batch_lemmas)
        for cand in batch_lemmas:
            ttk.Button(
                self.s2_buttons_holder,
                text=cand,
                command=lambda choice=cand, expected=q["answer_lemma"]: self._check_s2(choice, expected),
            ).pack(side="left", padx=3, pady=3)

    def _check_s2(self, choice: str, expected: str) -> None:
        if not self.book:
            return
        is_correct = choice == expected
        w = self.book.words.get(expected)
        if w:
            w.record_attempt(is_correct)
            refresh_word_mastery(w)
            save_book_index(self.book)
            self.on_stats_updated()
            self._update_word_banner(w)

        self.trace(f"Stage 2 completion: chose='{choice}', expected='{expected}', correct={is_correct}")
        if is_correct:
            self._set_feedback(f"Correct! '{expected}' fits the blank.")
        else:
            self._set_feedback(f"You chose '{choice}', but the blank belongs to '{expected}'.")

        self.s2_index += 1
        self._render_s2_question()

    # --- Stage 3: Sentence Translation (tev1:4b -> Main LLM) ---

    def _render_s3_item(self) -> None:
        if self.s3_index >= len(self.s3_items):
            self.prompt_var.set("All 3 stages completed for this batch! Click 'Load & Enrich Batch' for the next batch.")
            self._set_feedback("Batch finished.")
            return

        item = self.s3_items[self.s3_index]
        if self.book and item["lemma"] in self.book.words:
            self._update_word_banner(self.book.words[item["lemma"]])

        self.translation_input_var.set("")
        self.prompt_var.set(
            f"[{self.s3_index + 1}/{len(self.s3_items)}] Translate into English:\n\"{item['sentence']}\""
        )

    def _submit_translation_stage(self) -> None:
        if not self.book or self.s3_index >= len(self.s3_items):
            return
        user_trans = self.translation_input_var.get().strip()
        if not user_trans:
            return

        item = self.s3_items[self.s3_index]
        src_sent = item["sentence"]
        lang = self.book.language
        main_model = self.ollama.model

        self._set_feedback(f"Checking translation with decision model '{DECISION_MODEL}'...")
        self.trace(f"Stage 3: asking decision model '{DECISION_MODEL}' to validate translation...")

        def worker() -> str:
            verdict = self.ollama.judge_translation_with_decision_model(
                source_sentence=src_sent,
                user_translation=user_trans,
                source_lang=lang,
                decision_model=DECISION_MODEL,
            )
            is_valid = bool(verdict.get("valid"))
            reason = str(verdict.get("reason", ""))
            self.trace(f"Decision model ({DECISION_MODEL}) verdict: valid={is_valid} ({reason})")

            if is_valid:
                return f"✔ Valid translation ({DECISION_MODEL}): {reason}"

            self.trace(f"Translation invalid -> invoking main LLM '{main_model}' for detailed explanation...")
            explanation = self.ollama.explain_sentence(
                sentence=f"{src_sent}\n[Learner attempted: {user_trans} | Decision note: {reason}]",
                source_lang=lang,
            )
            return f"✖ Marked invalid by {DECISION_MODEL} ({reason})\n\n--- Explanation ({main_model}) ---\n{explanation}"

        def on_done(output: str) -> None:
            self._set_feedback(output)

        threading.Thread(target=lambda: self.after(0, lambda: on_done(worker())), daemon=True).start()

    def _next_s3(self) -> None:
        self.s3_index += 1
        self._render_s3_item()

    # --- Red Flag Word JSON Inspector ---

    def _open_red_flag_for_active_word(self) -> None:
        if not self.book:
            return
        entry = self._get_active_word_entry()
        if not entry:
            messagebox.showinfo("No Active Word", "Load a batch first to inspect a word's JSON.")
            return

        self.trace(f"🚩 Red Flag opened for lemma '{entry.lemma}'.")

        def handle_save(updated: WordEntry) -> None:
            assert self.book is not None
            self.book.words[updated.lemma] = updated
            save_book_index(self.book)
            self._update_word_banner(updated)
            self.on_stats_updated()
            self.trace(f"🚩 Saved manual JSON edits for '{updated.lemma}' to book JSON.")
            self._set_feedback(f"Saved manual JSON edits for '{updated.lemma}'.")

        RedFlagWordDialog(self, entry, on_save=handle_save)