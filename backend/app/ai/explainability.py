"""Explainable-AI helpers (Phase 6).

Method used: *sentence-level semantic similarity* between the study context and each
sentence of the content (blended with key-term presence). This is a model-agnostic
attribution and is valid for any embedding backend, including black-box APIs.

This module's sentence-level scores are NOT Integrated Gradients and are not labelled as such
(``METHOD`` is unchanged). Genuine token-level Integrated Gradients for the embedding-cosine component
lives in ``app/ai/integrated_gradients.py`` and is served separately by ``/relevance/explain-tokens``.
LRP is not implemented, and neither method is valid for the external LLM reviewer.
"""
from collections import Counter
from math import log
from typing import Dict, List, Optional, Sequence

import numpy as np

from app.ai.context import ContextProfile
from app.ai.embeddings import Embedder
from app.ai.text_utils import (
    GENERIC_STEMS,
    Tok,
    analyze_tokens,
    content_bigrams,
    content_stems,
    surface_map,
)

METHOD = "sentence-level semantic similarity"
HIGH, MEDIUM = 0.6, 0.3


def level_for(score: float) -> str:
    return "high" if score >= HIGH else "medium" if score >= MEDIUM else "low"


def score_sentences(
    embedder: Embedder, profile: ContextProfile, sentences: Sequence[str], sent_vecs: np.ndarray, ctx_vec: np.ndarray
) -> List[float]:
    """Per-sentence relevance in 0..1 (calibrated cosine blended with key-term presence)."""
    total_w = sum(profile.term_weights.values()) or 1.0
    scores: List[float] = []
    for sentence, vec in zip(sentences, sent_vecs):
        cal = embedder.calibrate(float(np.dot(vec, ctx_vec)))
        stems = set(content_stems(analyze_tokens(sentence)))
        hit = sum(w for st, w in profile.term_weights.items() if st in stems)
        kw = min(1.0, hit / (0.4 * total_w))
        scores.append(round(0.7 * cal + 0.3 * kw, 4))
    return scores


def matching_concepts(profile: ContextProfile, toks: List[Tok], limit: int = 8) -> List[str]:
    """Concepts in the content that connect to the study context."""
    surf = surface_map(toks)
    counts = Counter(content_stems(toks))
    bigram_counts = Counter(content_bigrams(toks))
    weights = profile.term_weights
    phrase_pairs = {p for p, _ in profile.phrases}

    candidates: List[tuple] = []  # (score, label, stems)
    for pair, w in profile.phrases:
        c = bigram_counts.get(pair, 0)
        if c:
            candidates.append((w * (1 + log(c)), f"{surf[pair[0]]} {surf[pair[1]]}", set(pair)))
    for pair, c in bigram_counts.items():
        if c >= 2 and pair not in phrase_pairs and not set(pair) & GENERIC_STEMS and any(s in weights for s in pair):
            candidates.append((0.6 * (1 + log(c)), f"{surf[pair[0]]} {surf[pair[1]]}", set(pair)))
    candidates.sort(key=lambda x: -x[0])

    chosen: List[str] = []
    covered: set = set()
    for _, label, stems in candidates:
        if label not in chosen and not stems <= covered:
            chosen.append(label)
            covered |= stems
    singles = [
        (w * (1 + log(counts[s])) * 0.7, surf[s], s) for s, w in weights.items() if counts.get(s) and s not in covered
    ]
    for _, label, _s in sorted(singles, key=lambda x: -x[0]):
        if label not in chosen:
            chosen.append(label)
    return chosen[:limit]


def build_reasons(
    terms_found: int,
    terms_total: int,
    frac_relevant: Optional[float],
    doc_sim: float,
    title_sim: Optional[float],
    low_content: bool,
    llm_reasoning: str = "",
) -> List[str]:
    reasons: List[str] = []
    if terms_total:
        reasons.append(f"{terms_found} of {terms_total} key terms from your study context appear in the content.")
    if frac_relevant is not None:
        reasons.append(f"{round(frac_relevant * 100)}% of the passages relate to your topic.")
    reasons.append(f"Overall semantic similarity to your context: {round(doc_sim * 100)}%.")
    if title_sim is not None:
        if title_sim >= 0.5:
            reasons.append("The page title matches your topic.")
        elif title_sim < 0.15:
            reasons.append("The page title does not mention your topic.")
    if low_content:
        reasons.append("Very little text was available, so the title carried most of the decision.")
    if llm_reasoning:
        reasons.append(f"AI reviewer: {llm_reasoning}")
    return reasons


def build_explanation(
    classification: str, score: float, topic: str, concepts: List[str], best_sentence: Optional[str], insufficient: bool
) -> str:
    pct = round(score)
    if insufficient:
        return "Allowed: FocusAI could not read a usable title, description or text for this page, so it could not judge it and did not block it."
    shown = ", ".join(concepts[:4])
    if classification == "RELEVANT":
        text = f"Allowed: this content is {pct}% relevant to your study topic \u201c{topic}\u201d."
        if shown:
            text += f" It covers {shown}."
        if best_sentence:
            text += f" Most relevant passage: \u201c{_clip(best_sentence)}\u201d"
        return text
    if classification == "PARTIALLY_RELEVANT":
        text = f"Warning: this content is only partly related ({pct}%) to \u201c{topic}\u201d."
        if shown:
            text += f" It touches on {shown}, but much of it is about other things."
        return text + " Make sure it supports your learning objective."
    text = f"Blocked: this content is only {pct}% relevant to \u201c{topic}\u201d and does not connect to your learning objectives."
    if shown:
        text += f" The only overlap found: {shown}."
    return text


def _clip(s: str, n: int = 160) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "\u2026"
