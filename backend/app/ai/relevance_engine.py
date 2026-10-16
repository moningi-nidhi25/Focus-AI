"""Semantic relevance engine (Phase 4).

Embedding/rule signals (all normalised to 0..1, then combined into a 0..100 "local" score):
  * chunk similarity   - mean of the 3 best passages vs. the context embedding
  * document similarity- whole content vs. the context embedding
  * keyword coverage   - weighted share of context terms/phrases found in the content
  * title similarity   - page title vs. the context embedding

How the final score is produced (three evaluation paths, reported as ``evaluation_path``):

  1. NORMAL CONTENT            -> ``embedding``
        final = local score. The LLM is not called: the embedding pipeline is calibrated and
        validated on the labelled set, deterministic, free and not prompt-injectable.

  2. AMBIGUOUS CONTENT         -> ``embedding+llm``
        Triggered when the local score is within the ambiguity band (default: 5 points below the
        WARN threshold up to the ALLOW threshold), i.e. where similarity alone cannot settle the
        decision.   final = 0.6 * local + 0.4 * llm.
        Embeddings keep the majority because they are the validated primary signal; the LLM gets
        enough weight (0.4) to move a score across a threshold when it strongly disagrees
        (e.g. local 60 + LLM 100 -> 76 ALLOW; local 40 + LLM 0 -> 24 BLOCK) but not enough to
        override clear embedding evidence on its own.

  3. LOW-CONTENT PAGES         -> ``llm+embedding``  (little body text, e.g. a video with only a
        title/description when no transcript could be fetched)
        Embeddings of a single short string are high-variance and key-term coverage is nearly
        binary, so here the LLM (which uses world knowledge about the title/description) is the
        main reviewer:   final = 0.3 * local + 0.7 * llm.   The embedding score is kept as an
        anchor so a hallucinating or manipulated LLM cannot fully decide alone.

  If the LLM is not configured or fails (timeout, HTTP error, malformed answer) the engine falls
  back to the embedding score alone and says so in ``llm_status`` / the reasons. With no body text
  the stricter low-content block rule applies (block only when local < ``low_content_block_below``).

  * INSUFFICIENT: when title + description + text together contain fewer than 3 meaningful words
    there is nothing for any model to judge. Only then is the neutral fail-open result returned
    (50 / ALLOW, ``evaluation_path="insufficient"``).

Thresholds are unchanged: ALLOW >= allow_threshold, WARN >= warn_threshold, otherwise BLOCK.
"""
import os
from collections import Counter
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.ai.context import ContextProfile
from app.ai.embeddings import Embedder, cosine, get_embedder
from app.ai.explainability import (
    METHOD,
    build_explanation,
    build_reasons,
    level_for,
    matching_concepts,
    score_sentences,
)
from app.ai.llm import LLMReviewer, LLMReviewInput, get_llm
from app.ai.text_utils import (
    analyze_tokens,
    content_bigrams,
    content_stems,
    make_chunks,
    split_sentences,
)

MAX_WORDS = 4000
MAX_HEATMAP_SENTENCES = 120
COVERAGE_SATURATION = 0.6
CLASS_TO_DECISION = {"RELEVANT": "ALLOW", "PARTIALLY_RELEVANT": "WARN", "DISTRACTING": "BLOCK"}


@dataclass
class EngineConfig:
    allow_threshold: float = 75.0  # score >= -> RELEVANT / ALLOW
    warn_threshold: float = 30.0  # score >= -> PARTIALLY_RELEVANT / WARN, below -> BLOCK
    low_content_words: int = 25  # fewer body words than this -> "low content"
    low_content_block_below: float = 15.0  # embeddings alone on little text: only block when clearly unrelated
    min_content_stems: int = 3  # fewer meaningful words than this in total -> nothing to judge
    llm_band: Optional[Tuple[float, float]] = None  # None -> (warn_threshold - 5, allow_threshold)
    llm_weight: float = 0.4  # LLM share when embeddings are ambiguous (see module docstring)
    llm_weight_low_content: float = 0.7  # LLM share when there is little text to embed

    @property
    def ambiguity_band(self) -> Tuple[float, float]:
        return self.llm_band or (self.warn_threshold - 5, self.allow_threshold)

    @classmethod
    def from_env(cls) -> "EngineConfig":
        return cls(
            allow_threshold=float(os.getenv("RELEVANCE_ALLOW_THRESHOLD", 75)),
            warn_threshold=float(os.getenv("RELEVANCE_WARN_THRESHOLD", 30)),
        )


@dataclass
class ContentInput:
    title: str = ""
    text: str = ""  # main body / transcript
    url: str = ""
    description: str = ""  # page or video description (kept separate so the LLM sees it labelled)
    metadata: str = ""  # e.g. "Channel: X | Category: Y | Tags: a, b"
    source_type: str = "webpage"


@dataclass
class AnalysisResult:
    score: float
    classification: str
    decision: str
    explanation: str
    reasons: List[str]
    concepts: List[str]
    signals: Dict[str, Optional[float]]
    heatmap: List[Tuple[str, float]]  # (sentence, score 0..1) in reading order
    highlighted: List[Tuple[str, float]]  # top sentences
    word_count: int
    insufficient_content: bool = False
    low_content: bool = False
    llm_used: bool = False
    embedding_backend: str = ""
    method: str = METHOD
    # How the score was produced: embedding | embedding+llm | llm+embedding | insufficient
    evaluation_path: str = "embedding"
    # not_configured | not_needed | ok | failed:<unavailable|timeout|http_error|malformed>
    llm_status: str = "not_configured"
    content_quality: str = "sufficient"  # sufficient | low | insufficient


class RelevanceEngine:
    def __init__(self, embedder: Optional[Embedder] = None, llm: Optional[LLMReviewer] = None, config: Optional[EngineConfig] = None):
        self.embedder = embedder or get_embedder()
        self.llm = llm
        self.config = config or EngineConfig()

    # ------------------------------------------------------------------ helpers
    def embed_context(self, profile: ContextProfile) -> np.ndarray:
        return self.embedder.embed([profile.text])[0]

    @staticmethod
    def _keyword_coverage(profile: ContextProfile, counts: Counter, title_stems: set, bigrams: set, title_bigrams: set) -> Tuple[float, int]:
        total = sum(profile.term_weights.values())
        if not total:
            return 0.0, 0
        got, found = 0.0, 0
        for stem_, w in profile.term_weights.items():
            mentions = counts.get(stem_, 0) + (2 if stem_ in title_stems else 0)
            if mentions:
                found += 1
            got += w * min(1.0, mentions / 2)
        cov = got / total
        if profile.phrases:
            ptotal = sum(w for _, w in profile.phrases)
            pgot = sum(w for pair, w in profile.phrases if pair in bigrams or pair in title_bigrams)
            cov = 0.7 * cov + 0.3 * (pgot / ptotal)
        # No single page covers every term of a multi-part goal: covering ~60% counts as full coverage.
        return min(1.0, cov / COVERAGE_SATURATION), found

    # ------------------------------------------------------------------ main
    def analyze(self, profile: ContextProfile, content: ContentInput, context_vector: Optional[np.ndarray] = None) -> AnalysisResult:
        cfg, emb = self.config, self.embedder
        title = (content.title or "").strip()
        description = (content.description or "").strip()
        # Title/description/metadata count as evidence just like body text (e.g. a video with no transcript).
        body = " ".join(p for p in (description, (content.metadata or "").strip(), content.text or "") if p)
        words = body.split()
        text = " ".join(words[:MAX_WORDS])
        word_count = len(words)
        llm_status = "not_configured" if self.llm is None else "not_needed"

        toks = analyze_tokens(f"{title}. {text}" if title else text)
        if len(content_stems(toks)) < cfg.min_content_stems:
            # Truly nothing meaningful to judge (blank page, "Home / Menu", an unknown video with no
            # readable title or description). Neutral fail-open: never block what we cannot read.
            return AnalysisResult(
                score=50.0, classification="PARTIALLY_RELEVANT", decision="ALLOW",
                explanation=build_explanation("PARTIALLY_RELEVANT", 50, profile.topic, [], None, True),
                reasons=["No usable title, description or text was available to judge this page."],
                concepts=[], signals={}, heatmap=[], highlighted=[], word_count=word_count,
                insufficient_content=True, low_content=True, embedding_backend=emb.name,
                evaluation_path="insufficient", llm_status=llm_status, content_quality="insufficient",
            )

        sentences = split_sentences(text, max_sentences=250) or ([text] if text else [])
        chunks = make_chunks(sentences)
        low_content = word_count < cfg.low_content_words
        ctx = context_vector if context_vector is not None else self.embed_context(profile)

        batch = list(sentences) + [c[2] for c in chunks] + ([title] if title else [])
        vecs = emb.embed(batch)
        ns, nc = len(sentences), len(chunks)
        sent_vecs, chunk_vecs = vecs[:ns], vecs[ns : ns + nc]
        title_vec = vecs[-1] if title else None

        # ---- semantic signals
        chunk_cals = [emb.calibrate(cosine(ctx, v)) for v in chunk_vecs]
        chunk_sim = float(np.mean(sorted(chunk_cals, reverse=True)[:3])) if chunk_cals else 0.0
        doc_sim = emb.calibrate(cosine(ctx, chunk_vecs.mean(axis=0))) if nc else 0.0
        title_sim = emb.calibrate(cosine(ctx, title_vec)) if title_vec is not None else None

        # ---- lexical coverage signal
        body_toks = analyze_tokens(text)
        title_toks = analyze_tokens(title)
        kw_cov, terms_found = self._keyword_coverage(
            profile, Counter(content_stems(body_toks)), set(content_stems(title_toks)),
            set(content_bigrams(body_toks)), set(content_bigrams(title_toks)),
        )

        if title_sim is not None:
            local = 0.35 * chunk_sim + 0.25 * doc_sim + 0.25 * kw_cov + 0.15 * title_sim
        else:
            local = 0.40 * chunk_sim + 0.30 * doc_sim + 0.30 * kw_cov
        score = local * 100

        # ---- sentence-level attribution (heatmap)
        sent_scores = score_sentences(emb, profile, sentences, sent_vecs, ctx) if ns else []
        order = sorted(range(ns), key=lambda i: -sent_scores[i])
        highlighted = [(sentences[i], sent_scores[i]) for i in order[:3] if sent_scores[i] >= 0.35]
        heatmap = list(zip(sentences, sent_scores))[:MAX_HEATMAP_SENTENCES]
        frac_relevant = (sum(1 for s in sent_scores if s >= 0.5) / ns) if ns else None
        concepts = matching_concepts(profile, toks)

        # ---- LLM contextual review: only for low-content pages or when embeddings are ambiguous
        llm_score, llm_reasoning, llm_used, path, notes = None, "", False, "embedding", []
        lo, hi = cfg.ambiguity_band
        needs_llm = low_content or lo <= score <= hi
        if self.llm is not None and needs_llm:
            excerpt = ""
            if (content.text or "").strip():  # description/metadata are sent labelled, not duplicated here
                best = [chunks[i][2] for i in sorted(range(nc), key=lambda i: -chunk_cals[i])[:3]]
                excerpt = " ... ".join(dict.fromkeys([chunks[0][2], *best]))[:3000]
            outcome = self.llm.review(LLMReviewInput(
                topic=profile.topic, objectives=profile.objectives, keywords=profile.keywords,
                title=title, description=description, metadata=(content.metadata or "").strip(),
                content=excerpt, url=content.url, source_type=content.source_type,
                allow_threshold=cfg.allow_threshold, warn_threshold=cfg.warn_threshold,
            ))
            if outcome.judgement is not None:
                judged = outcome.judgement
                w = cfg.llm_weight_low_content if low_content else cfg.llm_weight
                llm_used, llm_score, llm_reasoning, llm_status = True, judged.score, judged.reason, "ok"
                path = "llm+embedding" if low_content else "embedding+llm"
                score = (1 - w) * score + w * judged.score
                for c in judged.concepts:
                    if c.lower() not in (x.lower() for x in concepts):
                        concepts.append(c)
                concepts = concepts[:8]
            else:
                llm_status = f"failed:{outcome.status}"
                notes.append(f"The AI reviewer was unavailable ({outcome.status}); this decision uses the embedding analysis only.")

        score = round(max(0.0, min(100.0, score)), 1)
        # With little text and no LLM, embeddings alone are unreliable: only block when clearly unrelated.
        block_below = cfg.low_content_block_below if (low_content and not llm_used) else cfg.warn_threshold
        if score >= cfg.allow_threshold:
            classification = "RELEVANT"
        elif score >= block_below:
            classification = "PARTIALLY_RELEVANT"
        else:
            classification = "DISTRACTING"

        reasons = build_reasons(terms_found, len(profile.term_weights), frac_relevant, doc_sim, title_sim, low_content, llm_reasoning)
        reasons.extend(notes)
        explanation = build_explanation(classification, score, profile.topic, concepts, highlighted[0][0] if highlighted else None, False)
        pct = lambda v: None if v is None else round(v * 100, 1)  # noqa: E731
        return AnalysisResult(
            score=score, classification=classification, decision=CLASS_TO_DECISION[classification],
            explanation=explanation, reasons=reasons, concepts=concepts,
            signals={
                "chunk_similarity": pct(chunk_sim), "embedding_similarity": pct(doc_sim),
                "keyword_coverage": pct(kw_cov), "title_similarity": pct(title_sim),
                "local_score": round(local * 100, 1), "llm_score": llm_score,
            },
            heatmap=heatmap, highlighted=highlighted, word_count=word_count,
            low_content=low_content, llm_used=llm_used, embedding_backend=emb.name,
            evaluation_path=path, llm_status=llm_status, content_quality="low" if low_content else "sufficient",
        )


@lru_cache(maxsize=1)
def get_engine() -> RelevanceEngine:
    return RelevanceEngine(embedder=get_embedder(), llm=get_llm(), config=EngineConfig.from_env())
