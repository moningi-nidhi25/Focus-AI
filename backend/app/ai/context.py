"""Structured representation of a study context (topic + objectives + course material)."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from app.ai.text_utils import (
    GENERIC_STEMS,
    analyze_tokens,
    content_bigrams,
    surface_map,
    top_terms,
)

MAX_MATERIAL_CHARS = 60_000


@dataclass
class ContextProfile:
    topic: str
    objectives: str = ""
    material: str = ""
    text: str = ""  # text that is embedded to represent the context
    term_weights: Dict[str, float] = field(default_factory=dict)  # stem -> weight
    phrases: List[Tuple[Tuple[str, str], float]] = field(default_factory=list)  # (stem pair, weight)
    keywords: List[str] = field(default_factory=list)  # human-readable, for UI


def build_profile(topic: str, objectives: str = "", material: str = "") -> ContextProfile:
    topic = (topic or "").strip()
    objectives = (objectives or "").strip()
    material = (material or "")[:MAX_MATERIAL_CHARS]

    t_toks = analyze_tokens(topic)
    o_toks = analyze_tokens(objectives)
    material_terms = top_terms(material, 8) if material else []

    weights: Dict[str, float] = {}

    def add(stem_: str, w: float) -> None:
        if len(stem_) >= 3 and stem_ not in GENERIC_STEMS:
            weights[stem_] = max(weights.get(stem_, 0.0), w)

    for tok in t_toks:
        if tok.content:
            add(tok.stem, 2.0)
    for tok in o_toks:
        if tok.content:
            add(tok.stem, 1.0)
    for stem_, _ in material_terms:
        add(stem_, 0.5)

    phrases: Dict[Tuple[str, str], float] = {}
    for pair in content_bigrams(t_toks):
        if not set(pair) & GENERIC_STEMS:
            phrases[pair] = max(phrases.get(pair, 0.0), 2.0)
    for pair in content_bigrams(o_toks):
        if not set(pair) & GENERIC_STEMS:
            phrases[pair] = max(phrases.get(pair, 0.0), 1.0)

    # Human-readable keyword list: phrases first, then single words not already in a phrase.
    surfaces = {**surface_map(o_toks), **surface_map(t_toks)}
    surfaces.update({s: surf for s, surf in material_terms})
    keywords: List[str] = []
    in_phrase = set()
    for (a, b) in phrases:
        in_phrase.update((a, b))
        kw = f"{surfaces.get(a, a)} {surfaces.get(b, b)}"
        if kw not in keywords:
            keywords.append(kw)
    for stem_ in sorted(weights, key=lambda s: -weights[s]):
        if stem_ not in in_phrase:
            kw = surfaces.get(stem_, stem_)
            if kw not in keywords:
                keywords.append(kw)

    text_parts = [p for p in (topic, objectives, " ".join(s for _, s in material_terms)) if p]
    return ContextProfile(
        topic=topic,
        objectives=objectives,
        material=material,
        text=". ".join(text_parts),
        term_weights=weights,
        phrases=list(phrases.items()),
        keywords=keywords[:24],
    )
