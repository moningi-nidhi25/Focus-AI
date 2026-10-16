"""Lightweight, dependency-free NLP helpers used by the relevance engine.

Everything here is deterministic and works offline: tokenisation, a light
stemmer, sentence splitting, chunking and keyword extraction.
"""
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

STOPWORDS = frozenset(
    """a about above after again all also am an and any are aren't as at be because been
    before being below between both but by can can't cannot could did do does doing don't
    down during each few for from further had has have having he her here hers him his how
    i if in into is it its itself just let me more most my no nor not of off on once only or
    other our out over own same she should so some such than that the their theirs them then
    there these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours yourself may might must one
    two get got like via per etc s t""".split()
)

# Words that appear in many study goals ("understand ...", "basics of ...") and say
# nothing about the subject itself. They get no weight when matching content.
GENERIC_WORDS = frozenset(
    """understand learn learning study basic basics introduction intro overview concept
    concepts explain explanation know topic topics chapter lesson able student students
    course using use work works example examples way make need good best help important
    different thing things time new first many main key part""".split()
)

_TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[A-Z0-9\"'(\[])|\n+")


def stem(word: str) -> str:
    """Very small suffix stripper: deadlocks/deadlocked -> deadlock, allocation/allocate -> allocat."""
    w = word.lower()
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for suf in ("ing", "ed", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    if w.endswith("ion") and len(w) > 5:
        w = w[:-3]
    if w.endswith("e") and len(w) > 4:
        w = w[:-1]
    return w


GENERIC_STEMS = frozenset(stem(w) for w in GENERIC_WORDS)


@dataclass(frozen=True)
class Tok:
    stem: str
    surface: str
    content: bool  # False for stopwords / numbers / 1-char tokens


def analyze_tokens(text: str) -> List[Tok]:
    toks: List[Tok] = []
    for raw in _TOKEN_RE.findall((text or "").replace("\u2019", "'").lower()):
        is_content = raw not in STOPWORDS and len(raw) > 1 and not raw.isdigit()
        surface = raw[:-2] if raw.endswith("'s") else raw
        toks.append(Tok(stem(raw), surface, is_content))
    return toks


def content_stems(toks: List[Tok]) -> List[str]:
    return [t.stem for t in toks if t.content]


def content_bigrams(toks: List[Tok]) -> List[Tuple[str, str]]:
    """Adjacent content-word pairs (no stopword in between)."""
    return [(a.stem, b.stem) for a, b in zip(toks, toks[1:]) if a.content and b.content]


def _wrap_long(sentence: str, max_words: int = 45) -> List[str]:
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    return [" ".join(words[i : i + 30]) for i in range(0, len(words), 30)]


def split_sentences(text: str, min_words: int = 4, max_sentences: int = 400) -> List[str]:
    """Split text into sentences; long unpunctuated runs (e.g. transcripts) are windowed."""
    out: List[str] = []
    for part in _SENT_SPLIT_RE.split(text or ""):
        part = (part or "").strip()
        if not part:
            continue
        for piece in _wrap_long(part):
            if len(piece.split()) >= min_words:
                out.append(piece)
            if len(out) >= max_sentences:
                return out
    return out


def make_chunks(sentences: List[str], target_words: int = 70) -> List[Tuple[int, int, str]]:
    """Group consecutive sentences into ~target_words chunks. Returns (start, end_exclusive, text)."""
    chunks: List[Tuple[int, int, str]] = []
    start, words = 0, 0
    for i, s in enumerate(sentences):
        words += len(s.split())
        if words >= target_words:
            chunks.append((start, i + 1, " ".join(sentences[start : i + 1])))
            start, words = i + 1, 0
    if start < len(sentences):
        chunks.append((start, len(sentences), " ".join(sentences[start:])))
    return chunks


def top_terms(text: str, n: int = 8) -> List[Tuple[str, str]]:
    """Most frequent content words as (stem, surface) pairs, ignoring generic study words."""
    counts: Counter = Counter()
    surfaces: Dict[str, Counter] = {}
    for t in analyze_tokens(text):
        if t.content and len(t.stem) >= 3 and t.stem not in GENERIC_STEMS:
            counts[t.stem] += 1
            surfaces.setdefault(t.stem, Counter())[t.surface] += 1
    return [(s, surfaces[s].most_common(1)[0][0]) for s, _ in counts.most_common(n)]


def surface_map(toks: List[Tok]) -> Dict[str, str]:
    """stem -> most common surface form."""
    surfaces: Dict[str, Counter] = {}
    for t in toks:
        if t.content:
            surfaces.setdefault(t.stem, Counter())[t.surface] += 1
    return {s: c.most_common(1)[0][0] for s, c in surfaces.items()}
