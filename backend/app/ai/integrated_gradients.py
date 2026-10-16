"""Token-level Integrated Gradients (IG) for the sentence <-> study-context cosine similarity.

WHAT IS EXPLAINED
    Target function  F(x) = cos( pool(model(x)), c )
    * x  : the *word-embedding* matrix (L x H) of the sentence, i.e. the output of the model's input
           embedding layer (``inputs_embeds``). Token ids are discrete, so gradients are taken w.r.t.
           these continuous vectors. Position / segment embeddings are added inside the model and are
           NOT attributed.
    * pool: attention-masked mean pooling followed by L2 normalisation, exactly what the
           all-MiniLM-L6-v2 SentenceTransformer does (Transformer -> mean Pooling -> Normalize).
    * c   : the study-context embedding, produced by the existing pipeline (``Embedder.embed``) and
           held FIXED (no gradient flows into it).
    The cosine of two unit vectors equals the "semantic similarity" term already used by FocusAI.

WHAT IS *NOT* EXPLAINED
    The keyword-matching term, the calibration, the 0.7/0.3 blend, the thresholds and the final
    ALLOW/WARN/BLOCK decision are not covered. Only the embedding-cosine component is.

BASELINE
    Every non-special position is replaced by the [PAD] word embedding; special tokens ([CLS], [SEP])
    keep their own embedding (so they receive exactly zero attribution) and the attention mask is
    unchanged. The baseline is "same length, no lexical content". Attributions depend on this choice;
    a different baseline (e.g. zeros, [MASK]) gives different numbers.

PATH INTEGRAL
    IG_i = (x_i - x'_i) * integral_0^1 dF(x' + a (x - x')) / dx_i da, approximated with the trapezoid
    rule over ``steps`` intervals. Completeness check: sum_i IG_i should equal F(x) - F(x'). The gap is
    reported (``completeness_error``); if it exceeds the tolerance the step count is doubled once
    (up to ``MAX_STEPS``) and the final ``converged`` flag is returned honestly.

SIGNS / SCALES
    * ``attribution`` values are RAW IG attributions (units of cosine similarity). Positive = the word
      pushes the sentence embedding TOWARD the study-context embedding relative to the baseline,
      negative = pushes it away. They are not "relevance %" and not importance magnitudes.
    * ``normalized`` = attribution / max|attribution| in [-1, 1]: a DISPLAY-ONLY value for colouring.

Torch is imported lazily; the integration kernel, word aggregation and cache are torch-free so they can
be unit-tested without it.
"""
import hashlib
import logging
import re
import threading
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("focusai.ig")

METHOD = "integrated-gradients"
DEFAULT_STEPS = 32
MAX_STEPS = 64
MAX_TOKENS = 128
BATCH_SIZE = 16
REL_TOL = 0.05    # |sum(attr) - (F(x)-F(x'))| <= 5% of |F(x)-F(x')| ...
ABS_TOL = 0.002   # ... or <= 0.002 absolute, whichever is larger

TARGET_DESCRIPTION = "cosine similarity between the sentence embedding and the study-context embedding"
SCOPE_DESCRIPTION = (
    "Explains only the embedding-similarity component. It does not explain the keyword term, "
    "the score calibration, or the final ALLOW/WARN/BLOCK decision."
)
BASELINE_DESCRIPTION = "[PAD] word embeddings at every non-special position ([CLS]/[SEP] kept)"
SCORE_SEMANTICS = (
    "attribution = raw Integrated Gradients value (cosine units): positive moves the sentence toward the study "
    "context, negative moves it away. normalized = attribution / max|attribution|, for colouring only."
)


class AttributionUnavailable(Exception):
    """The configuration cannot produce IG (lexical backend, unsupported model, empty input...)."""

    def __init__(self, reason: str, message: str = ""):
        super().__init__(message or reason)
        self.reason = reason
        self.message = message or reason


class AttributionFailed(Exception):
    """IG ran but produced unusable numbers (non-finite values, ...)."""


# --------------------------------------------------------------------------- numeric kernel (numpy)
@dataclass
class IGResult:
    token_attributions: np.ndarray  # (L,) sum over the hidden dimension
    f_input: float
    f_baseline: float
    sum_attributions: float
    completeness_error: float       # |sum - (f_input - f_baseline)|
    relative_error: Optional[float]  # completeness_error / |f_input - f_baseline| (None if the denominator ~ 0)
    converged: bool
    steps: int


ValueGradFn = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


def integrated_gradients(
    x: np.ndarray, baseline: np.ndarray, fn: ValueGradFn, steps: int = DEFAULT_STEPS, batch_size: int = BATCH_SIZE
) -> IGResult:
    """Trapezoid-rule IG for a scalar function.

    ``fn(points)`` receives (B, L, H) points and returns (values (B,), gradients (B, L, H)).
    Attributions are summed over the hidden dimension H, giving one value per position L.
    """
    x = np.asarray(x, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if x.ndim != 2 or x.shape != baseline.shape:
        raise ValueError(f"x and baseline must both be (L, H); got {x.shape} and {baseline.shape}")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    alphas = np.linspace(0.0, 1.0, steps + 1)
    weights = np.full(steps + 1, 1.0 / steps)
    weights[0] = weights[-1] = 0.5 / steps
    delta = x - baseline

    grad_sum = np.zeros_like(x)
    values = np.empty(steps + 1, dtype=np.float64)
    for i in range(0, steps + 1, batch_size):
        a = alphas[i : i + batch_size]
        points = baseline[None, :, :] + a[:, None, None] * delta[None, :, :]
        v, g = fn(points)
        v, g = np.asarray(v, dtype=np.float64), np.asarray(g, dtype=np.float64)
        if g.shape != points.shape or v.shape != (len(a),):
            raise ValueError("fn must return values (B,) and gradients with the same shape as the points")
        values[i : i + len(a)] = v
        grad_sum += np.tensordot(weights[i : i + len(a)], g, axes=(0, 0))

    token_attr = (delta * grad_sum).sum(axis=1)
    if not (np.isfinite(token_attr).all() and np.isfinite(values[[0, -1]]).all()):
        raise AttributionFailed("non-finite values in attributions or target")

    f0, f1 = float(values[0]), float(values[-1])
    total = float(token_attr.sum())
    diff = f1 - f0
    err = abs(total - diff)
    rel = err / abs(diff) if abs(diff) > 1e-9 else None
    return IGResult(
        token_attributions=token_attr, f_input=f1, f_baseline=f0, sum_attributions=total,
        completeness_error=err, relative_error=rel, converged=err <= max(REL_TOL * abs(diff), ABS_TOL), steps=steps,
    )


# --------------------------------------------------------------------------- word aggregation (numpy)
_WORD_RE = re.compile(r"\w+(?:['’\-]\w+)*|[^\w\s]", re.UNICODE)


@dataclass
class WordAttribution:
    word: str
    start: int
    end: int
    attribution: float
    normalized: float = 0.0
    tokens: List[Tuple[str, float]] = field(default_factory=list)


def normalize_scores(values: Sequence[float]) -> List[float]:
    """Scale by max |value| into [-1, 1] (sign preserved). All-zero input stays zero."""
    peak = max((abs(v) for v in values), default=0.0)
    if peak <= 0 or not np.isfinite(peak):
        return [0.0 for _ in values]
    return [float(v) / peak for v in values]


def aggregate_words(
    text: str,
    tokens: Sequence[str],
    offsets: Sequence[Tuple[int, int]],
    special_mask: Sequence[bool],
    token_attr: Sequence[float],
) -> List[WordAttribution]:
    """Sum sub-word token attributions into the words/punctuation marks of ``text``.

    Summation (not mean/max) keeps the completeness property: the word scores add up to the sum of the
    non-special token scores, and the sign of each contribution is preserved. Special tokens carry
    no text and are skipped; words with no token (cut off by truncation) are omitted.
    """
    spans = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    starts = [s for s, _ in spans]
    words: Dict[int, WordAttribution] = {}
    for tok, (s, e), special, a in zip(tokens, offsets, special_mask, token_attr):
        if special or e <= s or not spans:
            continue
        idx = bisect_right(starts, s) - 1
        if idx < 0 or s >= spans[idx][1]:
            continue  # token lies in whitespace: cannot belong to a word
        w = words.get(idx)
        if w is None:
            ws, we = spans[idx]
            w = words[idx] = WordAttribution(word=text[ws:we], start=ws, end=we, attribution=0.0)
        w.attribution += float(a)
        w.tokens.append((tok, float(a)))
    ordered = [words[i] for i in sorted(words)]
    for w, n in zip(ordered, normalize_scores([w.attribution for w in ordered])):
        w.normalized = n
    return ordered


# --------------------------------------------------------------------------- cache (torch-free)
def cache_key(model_name: str, sentence: str, context_text: str, steps: int, max_tokens: int = MAX_TOKENS) -> str:
    """Digest of everything the result depends on. Changing the sentence, context or settings misses."""
    h = hashlib.sha256()
    for part in (METHOD, model_name, sentence, context_text, str(steps), str(max_tokens)):
        h.update(part.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


class AttributionCache:
    """Small thread-safe LRU. Keys are digests, so no page text is kept as a key."""

    def __init__(self, max_entries: int = 256):
        self.max_entries = max_entries
        self._data: "OrderedDict[str, object]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        return None

    def put(self, key: str, value: object) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


# --------------------------------------------------------------------------- model glue (torch)
@dataclass
class TokenExplanation:
    words: List[WordAttribution]
    n_tokens: int            # non-special tokens actually processed by the model (after truncation)
    n_tokens_total: int      # non-special tokens in the full sentence (== n_tokens unless truncated)
    truncated: bool
    steps: int
    f_input: float
    f_baseline: float
    sum_attributions: float
    completeness_error: float
    relative_error: Optional[float]
    converged: bool
    pipeline_deviation: Optional[float] = None  # |F(x) - cos(embedder.embed(sentence), c)|; None when not checked


_compute_lock = threading.Lock()  # one gradient computation at a time: bounds CPU/memory use


# Pooling modes are reported by sentence-transformers in different ways across releases (boolean
# ``pooling_mode_*`` attributes, a ``pooling_mode`` string/list, ``get_pooling_mode_str()``). We read all of
# them and normalise to a set of short names; only exactly {"mean"} is supported.
_BOOL_ATTR_TO_MODE = {
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_cls_token": "cls",
    "pooling_mode_max_tokens": "max",
    "pooling_mode_mean_sqrt_len_tokens": "mean_sqrt_len",
    "pooling_mode_weightedmean_tokens": "weightedmean",
    "pooling_mode_lasttoken": "lasttoken",
}
_ALIASES = {"mean_tokens": "mean", "cls_token": "cls", "max_tokens": "max", "mean_sqrt_len_tokens": "mean_sqrt_len",
            "weightedmean_tokens": "weightedmean", "last_token": "lasttoken"}


def _norm_mode(name: str) -> str:
    name = str(name).strip().lower()
    return _ALIASES.get(name, name)


def pooling_modes(pooling) -> Tuple[Optional[set], str]:
    """Return (set of active pooling modes or None if undeterminable, diagnostic text).

    Sources, in priority order: ``get_pooling_mode_str()``, a ``pooling_mode`` attribute (str or list),
    boolean ``pooling_mode_*`` attributes. The diagnostic lists everything that was inspected so an
    unsupported/unknown structure can be reported precisely instead of guessed at.
    """
    seen = {}
    modes: Optional[set] = None
    getter = getattr(pooling, "get_pooling_mode_str", None)
    if callable(getter):
        try:
            value = getter()
            seen["get_pooling_mode_str()"] = value
            if isinstance(value, str) and value.strip():
                modes = {_norm_mode(m) for m in value.split("+") if m.strip()}
        except Exception as exc:
            seen["get_pooling_mode_str()"] = f"<raised {type(exc).__name__}>"
    attr = getattr(pooling, "pooling_mode", None)
    if attr is not None:
        seen["pooling_mode"] = attr
        if modes is None:
            if isinstance(attr, str):
                modes = {_norm_mode(m) for m in attr.split("+") if m.strip()}
            elif isinstance(attr, (list, tuple, set)):
                modes = {_norm_mode(m) for m in attr}
    flags = {k: getattr(pooling, k) for k in _BOOL_ATTR_TO_MODE if hasattr(pooling, k)}
    if flags:
        seen.update(flags)
        if modes is None:
            modes = {m for k, m in _BOOL_ATTR_TO_MODE.items() if flags.get(k)}
    diag = f"{type(pooling).__name__}({', '.join(f'{k}={v!r}' for k, v in seen.items()) or 'no pooling attributes found'})"
    return modes, diag


def _components(embedder):
    """Pull tokenizer / HF model out of a SentenceTransformer-backed embedder, or explain why not.

    Supported structure: Transformer (HF encoder + fast tokenizer) -> mean Pooling -> optional Normalize.
    Anything else (Dense layers, CLS/max/weighted pooling, ...) would make our differentiable forward pass
    differ from the real embedding, so it is refused with a message naming what was found.
    """
    st = getattr(embedder, "st_model", None)
    if st is None:
        raise AttributionUnavailable(
            "unsupported_backend",
            "Token-level attribution needs the sentence-transformers embedding backend (gradients are not available for the lexical fallback).",
        )
    try:
        modules = list(st)
        transformer, pooling = modules[0], modules[1]
        tokenizer, model = transformer.tokenizer, transformer.auto_model
        max_len = int(getattr(transformer, "max_seq_length", None) or MAX_TOKENS)
    except Exception as exc:
        raise AttributionUnavailable("unsupported_model", f"Unsupported embedding model structure ({type(exc).__name__}).")

    modes, diag = pooling_modes(pooling)
    if modes is None:
        raise AttributionUnavailable("unsupported_model", f"Could not determine the pooling mode: {diag}.")
    if modes != {"mean"}:
        raise AttributionUnavailable("unsupported_model", f"Only mean pooling is supported; model uses {sorted(modes)} ({diag}).")
    extra = [type(m).__name__ for m in modules[2:]]
    if any(name != "Normalize" for name in extra):
        raise AttributionUnavailable(
            "unsupported_model",
            f"Unsupported layers after pooling: {extra}. Only an optional Normalize layer keeps the embedding equal to mean-pooled output.",
        )
    if not getattr(tokenizer, "is_fast", False):
        raise AttributionUnavailable("unsupported_model", "A fast tokenizer (with character offsets) is required.")
    if getattr(tokenizer, "pad_token_id", None) is None:
        raise AttributionUnavailable("unsupported_model", "The tokenizer has no [PAD] token to build the baseline.")
    return tokenizer, model, min(max_len, MAX_TOKENS)


PIPELINE_TOL = 0.01  # max |F(x) - cos(embedder.embed(sentence), c)| allowed before we refuse to report attributions


def explain(embedder, sentence: str, context_vector: np.ndarray, steps: int = DEFAULT_STEPS) -> TokenExplanation:
    """Run IG for ``sentence`` against ``context_vector`` (the study-context embedding from the pipeline)."""
    sentence = (sentence or "").strip()
    if not sentence:
        raise AttributionUnavailable("empty_input", "There is no text to explain.")
    steps = max(1, min(int(steps), MAX_STEPS))
    tokenizer, model, max_len = _components(embedder)
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on environment
        raise AttributionUnavailable("torch_missing", f"PyTorch is not available ({type(exc).__name__}).")

    ctx = np.asarray(context_vector, dtype=np.float64)
    norm = np.linalg.norm(ctx)
    if ctx.ndim != 1 or not np.isfinite(norm) or norm == 0:
        raise AttributionFailed("invalid study-context embedding")

    enc = tokenizer(
        sentence, truncation=True, max_length=max_len, return_offsets_mapping=True,
        return_special_tokens_mask=True, return_tensors="pt",
    )
    special = [bool(v) for v in enc["special_tokens_mask"][0].tolist()]
    offsets = [tuple(o) for o in enc["offset_mapping"][0].tolist()]
    if all(special):
        raise AttributionUnavailable("empty_input", "The text produced no analysable tokens.")
    # Count the full (untruncated) token length. ``verbose=False`` suppresses the Transformers
    # "sequence length is longer than the maximum" warning: this call only counts, nothing is fed to the model.
    n_total = len(tokenizer(sentence, add_special_tokens=False, truncation=False, verbose=False)["input_ids"])
    n_used = sum(1 for sp in special if not sp)
    truncated = n_total > n_used

    # Run on whatever device the model lives on (CPU, CUDA, MPS ...); nothing is hardcoded.
    try:
        device = next(model.parameters()).device
    except StopIteration:  # pragma: no cover - a model without parameters cannot be explained
        raise AttributionUnavailable("unsupported_model", "The model has no parameters.")
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    with _compute_lock:
        # Remember each submodule's own flag (not just the root's) so the exact state is restored.
        prior_modes = [(m, m.training) for m in model.modules()]
        model.eval()  # deterministic: no dropout in the attribution pass nor in the reference check
        try:
            emb_layer = model.get_input_embeddings()
            with torch.no_grad():
                x = emb_layer(ids)[0].detach()
                base = emb_layer(torch.full_like(ids, tokenizer.pad_token_id))[0].detach()
            keep = torch.tensor(special, dtype=torch.bool, device=device)
            base[keep] = x[keep]  # special tokens: x == baseline -> zero attribution by construction
            ctx_t = torch.as_tensor(ctx / norm, dtype=torch.float64).to(device=device, dtype=x.dtype)
            batch_mask = mask  # (1, L) on the model device

            def fn(points: np.ndarray):
                with torch.enable_grad():
                    # numpy (float64, CPU) -> model dtype -> model device; autograd starts at this leaf
                    p = torch.as_tensor(points).to(dtype=x.dtype).to(device).clone().requires_grad_(True)
                    out = model(inputs_embeds=p, attention_mask=batch_mask.expand(p.shape[0], -1)).last_hidden_state
                    m = batch_mask.unsqueeze(-1).to(out.dtype)
                    pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
                    vals = F.normalize(pooled, dim=-1) @ ctx_t
                    (grads,) = torch.autograd.grad(vals.sum(), p)
                # back to CPU *before* widening: MPS has no float64
                return vals.detach().cpu().double().numpy(), grads.detach().cpu().double().numpy()

            x_np, b_np = x.cpu().double().numpy(), base.cpu().double().numpy()
            result = integrated_gradients(x_np, b_np, fn, steps)
            if not result.converged and steps * 2 <= MAX_STEPS:
                result = integrated_gradients(x_np, b_np, fn, steps * 2)

            # The differentiable forward pass must reproduce the real embedding pipeline. Verified on this very
            # input, while the model is still in eval mode (the same mode the pipeline encodes in). Skipped when
            # truncated: the pipeline itself encodes up to max_seq_length, we cap at MAX_TOKENS.
            deviation = None
            embed = getattr(embedder, "embed", None)
            if callable(embed) and not truncated:
                ref = np.asarray(embed([sentence])[0], dtype=np.float64)
                ref_norm = np.linalg.norm(ref)
                if ref_norm > 0:
                    deviation = abs(float(np.dot(ref / ref_norm, ctx / norm)) - result.f_input)
                    if deviation > PIPELINE_TOL:
                        raise AttributionFailed(
                            f"differentiable forward pass deviates from the embedding pipeline by {deviation:.4f} (> {PIPELINE_TOL})"
                        )
        finally:
            for m, was_training in prior_modes:
                m.training = was_training

    tokens = tokenizer.convert_ids_to_tokens(ids[0].tolist())
    words = aggregate_words(sentence, tokens, offsets, special, result.token_attributions)
    return TokenExplanation(
        words=words, n_tokens=n_used, n_tokens_total=n_total, truncated=truncated, steps=result.steps,
        f_input=result.f_input, f_baseline=result.f_baseline, sum_attributions=result.sum_attributions,
        completeness_error=result.completeness_error, relative_error=result.relative_error, converged=result.converged,
        pipeline_deviation=deviation,
    )
