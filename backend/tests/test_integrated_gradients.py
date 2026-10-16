"""Integrated Gradients: API behaviour, real-torch path (tiny random BERT, no download) and the real MiniLM model.

Run:  uv run pytest tests/test_integrated_gradients.py -v
The torch/real-model tests skip (with a reason) when torch/transformers/the model weights are unavailable.
"""
import types

import numpy as np
import pytest

from app.ai import integrated_gradients as ig
from app.services import relevance_service


# ------------------------------------------------------------------ API (fake attribution, lexical engine)
def _fake_expl(sentence="Deadlocks stall processes"):
    words, pos = [], 0
    for i, w in enumerate(sentence.split()):
        start = sentence.index(w, pos)
        pos = start + len(w)
        words.append(ig.WordAttribution(word=w, start=start, end=pos, attribution=0.02 * (1 if i % 2 == 0 else -1),
                                        normalized=1.0 if i % 2 == 0 else -1.0, tokens=[(w.lower(), 0.02)]))
    return ig.TokenExplanation(words=words, n_tokens=len(words), n_tokens_total=len(words), truncated=False, steps=32, f_input=0.5, f_baseline=0.1,
                               sum_attributions=0.4, completeness_error=0.001, relative_error=0.0025, converged=True)


@pytest.fixture()
def fake_ig(monkeypatch):
    calls = []

    def fake_explain(embedder, sentence, ctx_vec, steps=32):
        calls.append((sentence, steps))
        return _fake_expl(sentence)

    monkeypatch.setattr(relevance_service, "_ig_supported", lambda e: True)
    monkeypatch.setattr(ig, "explain", fake_explain)
    relevance_service._get_ig_cache().clear()
    return calls


def _post(client, **body):
    return client.post("/api/relevance/explain-tokens", json=body)


def test_lexical_backend_reports_unavailable_not_error(client, session):
    r = _post(client, session_id=session["id"], sentence="Deadlocks stall processes")
    assert r.status_code == 200
    b = r.json()
    assert b["available"] is False and b["reason"] == "unsupported_backend" and b["words"] == []
    assert b["method"] == "integrated-gradients"


def test_explain_returns_signed_words_and_documents_scope(client, session, fake_ig):
    b = _post(client, session_id=session["id"], sentence="Deadlocks stall processes").json()
    assert b["available"] is True and b["cached"] is False
    assert [w["word"] for w in b["words"]] == ["Deadlocks", "stall", "processes"]
    assert b["words"][0]["attribution"] > 0 > b["words"][1]["attribution"]  # sign preserved
    assert all(-1 <= w["normalized"] <= 1 for w in b["words"])
    assert "keyword" in b["scope"] and "cosine" in b["target"] and b["converged"] is True
    assert "raw" in b["score_semantics"].lower()


def test_cache_hit_and_invalidation(client, session, fake_ig, make_user):
    s = "Deadlocks stall processes"
    assert _post(client, session_id=session["id"], sentence=s).json()["cached"] is False
    assert _post(client, session_id=session["id"], sentence=s).json()["cached"] is True
    assert len(fake_ig) == 1                                              # second call did not recompute
    assert _post(client, session_id=session["id"], sentence=s + " too").json()["cached"] is False  # new sentence
    assert _post(client, session_id=session["id"], sentence=s, steps=16).json()["cached"] is False  # new settings
    # a different study context must not reuse the entry
    assert _post(client, topic="Cooking pasta", sentence=s).json()["cached"] is False
    assert _post(client, topic="Cooking pasta", sentence=s).json()["cached"] is True
    assert _post(client, topic="Cooking risotto", sentence=s).json()["cached"] is False


def test_validation_and_errors(client, session, fake_ig):
    assert _post(client, sentence="x y").status_code == 422                       # no session / topic
    assert _post(client, topic="a", sentence="   ").status_code == 422             # blank sentence
    assert _post(client, topic="a", sentence="x" * 2001).status_code == 422        # too long
    assert _post(client, topic="a", sentence="x", steps=1000).status_code == 422   # unbounded cost
    assert _post(client, session_id="nope", sentence="x y").status_code == 404     # unknown session
    assert fake_ig == []


def test_requires_login_and_ownership(client, anon, session, fake_ig, make_user):
    assert anon.post("/api/relevance/explain-tokens", json={"topic": "a", "sentence": "x y"}).status_code == 401
    _, other = make_user("STUDENT")
    r = client.post("/api/relevance/explain-tokens", json={"session_id": session["id"], "sentence": "x y"}, headers=other)
    assert r.status_code == 404  # someone else's session looks like a missing one


def test_attribution_failure_is_reported_not_raised(client, session, monkeypatch):
    monkeypatch.setattr(relevance_service, "_ig_supported", lambda e: True)
    relevance_service._get_ig_cache().clear()

    def boom(*a, **k):
        raise RuntimeError("secret page text must not leak")

    monkeypatch.setattr(ig, "explain", boom)
    r = _post(client, session_id=session["id"], sentence="Deadlocks stall processes")
    assert r.status_code == 200 and r.json()["available"] is False and r.json()["reason"] == "attribution_failed"
    assert "secret" not in r.text


def test_unavailable_reason_is_passed_through(client, session, monkeypatch):
    monkeypatch.setattr(relevance_service, "_ig_supported", lambda e: True)
    monkeypatch.setattr(ig, "explain", lambda *a, **k: (_ for _ in ()).throw(ig.AttributionUnavailable("unsupported_model", "nope")))
    b = _post(client, session_id=session["id"], sentence="Deadlocks stall processes").json()
    assert b["available"] is False and b["reason"] == "unsupported_model"


def test_analyze_contract_is_unchanged(client, session):
    """The existing endpoint's response keys and explanation method must not change."""
    b = client.post("/api/relevance/analyze", json={
        "session_id": session["id"], "url": "https://x.org/a", "title": "Deadlock prevention",
        "text": "Deadlock prevention breaks one of the four Coffman conditions. The Banker's algorithm avoids deadlock.",
    }).json()
    from app.schemas.relevance import AnalyzeResponse
    assert set(b) == set(AnalyzeResponse.model_fields)
    assert b["explanation_method"] == "sentence-level semantic similarity"
    assert "words" not in b and "attribution" not in str(b.keys())


# ------------------------------------------------------------------ real torch path with a tiny random BERT
TINY_VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "dead", "##lock", "##s", "stall", "process", "##es", "the",
              "banker", "'", "s", "algorithm", "avoid", "cook", "pasta", ",", ".", "!", "un", "##happy"]


def _build_tiny_tokenizer(tf):
    """WordPiece tokenizer built directly with the `tokenizers` library, so the vocabulary is guaranteed to be
    used regardless of how a given transformers release treats `vocab_file=`."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

    ids = {t: i for i, t in enumerate(TINY_VOCAB)}
    tk = Tokenizer(models.WordPiece(ids, unk_token="[UNK]", max_input_chars_per_word=100))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", ids["[CLS]"]), ("[SEP]", ids["[SEP]"])],
    )
    tok = tf.PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                     sep_token="[SEP]", mask_token="[MASK]")
    # Fixture sanity check: the test below only exercises sub-word aggregation if this holds.
    assert tok.tokenize("deadlocks") == ["dead", "##lock", "##s"], tok.tokenize("deadlocks")
    assert tok.tokenize("unhappy") == ["un", "##happy"], tok.tokenize("unhappy")
    return tok


@pytest.fixture(scope="module")
def tiny_embedder():
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    tok = _build_tiny_tokenizer(tf)
    torch.manual_seed(0)
    cfg = tf.BertConfig(vocab_size=len(TINY_VOCAB), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=64)
    model = tf.BertModel(cfg).eval()
    transformer = types.SimpleNamespace(tokenizer=tok, auto_model=model, max_seq_length=16)
    pooling = types.SimpleNamespace(pooling_mode_mean_tokens=True)

    def embed(texts):  # the "existing pipeline" for this toy model: mean pooling + L2 normalisation, no gradients
        out = []
        for t in texts:
            enc = tok(t, truncation=True, max_length=16, return_tensors="pt")
            with torch.no_grad():
                h = model(**enc).last_hidden_state[0].mean(0)
            out.append(torch.nn.functional.normalize(h, dim=0).numpy())
        return np.asarray(out)

    return types.SimpleNamespace(st_model=[transformer, pooling], name="tiny-bert", tokenizer=tok, model=model, torch=torch, embed=embed)


def _ctx(tiny, text):
    return tiny.embed([text])[0]


def _expected_tokens(tokenizer, text, max_length):
    """What the model must have processed: the tokenizer's own truncated output without special tokens."""
    enc = tokenizer(text, truncation=True, max_length=max_length)
    toks = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    return toks[1:-1]


def _assert_aligned(e, tokenizer, text, max_length):
    """Token/attribution arrays are aligned and trailing words are not invented."""
    expected = _expected_tokens(tokenizer, text, max_length)
    got = [tok for w in e.words for tok, _ in w.tokens]
    assert got == expected                                   # same tokens, same order, nothing extra or missing
    assert e.n_tokens == len(expected)                       # token count == what the model processed
    assert all(np.isfinite(a) for w in e.words for _, a in w.tokens)
    assert sum(w.attribution for w in e.words) == pytest.approx(sum(a for w in e.words for _, a in w.tokens), abs=1e-9)
    assert sum(w.attribution for w in e.words) == pytest.approx(e.sum_attributions, abs=1e-5)   # special tokens carry 0
    ends = [w.end for w in e.words]
    assert ends == sorted(ends) and all(w.start < w.end for w in e.words)
    # every word in the output is literally a word of the input, in order, starting from the first one
    input_words = [m.group(0) for m in ig._WORD_RE.finditer(text)]
    assert [w.word for w in e.words] == input_words[: len(e.words)]
    return input_words


def test_tiny_bert_ig_end_to_end(tiny_embedder):
    t = tiny_embedder
    ctx = _ctx(t, "banker algorithm avoid deadlocks")
    sentence = "The banker's algorithm avoid deadlocks, stall processes!"
    e = ig.explain(t, sentence, ctx, steps=32)
    assert e.words and all(np.isfinite(w.attribution) and np.isfinite(w.normalized) for w in e.words)
    assert [w.word for w in e.words][:3] == ["The", "banker's", "algorithm"]
    # completeness: the attributions account for F(x) - F(baseline)
    assert e.converged and e.completeness_error <= max(0.05 * abs(e.f_input - e.f_baseline), 0.002)
    assert abs(sum(w.attribution for w in e.words) - e.sum_attributions) < 1e-5  # special tokens carry exactly 0
    # F(x) must equal the ordinary (non-IG) cosine of the same model; the runtime guard measured it too
    assert e.f_input == pytest.approx(float(np.dot(t.embed([sentence])[0], ctx)), abs=1e-4)
    assert e.pipeline_deviation is not None and e.pipeline_deviation < 1e-4
    assert any(abs(w.attribution) > 0 for w in e.words)  # not a degenerate all-zero result
    _assert_aligned(e, t.tokenizer, sentence, 16)
    # punctuation is its own word and keeps its own (signed) score
    assert [w.word for w in e.words] == ["The", "banker's", "algorithm", "avoid", "deadlocks", ",", "stall", "processes", "!"]


def test_tiny_bert_subword_merge_and_truncation(tiny_embedder):
    t = tiny_embedder
    ctx = _ctx(t, "banker algorithm")
    e = ig.explain(t, "deadlocks unhappy", ctx, steps=16)
    assert [w.word for w in e.words] == ["deadlocks", "unhappy"]
    assert [tok for tok, _ in e.words[0].tokens] == ["dead", "##lock", "##s"]
    assert [tok for tok, _ in e.words[1].tokens] == ["un", "##happy"]
    # a word's score is exactly the sum of its sub-word scores (sign preserved)
    for w in e.words:
        assert w.attribution == pytest.approx(sum(a for _, a in w.tokens), abs=1e-9)

    # long input: distinct words so an off-by-one or a falsely attributed trailing word would be visible
    cycle = ["stall", "pasta", "cook", "banker", "avoid", "algorithm", "process"]
    long = " ".join(cycle[i % len(cycle)] for i in range(100))
    e2 = ig.explain(t, long, ctx, steps=8)
    assert e2.truncated is True
    assert e2.n_tokens == 14 and e2.n_tokens_total > 14      # max_seq_length 16 minus [CLS]/[SEP]
    input_words = _assert_aligned(e2, t.tokenizer, long, 16)
    assert len(e2.words) == 14 < len(input_words)            # tail was cut, and nothing beyond it is reported
    assert e2.words[-1].end <= len(" ".join(input_words[:14]))


def test_tiny_bert_punctuation_and_word_sums_with_real_gradients(tiny_embedder):
    t = tiny_embedder
    e = ig.explain(t, "the banker's algorithm avoid deadlocks, stall processes!", _ctx(t, "banker algorithm avoid deadlocks"), steps=32)
    for w in e.words:  # every word score is exactly the sum of its sub-word scores, whatever the sign
        assert w.attribution == pytest.approx(sum(a for _, a in w.tokens), abs=1e-9)
        assert np.sign(w.normalized) == np.sign(w.attribution) or w.attribution == 0
    for mark in (",", "!"):
        item = [w for w in e.words if w.word == mark]
        assert len(item) == 1 and len(item[0].tokens) == 1 and item[0].attribution == pytest.approx(item[0].tokens[0][1])
    # sign preservation across aggregation itself is covered deterministically in tests/test_ig_core.py


def test_tiny_bert_short_and_empty_inputs(tiny_embedder):
    t = tiny_embedder
    ctx = _ctx(t, "banker")
    assert len(ig.explain(t, "pasta", ctx, steps=8).words) == 1
    with pytest.raises(ig.AttributionUnavailable) as e:
        ig.explain(t, "   ", ctx)
    assert e.value.reason == "empty_input"
    with pytest.raises(ig.AttributionFailed):
        ig.explain(t, "pasta", np.zeros_like(ctx))  # zero context vector is invalid


def test_forward_pass_deviation_is_refused(tiny_embedder, monkeypatch):
    """If the pipeline's embedding and our differentiable forward pass disagree, no attribution is reported."""
    t = tiny_embedder
    monkeypatch.setattr(ig, "PIPELINE_TOL", 1e-4)
    wrong = types.SimpleNamespace(st_model=t.st_model, name="x", embed=lambda texts: -t.embed(texts))  # opposite embedding
    with pytest.raises(ig.AttributionFailed, match="deviates"):
        ig.explain(wrong, "pasta cook", _ctx(t, "banker"), steps=8)
    ig.explain(t, "pasta cook", _ctx(t, "banker"), steps=8)  # the faithful embedder passes the same tolerance


def test_model_eval_state_restored(tiny_embedder):
    t = tiny_embedder
    t.model.train()
    ig.explain(t, "pasta", _ctx(t, "banker"), steps=8)
    assert t.model.training is True
    t.model.eval()


# ------------------------------------------------------------------ real all-MiniLM-L6-v2
@pytest.fixture(scope="module")
def minilm():
    pytest.importorskip("sentence_transformers")
    from app.ai.embeddings import SentenceTransformerEmbedder
    try:
        return SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as exc:  # offline / no cached weights
        pytest.skip(f"all-MiniLM-L6-v2 could not be loaded: {type(exc).__name__}")


def test_real_minilm_module_structure_is_supported(minilm):
    """Documents what the installed model actually looks like and that our detection reads it correctly."""
    st = minilm.st_model
    modules = list(st)
    names = [type(m).__name__ for m in modules]
    modes, diag = ig.pooling_modes(modules[1])
    print("\nMiniLM modules:", names, "| pooling:", diag, "| max_seq_length:", getattr(modules[0], "max_seq_length", None))
    assert modes == {"mean"}, diag
    assert all(n == "Normalize" for n in names[2:])
    tokenizer, model, max_len = ig._components(minilm)   # must not raise
    assert tokenizer.is_fast and max_len == min(ig.MAX_TOKENS, getattr(modules[0], "max_seq_length", ig.MAX_TOKENS))


def test_real_minilm_integrated_gradients(minilm):
    context = "Deadlocks. Understand deadlock prevention and the Banker's algorithm"
    sentence = "The Banker's algorithm avoids deadlock by simulating resource allocation before granting a request."
    ctx = minilm.embed([context])[0]
    e = ig.explain(minilm, sentence, ctx, steps=32)

    assert e.words and all(np.isfinite(w.attribution) for w in e.words)
    assert e.converged, f"completeness error {e.completeness_error} vs diff {e.f_input - e.f_baseline}"
    # the explained target is the same number the existing pipeline computes (tokenisation, mask, pooling, normalisation)
    assert e.f_input == pytest.approx(float(np.dot(minilm.embed([sentence])[0], ctx)), abs=2e-3)
    assert e.pipeline_deviation is not None and e.pipeline_deviation < 2e-3
    assert abs(sum(w.attribution for w in e.words) - (e.f_input - e.f_baseline)) <= max(0.05 * abs(e.f_input - e.f_baseline), 0.002)
    assert e.n_tokens >= len(e.words) and not e.truncated and e.n_tokens == e.n_tokens_total
    assert max(abs(w.normalized) for w in e.words) == pytest.approx(1.0)
    tokenizer = ig._components(minilm)[0]
    _assert_aligned(e, tokenizer, sentence, ig.MAX_TOKENS)
    print("\nMiniLM IG:", [(w.word, round(w.attribution, 4)) for w in e.words], "F=", round(e.f_input, 4), "F0=", round(e.f_baseline, 4))


def test_real_minilm_long_input_is_truncated(minilm):
    ctx = minilm.embed(["Deadlocks"])[0]
    cycle = ["deadlock", "prevention", "banker", "algorithm", "resource", "allocation", "process", "mutual", "exclusion", "circular"]
    long = " ".join(cycle[i % len(cycle)] for i in range(400))
    tokenizer, _, max_len = ig._components(minilm)
    e = ig.explain(minilm, long, ctx, steps=8)

    expected = _expected_tokens(tokenizer, long, max_len)
    assert e.truncated and len(expected) == max_len - 2          # the cap really bound: 126 content tokens
    assert e.n_tokens == len(expected) and e.n_tokens_total > e.n_tokens
    _assert_aligned(e, tokenizer, long, max_len)                 # tokens, words, offsets and sums are consistent
    assert len(e.words) < 400                                    # trailing words are not attributed
    assert e.pipeline_deviation is None                          # check is skipped when truncated, and says so
