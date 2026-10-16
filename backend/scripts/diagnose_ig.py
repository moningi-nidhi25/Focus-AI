"""Print what the installed sentence-transformers model actually looks like and whether Integrated Gradients
can run on it. Paste the output when reporting a problem.

    uv run python scripts/diagnose_ig.py
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 uv run python scripts/diagnose_ig.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.ai import integrated_gradients as ig  # noqa: E402


def main() -> int:
    import importlib.metadata as md

    for pkg in ("sentence-transformers", "transformers", "tokenizers", "torch", "numpy"):
        try:
            print(f"{pkg:24s} {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"{pkg:24s} NOT INSTALLED")

    from app.ai.embeddings import SentenceTransformerEmbedder

    name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    emb = SentenceTransformerEmbedder(name)
    st = emb.st_model
    print("\nembedder.st_model:", type(st).__module__ + "." + type(st).__name__)
    for i, m in enumerate(st):
        print(f"  st[{i}] = {type(m).__module__}.{type(m).__name__}")
    transformer, pooling = st[0], st[1]
    tok, model = transformer.tokenizer, transformer.auto_model
    print("\nst[0].max_seq_length     :", getattr(transformer, "max_seq_length", None))
    print("st[0].tokenizer          :", type(tok).__name__, "| is_fast =", getattr(tok, "is_fast", None), "| pad_token_id =", tok.pad_token_id)
    print("st[0].auto_model         :", type(model).__name__, "| training =", model.training)
    print("\nst[1] public attributes  :", {k: v for k, v in vars(pooling).items() if not k.startswith("_") and not hasattr(v, "parameters")})
    print("st[1] pooling_mode_*     :", {k: getattr(pooling, k) for k in dir(pooling) if k.startswith("pooling_mode")})
    print("st[1].get_pooling_mode_str:", pooling.get_pooling_mode_str() if hasattr(pooling, "get_pooling_mode_str") else "(absent)")
    print("pooling_modes() result   :", ig.pooling_modes(pooling))

    print("\ntokenizer.tokenize('deadlocks unhappiness') =", tok.tokenize("deadlocks unhappiness"))
    try:
        parts = ig._components(emb)
        print("\n_components(): OK (max_len =", parts[2], ")")
    except ig.AttributionUnavailable as exc:
        print("\n_components(): UNAVAILABLE ->", exc.reason, "|", exc.message)
        return 1

    ctx = emb.embed(["Deadlocks. Understand deadlock prevention and the Banker's algorithm"])[0]
    sentence = "The Banker's algorithm avoids deadlock by simulating resource allocation."
    e = ig.explain(emb, sentence, ctx, steps=32)
    ref = float(np.dot(emb.embed([sentence])[0], ctx))
    print(f"\nF(x) from IG forward pass      : {e.f_input:.6f}")
    print(f"cos(embed(sentence), context)   : {ref:.6f}   (deviation {abs(ref - e.f_input):.2e})")
    print(f"F(baseline)                     : {e.f_baseline:.6f}")
    print(f"sum(attributions)               : {e.sum_attributions:.6f}  (completeness error {e.completeness_error:.2e}, converged={e.converged}, steps={e.steps})")
    print("tokens processed / total        :", e.n_tokens, "/", e.n_tokens_total, "| truncated =", e.truncated)
    for w in e.words:
        print(f"  {w.word:14s} {w.attribution:+.5f}   {[t for t, _ in w.tokens]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
