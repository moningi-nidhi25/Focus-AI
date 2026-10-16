"""Evaluate the relevance engine on the labelled examples in scripts/data/relevance_eval.json.

    uv run python scripts/eval_relevance.py            # uses EMBEDDING_BACKEND from .env (auto)
    EMBEDDING_BACKEND=lexical uv run python scripts/eval_relevance.py

Prints per-example scores, accuracy, precision/recall/F1 per class, and the roadmap's
"false block" (relevant/partial content blocked) and "false allow" (distraction allowed) counts.
Use it to re-tune thresholds / calibration whenever the embedding model changes.
"""
import argparse
import json
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(dotenv_path=backend_dir / ".env")  # same approach as main.py: LLM_PROVIDER / LLM_API_KEY / EMBEDDING_* apply

from app.ai.context import build_profile  # noqa: E402
from app.ai.relevance_engine import ContentInput, get_engine  # noqa: E402

LABELS = ["RELEVANT", "PARTIALLY_RELEVANT", "DISTRACTING"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="relevance_eval.json",
                    help="dataset in scripts/data/ (relevance_eval_lowcontent.json = title/description-only pages, e.g. videos without transcript)")
    data = json.loads((Path(__file__).parent / "data" / ap.parse_args().file).read_text())
    engine = get_engine()
    print(f"Embedding backend: {engine.embedder.name} | LLM: {'on' if engine.llm else 'off'}\n")
    rows = []
    for group in data:
        profile = build_profile(group["context"]["topic"], group["context"].get("objectives", ""))
        print(f"== {profile.topic}")
        for ex in group["examples"]:
            r = engine.analyze(profile, ContentInput(title=ex["title"], text=ex.get("text", ""), description=ex.get("description", "")))
            rows.append((ex["label"], r.classification))
            mark = "ok " if ex["label"] == r.classification else "XX "
            print(
    f"  {mark}{r.score:5.1f}  true={ex['label']:<18} "
    f"pred={r.classification:<18} "
    f"llm_score={r.signals.get('llm_score')} "
    f"{ex['title'][:48]}"
    )
    n = len(rows)
    print(f"\nAccuracy: {sum(t == p for t, p in rows)}/{n} = {sum(t == p for t, p in rows) / n:.0%}")
    print(f"{'class':<20}{'precision':>10}{'recall':>9}{'F1':>7}")
    for lab in LABELS:
        tp = sum(1 for t, p in rows if t == p == lab)
        fp = sum(1 for t, p in rows if p == lab and t != lab)
        fn = sum(1 for t, p in rows if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{lab:<20}{prec:>10.2f}{rec:>9.2f}{f1:>7.2f}")
    false_block = sum(1 for t, p in rows if p == "DISTRACTING" and t != "DISTRACTING")
    false_allow = sum(1 for t, p in rows if p == "RELEVANT" and t == "DISTRACTING")
    print(f"False blocks (useful content blocked): {false_block} | False allows (distraction allowed): {false_allow}")


if __name__ == "__main__":
    main()
