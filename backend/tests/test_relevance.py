"""Phases 4-6: relevance engine, filtering decisions, explanations and heatmaps."""
from app.ai.context import build_profile
from app.ai.embeddings import LexicalEmbedder
from app.ai.llm import LLMJudgement
from app.ai.relevance_engine import ContentInput, EngineConfig, RelevanceEngine

RELEVANT_TEXT = (
    "The Banker's algorithm is a resource allocation and deadlock avoidance algorithm. It tests for safety by "
    "simulating the allocation of the maximum possible amounts of all resources before deciding whether an "
    "allocation should continue. Deadlock prevention works by breaking one of the four Coffman conditions: "
    "mutual exclusion, hold and wait, no preemption, or circular wait."
)
DISTRACTION_TEXT = (
    "This season's most talked about celebrity weddings featured stunning gowns from famous designers. "
    "Fans shared their favourite red carpet looks on social media. See the full gallery of the best moments."
)


def _analyze(client, session, **kw):
    return client.post("/api/relevance/analyze", json={"session_id": session["id"], **kw})


def test_relevant_content_is_allowed_with_explanation_and_heatmap(client, session):
    r = _analyze(client, session, url="https://en.wikipedia.org/wiki/Banker%27s_algorithm", title="Banker's algorithm - Wikipedia", text=RELEVANT_TEXT)
    assert r.status_code == 200
    b = r.json()
    assert b["decision"] == "ALLOW" and b["classification"] == "RELEVANT" and b["relevance_score"] >= 55
    assert b["explanation"].startswith("Allowed") and "relevant" in b["explanation"]
    assert any("banker" in c for c in b["matching_concepts"])
    assert b["heatmap"] and all({"text", "score", "level"} <= set(h) for h in b["heatmap"])
    assert b["highlighted_sentences"] and b["highlighted_sentences"][0]["level"] in ("high", "medium")
    assert b["explanation_method"] == "sentence-level semantic similarity"
    assert b["analysis_id"]


def test_distraction_is_blocked_and_logged(client, session):
    r = _analyze(client, session, url="https://example.com/celebs", title="Top 10 celebrity wedding dresses", text=DISTRACTION_TEXT)
    b = r.json()
    assert b["decision"] == "BLOCK" and b["classification"] == "DISTRACTING" and b["relevance_score"] < 30
    assert b["explanation"].startswith("Blocked")

    hist = client.get(f"/api/relevance/sessions/{session['id']}/history").json()
    assert hist[0]["decision"] == "BLOCK" and hist[0]["url"] == "https://example.com/celebs"

    from app.models import ActivityEvent, BlockedContent
    from app.utils.database import SessionLocal

    with SessionLocal() as db:
        assert db.query(BlockedContent).filter_by(session_id=session["id"]).count() == 1
        assert db.query(ActivityEvent).filter_by(session_id=session["id"], event_type="content_blocked").count() == 1


def test_override_is_recorded_as_distraction_attempt(client, session):
    r = client.post("/api/relevance/override", json={"session_id": session["id"], "url": "https://example.com/celebs"})
    assert r.status_code == 204
    from app.models import ActivityEvent
    from app.utils.database import SessionLocal

    with SessionLocal() as db:
        assert db.query(ActivityEvent).filter_by(session_id=session["id"], event_type="distraction_attempt").count() == 1


def test_youtube_is_not_blanket_blocked(client, session, monkeypatch):
    """Same domain, different videos -> different decisions."""
    from app.services import content_extractor as ce

    monkeypatch.setattr(ce, "_fetch_transcript", lambda vid: RELEVANT_TEXT if vid == "good" else DISTRACTION_TEXT)
    monkeypatch.setattr(ce, "_youtube_page_meta", lambda vid: {"title": "", "description": ""})
    titles = {"good": "Deadlock prevention and Banker's algorithm", "bad": "Celebrity wedding gowns"}

    class R:
        status_code = 200
        def __init__(self, vid): self.vid = vid
        def json(self): return {"title": titles[self.vid]}

    monkeypatch.setattr(ce.requests, "get", lambda url, **k: R(k["params"]["url"].split("v=")[1]))
    good = _analyze(client, session, url="https://www.youtube.com/watch?v=good").json()
    bad = _analyze(client, session, url="https://www.youtube.com/watch?v=bad").json()
    assert good["decision"] == "ALLOW" and bad["decision"] == "BLOCK"


def test_empty_page_is_not_blocked(client, session):
    # nothing to judge -> never block
    from app.ai.relevance_engine import get_engine

    res = get_engine().analyze(build_profile("Deadlocks"), ContentInput(title="", text=""))
    assert res.decision == "ALLOW" and res.insufficient_content


def test_title_only_unrelated_page_blocks_only_when_clearly_unrelated(client, session):
    b = _analyze(client, session, url="https://x.com/a", title="Celebrity wedding dresses", text="Gallery").json()
    assert b["low_content"] is True and b["decision"] == "BLOCK"
    b = _analyze(client, session, url="https://x.com/b", title="Home", text="Menu").json()
    assert b["decision"] == "ALLOW" and b["insufficient_content"] is True  # nothing meaningful to judge


def test_stateless_preview_with_topic(client):
    text = (
        "In photosynthesis, chlorophyll absorbs light in the chloroplast. The light-dependent reactions make ATP and NADPH. "
        "The Calvin cycle then uses them to fix carbon dioxide into glucose in the stroma of the chloroplast."
    )
    r = client.post("/api/relevance/analyze", json={"topic": "Photosynthesis", "learning_objectives": "Calvin cycle", "text": text})
    assert r.status_code == 200 and r.json()["analysis_id"] is None and r.json()["decision"] == "ALLOW"


def test_validation_and_errors(client):
    assert client.post("/api/relevance/analyze", json={"text": "x"}).status_code == 422  # no session/topic
    assert client.post("/api/relevance/analyze", json={"topic": "x"}).status_code == 422  # no content
    assert client.post("/api/relevance/analyze", json={"session_id": "nope", "text": "x y z"}).status_code == 404
    r = client.post("/api/relevance/analyze", json={"topic": "x", "url": "http://127.0.0.1/"})
    assert r.status_code == 422


def test_engine_info(client):
    b = client.get("/api/relevance/engine").json()
    assert b["embedding_backend"].startswith("lexical") and b["llm_enabled"] is False and b["allow_threshold"] == 75


