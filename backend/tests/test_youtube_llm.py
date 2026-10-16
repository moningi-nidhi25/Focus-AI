"""YouTube extraction, low-content handling and the structured LLM reviewer.

Offline: the three YouTube network fetches and the LLM HTTP call are replaced by fakes.
"""
import time

import httpx
import pytest

from app.ai.context import build_profile
from app.ai.embeddings import LexicalEmbedder
from app.ai.llm import LLMJudgement, LLMOutcome, LLMReasoner, LLMReviewInput, LLMValidationError, validate_judgement
from app.ai.relevance_engine import ContentInput, EngineConfig, RelevanceEngine
from app.services import content_extractor as ce

PROFILE = build_profile("Operating System Deadlocks", "Understand deadlock prevention, detection, and Banker's algorithm.")
RELEVANT_DESC = (
    "In this lecture we cover operating system deadlocks: the four Coffman conditions, deadlock prevention, "
    "deadlock detection with a resource allocation graph, and the Banker's algorithm for deadlock avoidance."
)
RELEVANT_TRANSCRIPT = (
    "A deadlock happens when processes wait forever for resources held by each other. Deadlock prevention breaks one "
    "of the Coffman conditions: mutual exclusion, hold and wait, no preemption or circular wait. The Banker's algorithm "
    "checks whether granting a resource request leaves the system in a safe state. Deadlock detection builds a "
    "resource allocation graph and looks for cycles."
)
ESPORTS_TITLE = "Squad wins the grand final | Rookie one man army | Mobile esports semi finals day 1"
ESPORTS_DESC = "Highlights from the semi finals of the mobile esports tournament. Best clutch plays, squad kills and the winning team."


# ----------------------------------------------------------------------------------- fakes
class FakeLLM:
    """Stands in for LLMReasoner: returns a fixed verdict or a failure status."""

    def __init__(self, score=None, status="ok", cls=None, concepts=("deadlock",)):
        self.score, self.status, self.cls, self.concepts = score, status, cls, list(concepts)
        self.calls = []

    def review(self, inp: LLMReviewInput) -> LLMOutcome:
        self.calls.append(inp)
        if self.status != "ok":
            return LLMOutcome(self.status, detail="boom")
        cls = self.cls or ("RELEVANT" if self.score >= 75 else "PARTIALLY_RELEVANT" if self.score >= 30 else "DISTRACTING")
        return LLMOutcome("ok", LLMJudgement(cls, self.score, "Contextual review.", self.concepts))


def engine(llm=None, **cfg):
    return RelevanceEngine(LexicalEmbedder(), llm, EngineConfig(**cfg))


def patch_youtube(monkeypatch, title="", description="", transcript=None, keywords=(), channel="", category="", oembed=True):
    """Fake the three network fetches. transcript=None -> captions unavailable."""
    def no_transcript(vid):
        raise RuntimeError("TranscriptsDisabled")

    monkeypatch.setattr(ce, "_fetch_transcript", (lambda vid: transcript) if transcript is not None else no_transcript)
    monkeypatch.setattr(ce, "_fetch_oembed_title", (lambda vid: title) if oembed else (lambda vid: (_ for _ in ()).throw(RuntimeError("down"))))
    monkeypatch.setattr(ce, "_youtube_page_meta", lambda vid: {
        "title": "", "description": description, "keywords": list(keywords), "channel": channel, "category": category,
    })


def analyze(client, session, vid, **kw):
    return client.post("/api/relevance/analyze", json={"session_id": session["id"], "url": f"https://www.youtube.com/watch?v={vid}", **kw})


# --------------------------------------------------------------------- text normalisation
def test_normalize_video_text_removes_noise():
    raw = "Watch at 12:30! https://example.com/x?y=1 follow me@mail.com [Music] >> Hello!!!!!! world"
    out = ce.normalize_video_text(raw)
    assert out == "Watch at ! follow Hello! world"


def test_generic_video_titles_detected():
    assert all(ce.is_generic_video_title(t) for t in ("", "YouTube", "(3) YouTube", "- YouTube"))
    assert not ce.is_generic_video_title("Deadlocks explained")


# ------------------------------------------------------------------ YouTube via the API
def test_youtube_relevant_with_transcript_is_allowed(client, session, monkeypatch):
    patch_youtube(monkeypatch, "Deadlock prevention and Banker's algorithm", RELEVANT_DESC, RELEVANT_TRANSCRIPT, ["operating systems"], "CS Lectures", "Education")
    b = analyze(client, session, "relevantA").json()
    assert b["decision"] == "ALLOW" and b["classification"] == "RELEVANT"
    assert b["content_sources"]["transcript"] is True and b["content_sources"]["transcript_status"] == "ok"
    assert b["evaluation_path"] == "embedding" and b["content_quality"] == "sufficient"


def test_youtube_relevant_title_and_description_only_is_not_blocked(client, session, monkeypatch):
    patch_youtube(monkeypatch, "Deadlock prevention and Banker's algorithm", RELEVANT_DESC, transcript=None)
    b = analyze(client, session, "relevantB").json()
    assert b["decision"] in ("ALLOW", "WARN") and b["relevance_score"] >= 30  # transcript failure never blocks
    assert b["content_sources"]["transcript"] is False and b["content_sources"]["transcript_status"].startswith("unavailable")
    assert b["content_sources"]["description"] is True


def test_youtube_irrelevant_without_transcript_is_blocked_not_50_allow(client, session, monkeypatch):
    patch_youtube(monkeypatch, ESPORTS_TITLE, ESPORTS_DESC, transcript=None, category="Gaming")
    b = analyze(client, session, "gameA").json()
    assert b["decision"] == "BLOCK" and b["classification"] == "DISTRACTING"
    assert b["relevance_score"] < 30 and b["relevance_score"] != 50
    assert b["insufficient_content"] is False and b["evaluation_path"] != "insufficient"


def test_youtube_irrelevant_title_only_is_blocked(client, session, monkeypatch):
    patch_youtube(monkeypatch, ESPORTS_TITLE, "", transcript=None)
    b = analyze(client, session, "gameB").json()
    assert b["decision"] == "BLOCK" and b["content_quality"] == "low"


def test_youtube_generic_client_title_is_replaced_by_authoritative_title(client, session, monkeypatch):
    """The SPA may still show 'YouTube' as the tab title; the server title for the video id wins."""
    patch_youtube(monkeypatch, ESPORTS_TITLE, ESPORTS_DESC, transcript=None)
    b = analyze(client, session, "gameC", title="YouTube").json()
    assert b["title"] == ESPORTS_TITLE and b["decision"] == "BLOCK"


def test_youtube_uses_extension_metadata_when_server_fetch_fails(client, session, monkeypatch):
    patch_youtube(monkeypatch, "", "", transcript=None, oembed=False)
    b = analyze(client, session, "gameD", title=ESPORTS_TITLE, metadata={"description": ESPORTS_DESC, "channel": "Esports Daily", "keywords": ["esports"]}).json()
    assert b["decision"] == "BLOCK" and b["content_sources"]["description"] is True


def test_youtube_nothing_readable_fails_open_neutral(client, session, monkeypatch):
    patch_youtube(monkeypatch, "", "", transcript=None, oembed=False)
    r = analyze(client, session, "emptyA", title="YouTube")
    b = r.json()
    assert r.status_code == 200 and b["decision"] == "ALLOW" and b["relevance_score"] == 50
    assert b["insufficient_content"] is True and b["evaluation_path"] == "insufficient"


def test_youtube_transcript_timeout_does_not_hang_or_fail(client, session, monkeypatch):
    patch_youtube(monkeypatch, ESPORTS_TITLE, ESPORTS_DESC, transcript="x")
    monkeypatch.setattr(ce, "_fetch_transcript", lambda vid: time.sleep(2) or "late")
    monkeypatch.setattr(ce, "YT_FETCH_TIMEOUT", 0.3)
    start = time.time()
    b = analyze(client, session, "slowA").json()
    assert time.time() - start < 1.5
    assert b["content_sources"]["transcript_status"] == "timeout" and b["decision"] == "BLOCK"


def test_youtube_parts_are_cached_per_video(monkeypatch):
    calls = []
    monkeypatch.setattr(ce, "_fetch_transcript", lambda vid: calls.append(vid) or "deadlock transcript text here")
    monkeypatch.setattr(ce, "_fetch_oembed_title", lambda vid: "Title")
    monkeypatch.setattr(ce, "_youtube_page_meta", lambda vid: {"description": "d"})
    ce.extract_youtube_content("cacheA", "https://youtu.be/cacheA")
    ce.extract_youtube_content("cacheA", "https://youtu.be/cacheA")
    assert calls == ["cacheA"]


def test_youtube_page_meta_parses_player_response(monkeypatch):
    html = ('<html><script>var ytInitialPlayerResponse = {"videoDetails":{"title":"T","shortDescription":"Full desc",'
            '"keywords":["a","b"],"author":"Chan"},"microformat":{"playerMicroformatRenderer":{"category":"Education"}}};'
            "var x=1;</script></html>")
    monkeypatch.setattr(ce, "fetch_url", lambda url, timeout=5: html)
    m = ce._youtube_page_meta("vid")
    assert (m["title"], m["description"], m["keywords"], m["channel"], m["category"]) == ("T", "Full desc", ["a", "b"], "Chan", "Education")


# ------------------------------------------------------------------- engine: normal / low content
def test_normal_text_high_and_low_relevance_do_not_call_llm():
    llm = FakeLLM(score=50)
    hi = engine(llm).analyze(PROFILE, ContentInput(title="Deadlock prevention", text=RELEVANT_TRANSCRIPT * 2))
    lo = engine(llm).analyze(PROFILE, ContentInput(title="Celebrity wedding gowns", text="Fans shared red carpet looks from the wedding gala. " * 8))
    assert hi.decision == "ALLOW" and lo.decision == "BLOCK"
    assert llm.calls == [] and hi.llm_status == lo.llm_status == "not_needed" and hi.evaluation_path == "embedding"


def test_low_content_page_uses_llm_as_main_reviewer():
    base = engine(None).analyze(PROFILE, ContentInput(title=ESPORTS_TITLE, description="Tournament highlights."))
    llm = FakeLLM(score=5)
    r = engine(llm).analyze(PROFILE, ContentInput(title=ESPORTS_TITLE, description="Tournament highlights."))
    assert llm.calls and r.llm_used and r.evaluation_path == "llm+embedding" and r.content_quality == "low"
    assert r.score == pytest.approx(round(0.3 * base.signals["local_score"] + 0.7 * 5, 1), abs=0.11)
    assert r.decision == "BLOCK"
    call = llm.calls[0]
    assert call.title == ESPORTS_TITLE and call.description == "Tournament highlights." and call.topic == "Operating System Deadlocks"


def test_low_content_llm_can_rescue_a_relevant_video_the_embeddings_underrate():
    r = engine(FakeLLM(score=95)).analyze(PROFILE, ContentInput(title="Why programs freeze: circular waiting explained", description="A short lesson."))
    assert r.llm_used and r.decision in ("ALLOW", "WARN") and r.score >= 30


def test_ambiguous_content_blends_embedding_and_llm_with_embedding_primary():
    content = ContentInput(title="CPU scheduling in operating systems", text="An operating system scheduler picks which process runs next. " * 4 + "Round robin and priority scheduling are common algorithms. Context switching has a cost for every process. " * 2)
    plain = engine(None).analyze(PROFILE, content)
    assert 25 <= plain.score <= 75, plain.score  # ambiguous band
    llm = FakeLLM(score=10, concepts=("scheduling",))
    r = engine(llm).analyze(PROFILE, content)
    assert r.llm_used and r.evaluation_path == "embedding+llm"
    assert r.score == pytest.approx(round(0.6 * plain.score + 0.4 * 10, 1), abs=0.11)
    assert "scheduling" in r.concepts and any("AI reviewer" in x for x in r.reasons)


def test_thresholds_are_unchanged_defaults():
    cfg = EngineConfig()
    assert (cfg.allow_threshold, cfg.warn_threshold) == (75.0, 30.0) and cfg.ambiguity_band == (25.0, 75.0)


def test_insufficient_content_never_calls_llm_and_fails_open():
    llm = FakeLLM(score=0)
    r = engine(llm).analyze(PROFILE, ContentInput(title="Home", text="Menu"))
    assert r.insufficient_content and r.decision == "ALLOW" and r.score == 50 and llm.calls == []
    assert r.evaluation_path == "insufficient" and r.content_quality == "insufficient"


@pytest.mark.parametrize("status", ["unavailable", "timeout", "http_error", "malformed"])
def test_llm_failure_falls_back_to_embedding_only(status):
    content = ContentInput(title=ESPORTS_TITLE, description=ESPORTS_DESC)
    plain = engine(None).analyze(PROFILE, content)
    r = engine(FakeLLM(status=status)).analyze(PROFILE, content)
    assert r.llm_used is False and r.llm_status == f"failed:{status}" and r.evaluation_path == "embedding"
    assert r.score == plain.score and r.decision == plain.decision == "BLOCK"  # still blocks on embeddings alone
    assert any("unavailable" in x for x in r.reasons)


def test_low_content_without_llm_uses_stricter_block_rule():
    # Unknown-but-unrelated short title: embeddings alone block only when clearly unrelated.
    r = engine(None).analyze(PROFILE, ContentInput(title="Celebrity wedding gowns", description="Gallery of looks."))
    assert r.low_content and r.decision == "BLOCK"


def test_llm_used_low_content_applies_normal_warn_threshold():
    # LLM says 25 (distracting); combined with a ~0 local score this is < 30 and must BLOCK even though < 15 is not met.
    r = engine(FakeLLM(score=25)).analyze(PROFILE, ContentInput(title="Celebrity wedding gowns", description="Gallery of looks."))
    assert r.llm_used and 15 <= r.score < 30 and r.decision == "BLOCK"


# ---------------------------------------------------------------- API with an LLM configured
def test_api_reports_llm_status_and_path(client, session, monkeypatch):
    from app.services import relevance_service

    llm = FakeLLM(score=3)
    monkeypatch.setattr(relevance_service, "get_engine", lambda: engine(llm))
    patch_youtube(monkeypatch, ESPORTS_TITLE, ESPORTS_DESC, transcript=None)
    b = analyze(client, session, "gameLLM").json()
    assert b["llm_used"] is True and b["llm_status"] == "ok" and b["evaluation_path"] == "llm+embedding" and b["decision"] == "BLOCK"


def test_api_survives_llm_failure(client, session, monkeypatch):
    from app.services import relevance_service

    monkeypatch.setattr(relevance_service, "get_engine", lambda: engine(FakeLLM(status="timeout")))
    patch_youtube(monkeypatch, ESPORTS_TITLE, ESPORTS_DESC, transcript=None)
    r = analyze(client, session, "gameLLM2")
    assert r.status_code == 200 and r.json()["llm_status"] == "failed:timeout" and r.json()["decision"] == "BLOCK"


def test_unreadable_url_returns_422_so_the_extension_fails_open(client, session):
    r = client.post("/api/relevance/analyze", json={"session_id": session["id"], "url": "http://127.0.0.1/x"})
    assert r.status_code == 422  # extension treats any non-2xx as "analysis failed" and lets the page load


# ------------------------------------------------------------------ LLM response validation
GOOD = {"classification": "DISTRACTING", "relevance_score": 8, "reason": " Esports content. ", "concepts": ["  ", "gaming", "x" * 200]}


def test_validate_accepts_and_trims_a_good_verdict():
    j = validate_judgement(GOOD)
    assert (j.classification, j.relevance_score, j.reason) == ("DISTRACTING", 8.0, "Esports content.")
    assert j.concepts[0] == "gaming" and len(j.concepts[1]) == 60


@pytest.mark.parametrize("bad", [
    None, "text", [], {},
    {**GOOD, "classification": "MAYBE"},
    {**GOOD, "relevance_score": "8"},
    {**GOOD, "relevance_score": True},
    {**GOOD, "relevance_score": float("nan")},
    {**GOOD, "relevance_score": 140},
    {**GOOD, "relevance_score": -3},
    {**GOOD, "classification": "RELEVANT", "relevance_score": 5},  # contradicts its own score
    {**GOOD, "concepts": "gaming"},
    {**GOOD, "concepts": [1, 2]},
    {**GOOD, "reason": 5},
])
def test_validate_rejects_malformed_verdicts(bad):
    with pytest.raises(LLMValidationError):
        validate_judgement(bad)


def test_validate_tolerates_adjacent_band_disagreement():
    assert validate_judgement({**GOOD, "classification": "PARTIALLY_RELEVANT", "relevance_score": 76}).relevance_score == 76


# -------------------------------------------------------------- LLMReasoner (HTTP mocked)
class FakeResponse:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=httpx.Request("POST", "http://x"), response=httpx.Response(self.status_code))

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def reasoner():
    return LLMReasoner("key", "model-x", "https://llm.example/v1/messages", 5)


def review_input():
    return LLMReviewInput(topic="OS Deadlocks", objectives="Banker's algorithm", keywords=["deadlock", "banker algorithm"],
                          title="Some title", description="Some description", content="Some transcript", url="https://youtu.be/x", source_type="youtube")


def test_reasoner_sends_structured_request_and_parses_tool_call(monkeypatch):
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, headers=headers, body=json)
        return FakeResponse({"content": [{"type": "tool_use", "name": "report_relevance", "input": GOOD}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    out = reasoner().review(review_input())
    assert out.status == "ok" and out.judgement.relevance_score == 8
    body = seen["body"]
    assert body["tool_choice"] == {"type": "tool", "name": "report_relevance"} and body["model"] == "model-x"
    assert seen["headers"]["x-api-key"] == "key"
    prompt = body["messages"][0]["content"]
    for label in ("STUDY TOPIC: OS Deadlocks", "LEARNING OBJECTIVE: Banker's algorithm", "KEYWORDS / CONCEPTS: deadlock, banker algorithm",
                  "PAGE TITLE: Some title", "PAGE DESCRIPTION: Some description", "EXTRACTED CONTENT / TRANSCRIPT: Some transcript"):
        assert label in prompt
    assert "<content>" in prompt and "untrusted" in body["system"]


def test_reasoner_accepts_json_in_plain_text_fallback(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse({"content": [{"type": "text", "text": 'Sure: {"classification":"RELEVANT","relevance_score":88,"reason":"ok","concepts":[]}'}]}))
    assert reasoner().review(review_input()).judgement.relevance_score == 88


@pytest.mark.parametrize("body", [
    {"content": [{"type": "text", "text": "I think it is irrelevant."}]},  # prose, no JSON
    {"content": [{"type": "text", "text": "{not json}"}]},
    {"content": [{"type": "tool_use", "name": "report_relevance", "input": {"classification": "RELEVANT", "relevance_score": "high"}}]},
    {"content": []},
])
def test_reasoner_reports_malformed_responses(monkeypatch, body):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse(body))
    out = reasoner().review(review_input())
    assert out.status == "malformed" and out.judgement is None


@pytest.mark.parametrize("raised,status", [
    (httpx.ReadTimeout("slow"), "timeout"),
    (httpx.ConnectError("down"), "unavailable"),
])
def test_reasoner_reports_network_failures(monkeypatch, raised, status):
    def boom(*a, **k):
        raise raised

    monkeypatch.setattr(httpx, "post", boom)
    assert reasoner().review(review_input()).status == status


def test_reasoner_reports_http_errors_and_bad_envelopes(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse({}, status=401))
    assert reasoner().review(review_input()).status == "http_error"
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse(ValueError("not json")))
    assert reasoner().review(review_input()).status == "unavailable"


def test_get_llm_uses_existing_env_configuration(monkeypatch):
    from app.ai.llm import get_llm

    monkeypatch.setenv("LLM_API_KEY", "")
    assert get_llm() is None
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_API_URL", "https://u")
    r = get_llm()
    assert (r.api_key, r.model, r.url) == ("k", "m", "https://u")
