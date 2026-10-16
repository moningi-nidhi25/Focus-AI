"""Gemini provider: request format, response parsing, validation, API errors, configuration.

All HTTP is mocked (httpx.post); no API key or network is needed.
"""
import json
import logging

import httpx
import pytest

from app.ai.context import build_profile
from app.ai.embeddings import LexicalEmbedder
from app.ai.llm import (
    AnthropicReasoner, GeminiReasoner, LLMReasoner, LLMReviewInput, get_llm,
)
from app.ai.relevance_engine import ContentInput, EngineConfig, RelevanceEngine

SECRET = "AIzaSy-SECRET-KEY-1234567890"
GOOD = {"classification": "DISTRACTING", "relevance_score": 6, "reason": "Esports highlights, unrelated to deadlocks.", "concepts": ["esports"]}


def gemini_body(payload, finish="STOP"):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": finish}]}


class FakeResponse:
    def __init__(self, body=None, status=200, text=None):
        self._body, self.status_code = body, status
        self.text = text if text is not None else (json.dumps(body) if isinstance(body, (dict, list)) else str(body))

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=httpx.Request("POST", "http://x"), response=self)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def reasoner(**kw):
    return GeminiReasoner(SECRET, kw.pop("model", "gemini-3.5-flash-lite"), timeout=5, **kw)


def review_input(**kw):
    base = dict(topic="Operating System Deadlocks", objectives="Understand Banker's algorithm", keywords=["deadlock", "banker algorithm"],
                title="Squad wins the grand final", description="Tournament highlights", content="Casters talk about the match",
                url="https://www.youtube.com/watch?v=x", source_type="youtube")
    base.update(kw)
    return LLMReviewInput(**base)


def mock_post(monkeypatch, *responses):
    """Return queued responses in order; record every call."""
    calls, queue = [], list(responses)

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


# ------------------------------------------------------------------------- request format
def test_request_uses_generate_content_endpoint_header_auth_and_json_mode(monkeypatch):
    calls = mock_post(monkeypatch, FakeResponse(gemini_body(GOOD)))
    out = reasoner().review(review_input())
    assert out.status == "ok"
    c = calls[0]
    assert c["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"
    assert c["headers"]["x-goog-api-key"] == SECRET and c["timeout"] == 5
    assert SECRET not in c["url"] and SECRET not in json.dumps(c["json"])  # key only in the header
    gen = c["json"]["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert gen["responseJsonSchema"]["properties"]["classification"]["enum"] == ["RELEVANT", "PARTIALLY_RELEVANT", "DISTRACTING"]
    assert gen["responseJsonSchema"]["required"] == ["classification", "relevance_score", "reason", "concepts"]
    assert "temperature" not in gen and "thinkingConfig" not in gen  # Gemini 3.x guidance: leave defaults


def test_prompt_contains_all_structured_fields_and_marks_content_untrusted(monkeypatch):
    calls = mock_post(monkeypatch, FakeResponse(gemini_body(GOOD)))
    reasoner().review(review_input())
    body = calls[0]["json"]
    prompt = body["contents"][0]["parts"][0]["text"]
    for label in ("STUDY TOPIC: Operating System Deadlocks", "LEARNING OBJECTIVE: Understand Banker's algorithm",
                  "KEYWORDS / CONCEPTS: deadlock, banker algorithm", "PAGE TITLE: Squad wins the grand final",
                  "PAGE DESCRIPTION: Tournament highlights", "EXTRACTED CONTENT / TRANSCRIPT: Casters talk about the match"):
        assert label in prompt
    system = body["systemInstruction"]["parts"][0]["text"]
    assert "untrusted" in system and "JSON object" in system and "75" in system and "30" in system
    assert body["contents"][0]["role"] == "user"


def test_model_prefix_is_normalised_and_bad_names_rejected():
    assert GeminiReasoner("k", "models/gemini-3.6-flash").endpoint.endswith("/models/gemini-3.6-flash:generateContent")
    for bad in ("../../admin", "gemini 3", "gemini?x=1", "a/b"):
        with pytest.raises(ValueError):
            GeminiReasoner("k", bad)


# ------------------------------------------------------------------------- successful parsing
def test_successful_response_is_validated_into_a_judgement(monkeypatch):
    mock_post(monkeypatch, FakeResponse(gemini_body(GOOD)))
    out = reasoner().review(review_input())
    j = out.judgement
    assert out.status == "ok" and (j.classification, j.relevance_score, j.concepts) == ("DISTRACTING", 6.0, ["esports"])
    assert j.reason.startswith("Esports")


def test_json_in_code_fence_and_multiple_parts_and_thought_parts(monkeypatch):
    fenced = "```json\n" + json.dumps(GOOD) + "\n```"
    body = {"candidates": [{"content": {"parts": [{"text": "internal reasoning", "thought": True}, {"text": fenced[:20]}, {"text": fenced[20:]}]}, "finishReason": "STOP"}]}
    mock_post(monkeypatch, FakeResponse(body))
    out = reasoner().review(review_input())
    assert out.status == "ok" and out.judgement.relevance_score == 6


def test_concepts_are_trimmed_and_capped(monkeypatch):
    payload = {**GOOD, "concepts": ["  a  ", "", "b", "c", "d", "e", "f"]}
    mock_post(monkeypatch, FakeResponse(gemini_body(payload)))
    assert reasoner().review(review_input()).judgement.concepts == ["a", "b", "c", "d", "e"]


# ------------------------------------------------------------------------- malformed responses
@pytest.mark.parametrize("body", [
    gemini_body("I think this is unrelated."),                                  # prose, no JSON
    gemini_body("{not valid json"),                                             # invalid JSON
    gemini_body('{"classification": "DISTRACTING"}'),                           # missing keys
    gemini_body({"classification": "DISTRACTING", "relevance_score": "low", "reason": "r", "concepts": []}),   # score not a number
    gemini_body({"classification": "MAYBE", "relevance_score": 10, "reason": "r", "concepts": []}),            # bad label
    gemini_body({"classification": "DISTRACTING", "relevance_score": 150, "reason": "r", "concepts": []}),     # out of range
    gemini_body({"classification": "RELEVANT", "relevance_score": 3, "reason": "r", "concepts": []}),          # contradicts itself
    gemini_body({"classification": "DISTRACTING", "relevance_score": 3, "reason": "r", "concepts": "x"}),      # concepts wrong type
    gemini_body("[1, 2, 3]"),                                                   # JSON but not an object
    gemini_body('{"classification": "DISTRACTING", "relevance_score": 5, "reas', finish="MAX_TOKENS"),          # truncated
    {"candidates": []},                                                         # nothing returned
    {"candidates": [{"finishReason": "SAFETY"}]},                               # candidate without content
    {"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]},       # empty parts
    {"candidates": ["oops"]},                                                   # wrong candidate type
    {"candidates": None},
    {"promptFeedback": {"blockReason": "SAFETY"}},                              # prompt blocked
    {},                                                                         # empty envelope
    [],                                                                         # envelope is a list
])
def test_malformed_gemini_responses_are_rejected_not_crashed(monkeypatch, body):
    mock_post(monkeypatch, FakeResponse(body))
    out = reasoner().review(review_input())
    assert out.status == "malformed" and out.judgement is None and out.detail


def test_blocked_prompt_and_truncation_are_explained(monkeypatch):
    mock_post(monkeypatch, FakeResponse({"promptFeedback": {"blockReason": "SAFETY"}}))
    assert "blocked" in reasoner().review(review_input()).detail
    mock_post(monkeypatch, FakeResponse(gemini_body('{"classification": "DIS', finish="MAX_TOKENS")))
    assert "MAX_TOKENS" in reasoner().review(review_input()).detail


def test_non_json_http_body_is_unavailable_not_a_crash(monkeypatch):
    mock_post(monkeypatch, FakeResponse(ValueError("Expecting value")))
    assert reasoner().review(review_input()).status == "unavailable"


# ------------------------------------------------------------------------- API / network errors
@pytest.mark.parametrize("code", [401, 403, 404, 429, 500, 503])
def test_http_errors_are_reported(monkeypatch, code):
    err = {"error": {"code": code, "message": "something happened", "status": "SOME_STATUS"}}
    mock_post(monkeypatch, FakeResponse(err, status=code))
    out = reasoner().review(review_input())
    assert out.status == "http_error" and f"HTTP {code}" in out.detail and "SOME_STATUS" in out.detail


def test_timeout_and_connection_errors(monkeypatch):
    mock_post(monkeypatch, httpx.ReadTimeout("slow"))
    assert reasoner().review(review_input()).status == "timeout"
    mock_post(monkeypatch, httpx.ConnectError("no route"))
    assert reasoner().review(review_input()).status == "unavailable"


def test_unexpected_exception_never_escapes(monkeypatch):
    mock_post(monkeypatch, RuntimeError("kaboom"))
    assert reasoner().review(review_input()).status == "unavailable"


def test_schema_rejection_retries_once_in_plain_json_mode(monkeypatch):
    rejected = FakeResponse({"error": {"message": "Unknown name responseJsonSchema"}}, status=400, text='{"error": {"message": "Invalid JSON payload: responseJsonSchema schema not supported"}}')
    calls = mock_post(monkeypatch, rejected, FakeResponse(gemini_body(GOOD)))
    out = reasoner().review(review_input())
    assert out.status == "ok" and len(calls) == 2
    assert "responseJsonSchema" in calls[0]["json"]["generationConfig"]
    assert "responseJsonSchema" not in calls[1]["json"]["generationConfig"]
    assert calls[1]["json"]["generationConfig"]["responseMimeType"] == "application/json"


def test_other_400_errors_are_not_retried(monkeypatch):
    calls = mock_post(monkeypatch, FakeResponse({"error": {"message": "API key not valid"}}, status=400, text='{"error": {"message": "API key not valid"}}'))
    out = reasoner().review(review_input())
    assert out.status == "http_error" and len(calls) == 1


# ------------------------------------------------------------------------- secrets never leak
def test_api_key_never_appears_in_logs_errors_or_repr(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    echo = {"error": {"status": "UNAUTHENTICATED", "message": f"API key not valid: {SECRET}"}}
    mock_post(monkeypatch, FakeResponse(echo, status=401), httpx.ConnectError(f"failed for key {SECRET}"),
              FakeResponse(gemini_body(f"leaked {SECRET}")))
    r = reasoner()
    outcomes = [r.review(review_input()) for _ in range(3)]
    blob = " ".join([o.detail for o in outcomes] + [caplog.text, repr(r), str(r)])
    assert SECRET not in blob and "***" in blob


# ------------------------------------------------------------------------- configuration
@pytest.fixture()
def env(monkeypatch):
    for k in ("LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL", "LLM_API_URL", "LLM_TIMEOUT_SECONDS"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_no_key_means_llm_disabled_for_every_provider(env):
    assert get_llm() is None
    env.setenv("LLM_PROVIDER", "gemini")
    assert get_llm() is None


def test_gemini_provider_defaults_to_verified_stable_model(env):
    env.setenv("LLM_PROVIDER", "gemini")
    env.setenv("LLM_API_KEY", SECRET)
    r = get_llm()
    assert isinstance(r, GeminiReasoner) and r.model == "gemini-3.5-flash-lite" and r.timeout == 15.0
    assert r.endpoint.startswith("https://generativelanguage.googleapis.com/v1beta/models/")


def test_gemini_environment_overrides(env):
    env.setenv("LLM_PROVIDER", " Gemini ")
    env.setenv("LLM_API_KEY", SECRET)
    env.setenv("LLM_MODEL", "gemini-3.6-flash")
    env.setenv("LLM_TIMEOUT_SECONDS", "7.5")
    env.setenv("LLM_API_URL", "https://proxy.example/v1beta/")
    r = get_llm()
    assert (r.model, r.timeout, r.endpoint) == ("gemini-3.6-flash", 7.5, "https://proxy.example/v1beta/models/gemini-3.6-flash:generateContent")


def test_leftover_anthropic_model_name_does_not_break_gemini(env):
    env.setenv("LLM_PROVIDER", "gemini")
    env.setenv("LLM_API_KEY", SECRET)
    env.setenv("LLM_MODEL", "claude-haiku-5-5")
    assert get_llm().model == "gemini-3.5-flash-lite"


@pytest.mark.parametrize("timeout", ["abc", "0", "-3"])
def test_invalid_timeout_falls_back_to_default(env, timeout):
    env.setenv("LLM_PROVIDER", "gemini")
    env.setenv("LLM_API_KEY", SECRET)
    env.setenv("LLM_TIMEOUT_SECONDS", timeout)
    assert get_llm().timeout == 15.0


def test_unknown_provider_or_bad_model_disables_llm_instead_of_crashing(env):
    env.setenv("LLM_API_KEY", SECRET)
    env.setenv("LLM_PROVIDER", "openai")
    assert get_llm() is None
    env.setenv("LLM_PROVIDER", "gemini")
    env.setenv("LLM_MODEL", "../../etc")
    assert get_llm() is None


def test_anthropic_remains_the_default_provider(env):
    env.setenv("LLM_API_KEY", "k")
    r = get_llm()
    assert isinstance(r, AnthropicReasoner) and r is not None and LLMReasoner is AnthropicReasoner
    assert r.model == "claude-haiku-5-5" and r.url == "https://api.anthropic.com/v1/messages"
    env.setenv("LLM_PROVIDER", "anthropic")
    assert isinstance(get_llm(), AnthropicReasoner)


# ------------------------------------------------------------------------- engine integration
PROFILE = build_profile("Operating System Deadlocks", "Understand deadlock prevention, detection, and Banker's algorithm.")
SHORT = ContentInput(title="Squad wins the grand final | mobile esports semi finals", description="Tournament highlights.", source_type="youtube")


def test_engine_uses_gemini_for_low_content_with_the_existing_weights(monkeypatch):
    mock_post(monkeypatch, FakeResponse(gemini_body(GOOD)))
    base = RelevanceEngine(LexicalEmbedder(), None).analyze(PROFILE, SHORT)
    r = RelevanceEngine(LexicalEmbedder(), reasoner(), EngineConfig()).analyze(PROFILE, SHORT)
    assert r.llm_used and r.llm_status == "ok" and r.evaluation_path == "llm+embedding"
    assert r.score == pytest.approx(round(0.3 * base.signals["local_score"] + 0.7 * 6, 1), abs=0.11)  # unchanged 0.3 / 0.7
    assert r.decision == "BLOCK" and "esports" in r.concepts
    assert (EngineConfig().allow_threshold, EngineConfig().warn_threshold, EngineConfig().llm_weight) == (75.0, 30.0, 0.4)


@pytest.mark.parametrize("response,status", [
    (FakeResponse(gemini_body("garbage")), "malformed"),
    (FakeResponse({"error": {}}, status=429), "http_error"),
    (httpx.ReadTimeout("slow"), "timeout"),
    (httpx.ConnectError("down"), "unavailable"),
])
def test_engine_falls_back_to_embeddings_when_gemini_fails(monkeypatch, response, status):
    mock_post(monkeypatch, response)
    plain = RelevanceEngine(LexicalEmbedder(), None).analyze(PROFILE, SHORT)
    r = RelevanceEngine(LexicalEmbedder(), reasoner(), EngineConfig()).analyze(PROFILE, SHORT)
    assert r.llm_used is False and r.llm_status == f"failed:{status}" and r.evaluation_path == "embedding"
    assert (r.score, r.decision) == (plain.score, plain.decision)


def test_api_reports_provider_without_exposing_the_key(client, monkeypatch):
    from app.api import relevance as relevance_api

    monkeypatch.setattr(relevance_api, "get_engine", lambda: RelevanceEngine(LexicalEmbedder(), reasoner(), EngineConfig()))
    body = client.get("/api/relevance/engine").json()
    assert body["llm_enabled"] is True and body["llm_provider"] == "gemini" and body["llm_model"] == "gemini-3.5-flash-lite"
    assert SECRET not in json.dumps(body)
