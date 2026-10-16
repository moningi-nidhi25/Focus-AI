"""Phase 7 scores and Phase 8 LSTM."""
import pytest

from app.ml.model import FEATURES, HORIZON, LOOKBACK
from app.services.prediction_service import ARTIFACT

needs_model = pytest.mark.skipif(not ARTIFACT.exists(), reason="run scripts/train_lstm.py first")


def test_events_produce_documented_scores(client, session):
    """Hand-calculated with the formulas in metrics_service: dwell 150s (120s relevant), 1 tab switch,
    no distractions, 2.5 min of activity -> focus = 100*(0.5*0.8 + 0.25*1 + 0.25*0.6) = 80,
    productivity = 100*(0.6*0.8 + 0.4*min(1, 2/2.5)) = 80."""
    sid = session["id"]
    events = [
        {"event_type": "page_open", "metadata": {"host": "example.com"}},
        {"event_type": "dwell_time", "metadata": {"seconds": 120, "decision": "ALLOW"}},
        {"event_type": "dwell_time", "metadata": {"seconds": 30, "decision": "BLOCK"}},
        {"event_type": "tab_switch", "metadata": {}},
    ]
    r = client.post("/api/activity/events", json={"session_id": sid, "events": events})
    assert r.status_code == 201 and r.json()["stored"] == 4

    s = client.get(f"/api/analytics/sessions/{sid}/summary").json()
    assert s["total_dwell_seconds"] == 150 and s["relevant_seconds"] == 120
    assert s["tab_switch_count"] == 1 and s["distraction_count"] == 0
    assert s["focus_score"] == pytest.approx(80.0, abs=1.0)
    assert s["productivity_score"] == pytest.approx(80.0, abs=1.0)


def test_distractions_lower_the_focus_score(client, session):
    sid = session["id"]
    calm = [{"event_type": "dwell_time", "metadata": {"seconds": 100, "decision": "ALLOW"}}]
    client.post("/api/activity/events", json={"session_id": sid, "events": calm})
    before = client.get(f"/api/analytics/sessions/{sid}/summary").json()["focus_score"]
    noisy = [{"event_type": "distraction_attempt", "metadata": {}}] * 6
    client.post("/api/activity/events", json={"session_id": sid, "events": noisy})
    after = client.get(f"/api/analytics/sessions/{sid}/summary").json()
    assert after["distraction_count"] == 6 and after["focus_score"] < before


def test_invalid_event_type_is_rejected(client, session):
    r = client.post("/api/activity/events",
                    json={"session_id": session["id"], "events": [{"event_type": "hack", "metadata": {}}]})
    assert r.status_code == 422


def test_synthetic_windows_have_the_right_shape():
    from app.ml.synthetic import generate_sessions, make_windows

    x, y = make_windows(generate_sessions(5, seed=1))
    assert x.shape[1:] == (LOOKBACK, len(FEATURES)) and y.shape[1] == HORIZON
    assert (y >= 0).all() and (y <= 100).all()


def test_model_output_shape():
    import torch
    from app.ml.model import FocusLSTM

    out = FocusLSTM()(torch.zeros(2, LOOKBACK, len(FEATURES)))
    assert tuple(out.shape) == (2, HORIZON)


@needs_model
def test_prediction_needs_history(client, session):
    assert client.post(f"/api/prediction/sessions/{session['id']}").status_code == 422


@needs_model
def test_prediction_curve_is_valid(client, session):
    sid = session["id"]
    for _ in range(3):
        assert client.post(f"/api/analytics/sessions/{sid}/snapshot").status_code == 200
    r = client.post(f"/api/prediction/sessions/{sid}")
    assert r.status_code == 200
    body = r.json()
    assert len(body["curve"]) == HORIZON
    for p in body["curve"]:
        assert 0 <= p["lower"] <= p["predicted"] <= p["upper"] <= 100