import json
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.ml.model import FEATURES, LOOKBACK, FocusLSTM, normalize
from app.models import FocusMetric, FocusPrediction, StudySession

ARTIFACT = Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "focus_lstm.pt"
ALERT_THRESHOLD = 50.0   # warn when predicted focus falls below this
MIN_HISTORY = 3
_cache: dict = {}


class ModelNotTrained(Exception):
    pass


class NotEnoughData(Exception):
    pass


def _load():
    if "model" not in _cache:
        if not ARTIFACT.exists():
            raise ModelNotTrained()
        import torch  # imported lazily so the API still starts without torch
        ck = torch.load(ARTIFACT, map_location="cpu", weights_only=True)
        model = FocusLSTM()
        model.load_state_dict(ck["state_dict"])
        model.eval()
        _cache["model"] = (model, np.array(ck["residual_std"], dtype=np.float32), ck.get("trained_on", "unknown"))
    return _cache["model"]


def predict(db: Session, session: StudySession) -> dict:
    import torch
    model, resid_std, trained_on = _load()

    rows = (db.query(FocusMetric).filter(FocusMetric.session_id == session.id)
            .order_by(FocusMetric.calculated_at.desc()).limit(LOOKBACK).all())[::-1]
    if len(rows) < MIN_HISTORY:
        raise NotEnoughData(f"Need at least {MIN_HISTORY} activity snapshots, found {len(rows)}. "
                            "Keep studying with the extension on for a few more minutes.")
    data = [[getattr(r, f) or 0 for f in FEATURES] for r in rows]
    while len(data) < LOOKBACK:          # left-pad short histories with the earliest snapshot
        data.insert(0, data[0])

    x = torch.from_numpy(normalize(data)).unsqueeze(0)
    with torch.no_grad():
        pred = model(x)[0].clamp(0, 1).numpy() * 100

    curve = [{"window": i + 1, "predicted": round(float(p), 1),
              "lower": round(float(max(0.0, p - 1.96 * s)), 1),
              "upper": round(float(min(100.0, p + 1.96 * s)), 1)}
             for i, (p, s) in enumerate(zip(pred, resid_std))]

    alert_window = next((c["window"] for c in curve if c["predicted"] < ALERT_THRESHOLD), None)
    current = float(rows[-1].focus_score)
    if alert_window:
        message = f"Your focus is predicted to drop below {int(ALERT_THRESHOLD)} within {alert_window} windows. Consider a short break."
    else:
        message = "Your focus is predicted to stay steady."

    # "confidence" is a simple heuristic from the validation error band (1.0 = very tight).
    confidence = round(float(max(0.0, 1 - resid_std.mean() / 25)), 2)
    db.add(FocusPrediction(session_id=session.id, predicted_focus_score=curve[-1]["predicted"],
                           target_time_window=f"next {len(curve)} windows", confidence=confidence,
                           curve_data=json.dumps(curve)))
    db.commit()

    return {"session_id": session.id, "current_focus": round(current, 1), "curve": curve,
            "alert": alert_window is not None, "alert_window": alert_window, "message": message,
            "history_points": len(rows),
            "model_note": f"LSTM trained on {trained_on} data; treat as a demonstration until retrained on real sessions."}