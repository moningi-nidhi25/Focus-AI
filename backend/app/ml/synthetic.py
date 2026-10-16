"""Synthetic study sessions (roadmap Phase 8, Stage 1).

A hidden 'true focus' level drifts down with fatigue, drops after distraction episodes and
recovers after breaks. The observable columns are noisy functions of it. This is only a
stand-in until real FocusAI data is collected (Stage 3)."""
import numpy as np

from app.ml.model import HORIZON, LOOKBACK


def simulate_session(rng: np.random.Generator) -> np.ndarray:
    steps = int(rng.integers(30, 61))
    focus = rng.uniform(0.6, 0.95)
    fatigue = rng.uniform(0.003, 0.015)
    distract_p = rng.uniform(0.02, 0.12)
    since_break = 0
    rows = []
    for _ in range(steps):
        focus -= fatigue * (1 + since_break / 30)
        focus += rng.normal(0, 0.02)
        distracted = rng.random() < distract_p * (1.5 - focus)
        if distracted:
            focus -= rng.uniform(0.05, 0.2)
        if rng.random() < 0.03:
            focus += rng.uniform(0.15, 0.3)
            since_break = 0
        else:
            since_break += 1
        focus = float(np.clip(focus, 0.05, 1.0))

        dwell = float(np.clip(rng.normal(55 * (0.6 + 0.4 * focus), 6), 5, 60))
        tabs = int(rng.poisson(0.5 + (1 - focus) * 8))
        dist = int(rng.poisson((1 - focus) * 1.8)) + int(distracted)
        measured = float(np.clip(focus * 100 + rng.normal(0, 4) - 3 * dist, 0, 100))
        productivity = float(np.clip(0.8 * measured + rng.normal(0, 5), 0, 100))
        rows.append([measured, productivity, dwell, tabs, dist])
    return np.array(rows, dtype=np.float32)


def generate_sessions(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    return [simulate_session(rng) for _ in range(n)]


def make_windows(sessions):
    """Sliding windows: X = last LOOKBACK rows, y = next HORIZON focus scores (0..100)."""
    xs, ys = [], []
    for s in sessions:
        for i in range(0, len(s) - LOOKBACK - HORIZON + 1):
            xs.append(s[i:i + LOOKBACK])
            ys.append(s[i + LOOKBACK:i + LOOKBACK + HORIZON, 0])
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)