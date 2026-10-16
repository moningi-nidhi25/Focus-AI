"""Train the focus-decay LSTM on synthetic data:  uv run python scripts/train_lstm.py"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.model import HORIZON, LOOKBACK, FocusLSTM, normalize
from app.ml.synthetic import generate_sessions, make_windows

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

sessions = generate_sessions(2000, SEED)
n_train, n_val = int(0.70 * len(sessions)), int(0.15 * len(sessions))
parts = {  # split by SESSION so no session leaks across train/val/test
    "train": sessions[:n_train],
    "val": sessions[n_train:n_train + n_val],
    "test": sessions[n_train + n_val:],
}
raw = {k: make_windows(v) for k, v in parts.items()}
X = {k: torch.from_numpy(normalize(v[0])) for k, v in raw.items()}
Y = {k: torch.from_numpy(v[1] / 100.0) for k, v in raw.items()}
print({k: len(v) for k, v in X.items()}, "windows")

model = FocusLSTM()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
loader = DataLoader(TensorDataset(X["train"], Y["train"]), batch_size=128, shuffle=True)


def predict_points(x):
    model.eval()
    with torch.no_grad():
        return model(x).clamp(0, 1).numpy() * 100


best, best_state, bad = 1e9, None, 0
for epoch in range(1, 41):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()
    val_mae = float(np.abs(predict_points(X["val"]) - raw["val"][1]).mean())
    print(f"epoch {epoch:2d}  val MAE {val_mae:.2f} points")
    if val_mae < best - 0.01:
        best, best_state, bad = val_mae, copy.deepcopy(model.state_dict()), 0
    else:
        bad += 1
        if bad >= 5:
            break

model.load_state_dict(best_state)
resid_val = predict_points(X["val"]) - raw["val"][1]
residual_std = [float(v) for v in resid_val.std(axis=0)]

pred_test, y_test = predict_points(X["test"]), raw["test"][1]
persistence = np.repeat(raw["test"][0][:, -1, 0:1], HORIZON, axis=1)  # "focus stays the same"
print("\nTEST MAE (score points) per future window")
print("LSTM       ", np.round(np.abs(pred_test - y_test).mean(axis=0), 2))
print("Persistence", np.round(np.abs(persistence - y_test).mean(axis=0), 2))

out = Path(__file__).resolve().parent.parent / "app" / "ml" / "artifacts"
out.mkdir(parents=True, exist_ok=True)
torch.save({"state_dict": model.state_dict(), "residual_std": residual_std,
            "lookback": LOOKBACK, "horizon": HORIZON, "trained_on": "synthetic"}, out / "focus_lstm.pt")
print("saved", out / "focus_lstm.pt")