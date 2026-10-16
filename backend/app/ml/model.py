import numpy as np
from torch import nn

# Same columns as the focus_metrics table, in this order.
FEATURES = ["focus_score", "productivity_score", "dwell_time_seconds", "tab_switch_count", "distraction_count"]
NORM = np.array([100.0, 100.0, 60.0, 10.0, 3.0], dtype=np.float32)  # fixed scales -> roughly 0..1
LOOKBACK = 10   # past windows the model sees
HORIZON = 5     # future windows it predicts


def normalize(rows) -> np.ndarray:
    return np.clip(np.asarray(rows, dtype=np.float32) / NORM, 0.0, 2.0)


class FocusLSTM(nn.Module):
    def __init__(self, n_features: int = len(FEATURES), hidden: int = 48, horizon: int = HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, horizon))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # focus/100 for each future window