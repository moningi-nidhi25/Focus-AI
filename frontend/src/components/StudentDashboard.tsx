import { useEffect, useState } from "react";
import { dash, fmtChange, fmtDate, fmtMinutes } from "../dashApi";
import type { Prediction, StudentOverview } from "../dashApi";
import { DaysSelect, DecayChart, DistractionChart, FocusTrend, Insights, MinutesChart, StatCard } from "./charts";

export default function StudentDashboard() {
  const [days, setDays] = useState(14);
  const [data, setData] = useState<StudentOverview | null>(null);
  const [error, setError] = useState("");
  const [pred, setPred] = useState<Prediction | null>(null);
  const [predError, setPredError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    dash.student(days)
      .then((d) => { if (!cancelled) { setData(d); setError(""); } })
      .catch((e: Error) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [days]);

  async function predict() {
    const id = data?.recent_sessions[0]?.id;
    if (!id) { setPredError("Start a study session first."); return; }
    setBusy(true);
    setPredError("");
    try {
      setPred(await dash.predict(id));
    } catch (e) {
      setPredError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setBusy(false);
    }
  }

  if (error) {
    return (
      <div className="d-wrap">
        <div className="d-error">
          <span>⚠️ {error}</span>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="d-wrap">
        <div className="d-card" style={{ textAlign: "center", padding: "48px 24px" }}>
          <p className="d-hint" style={{ fontSize: "14px" }}>Loading your dashboard metrics...</p>
        </div>
      </div>
    );
  }

  const s = data.summary;

  return (
    <div className="d-wrap">
      <div className="d-header-row">
        <div className="d-title-group">
          <h2>My Dashboard</h2>
          <p>Real-time analytics, study history, and focus predictions.</p>
        </div>
        <DaysSelect value={days} onChange={setDays} />
      </div>

      <div className="d-grid">
        <StatCard label="Avg focus score" value={s.avg_focus} hint={fmtChange(data.focus_change_vs_previous_week)} />
        <StatCard label="Productivity score" value={s.avg_productivity} />
        <StatCard label="Study time" value={fmtMinutes(s.study_minutes)} />
        <StatCard label="Distractions" value={s.distractions} />
        <StatCard label="Blocked pages" value={data.blocked_count} />
        <StatCard label="Tab switches" value={s.tab_switches} />
      </div>

      <Insights items={data.insights} />

      <div className="d-card">
        <div className="d-card-header">
          <h3>Focus score by day</h3>
          <p className="d-hint">Evaluates session attentiveness on a 0–100 scale.</p>
        </div>
        <FocusTrend data={data.trend} />
      </div>

      <div className="d-card">
        <div className="d-card-header">
          <h3>Study minutes by day</h3>
          <p className="d-hint">Total focused time recorded across study sessions.</p>
        </div>
        <MinutesChart data={data.trend} />
      </div>

      <div className="d-card">
        <div className="d-card-header">
          <h3>Distractions by day</h3>
          <p className="d-hint">Instances of visits to distracting or off-topic websites.</p>
        </div>
        <DistractionChart data={data.trend} />
      </div>

      <div className="d-card">
        <div className="d-card-header">
          <h3>Focus Decay Curve</h3>
          <p className="d-hint">Predicts your focus progression for the next 5 activity windows based on recent sessions.</p>
        </div>
        <div className="d-row" style={{ marginBottom: "14px" }}>
          <button className="d-btn" onClick={predict} disabled={busy}>
            {busy ? "Analyzing decay patterns..." : "Predict my focus"}
          </button>
        </div>
        {predError && <div className="d-error">{predError}</div>}
        {pred ? (
          <>
            <div className={pred.alert ? "d-alert" : "d-ok"}>
              <span>{pred.alert ? "⚠️" : "✓"}</span>
              <span>{pred.message}</span>
            </div>
            <DecayChart current={pred.current_focus} curve={pred.curve} />
            <p className="d-hint" style={{ marginTop: "12px" }}>{pred.model_note}</p>
          </>
        ) : data.focus_decay ? (
          <>
            <p className="d-hint" style={{ marginBottom: "8px" }}>Last saved prediction ({fmtDate(data.focus_decay.created_at)})</p>
            <DecayChart curve={data.focus_decay.curve} />
          </>
        ) : null}
      </div>

      <div className="d-card">
        <div className="d-card-header">
          <h3>Recent sessions</h3>
          <p className="d-hint">Your latest study session milestones and focus averages.</p>
        </div>
        {data.recent_sessions.length === 0 ? (
          <p className="d-hint" style={{ padding: "16px 0" }}>No sessions yet. Start a study session to track your activity.</p>
        ) : (
          <div className="d-table-wrapper">
            <table className="d-table">
              <thead>
                <tr>
                  <th>Session Title</th>
                  <th>Started</th>
                  <th>Avg Focus</th>
                </tr>
              </thead>
              <tbody>
                {data.recent_sessions.map((r) => (
                  <tr key={r.id}>
                    <td style={{ fontWeight: 500 }}>{r.title}</td>
                    <td className="muted">{fmtDate(r.start_time)}</td>
                    <td>
                      {r.avg_focus != null ? (
                        <span style={{ fontWeight: 600 }}>{r.avg_focus}</span>
                      ) : (
                        <span className="muted">–</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="d-card">
        <div className="d-card-header">
          <h3>Relevance history</h3>
          <p className="d-hint">Recent web pages evaluated by the FocusAI engine against your study goal.</p>
        </div>
        {data.relevance_history.length === 0 ? (
          <p className="d-hint" style={{ padding: "16px 0" }}>No analyzed pages in this period.</p>
        ) : (
          <div className="d-table-wrapper">
            <table className="d-table">
              <thead>
                <tr>
                  <th>Page</th>
                  <th>Relevance</th>
                  <th>Decision</th>
                </tr>
              </thead>
              <tbody>
                {data.relevance_history.map((h, i) => (
                  <tr key={i}>
                    <td className="cell-url" title={h.url}>
                      {h.title || h.url}
                    </td>
                    <td>
                      <span style={{ fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
                        {Math.round(h.relevance_score)}%
                      </span>
                    </td>
                    <td>
                      <span className={`d-badge ${h.decision.toLowerCase()}`}>
                        {h.decision}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}