import { useEffect, useState } from "react";
import { dash, fmtMinutes } from "../dashApi";
import type { AdminOverview } from "../dashApi";
import { DaysSelect, DistractionChart, FocusTrend, MinutesChart, StatCard } from "./charts";

export default function AdminDashboard() {
  const [days, setDays] = useState(14);
  const [data, setData] = useState<AdminOverview | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    dash.admin(days)
      .then((d) => { if (!cancelled) { setData(d); setError(""); } })
      .catch((e: Error) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [days]);

  if (error) return <div className="d-wrap"><div className="d-error">{error}</div></div>;
  if (!data) {
    return (
      <div className="d-wrap">
        <div className="d-card" style={{ textAlign: "center", padding: "48px 24px" }}>
          <p className="d-hint">Loading platform analytics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="d-wrap">
      <div className="d-header-row">
        <div className="d-title-group">
          <h2>Administrator Dashboard</h2>
          <p>System-wide health indicators, usage totals, and aggregate trends.</p>
        </div>
        <DaysSelect value={days} onChange={setDays} />
      </div>

      <div className="d-grid">
        <StatCard label="Total students" value={data.total_students} />
        <StatCard label="Active teachers" value={data.active_teachers} hint={`${data.total_teachers} teacher accounts`} />
        <StatCard label="Cohorts / courses" value={data.total_cohorts} />
        <StatCard label="Study sessions" value={data.total_sessions} />
        <StatCard label="Active students (7d)" value={data.active_students_7d} />
        <StatCard label="Platform study hours" value={data.platform_study_hours_all_time} hint="All time total" />
        <StatCard label="Avg focus" value={data.summary.avg_focus} hint={`Last ${data.days} days`} />
        <StatCard label="Study time" value={fmtMinutes(data.summary.study_minutes)} hint={`Last ${data.days} days`} />
      </div>

      <div className="d-card">
        <div className="d-card-header">
          <h3>Platform focus by day</h3>
          <p className="d-hint">Cross-platform average student focus progression.</p>
        </div>
        <FocusTrend data={data.trend} />
      </div>

      <div className="d-card">
        <div className="d-card-header">
          <h3>Platform study minutes</h3>
          <p className="d-hint">Total volume of recorded study minutes.</p>
        </div>
        <MinutesChart data={data.trend} />
      </div>

      <div className="d-card">
        <div className="d-card-header">
          <h3>Platform distraction trend</h3>
          <p className="d-hint">Overall distraction event frequency across all sessions.</p>
        </div>
        <DistractionChart data={data.trend} />
      </div>

      <div className="d-card">
        <div className="d-card-header">
          <h3>System Health & Infrastructure</h3>
          <p className="d-hint">Core database connectivity and ML model operational status.</p>
        </div>
        <div style={{ display: "flex", gap: "24px", flexWrap: "wrap", marginTop: "12px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", fontSize: "14px" }}>
            <span style={{ color: "var(--muted)" }}>Database:</span>
            <span className={`d-badge ${data.system.database === "ok" ? "allow" : "block"}`}>
              {data.system.database}
            </span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", fontSize: "14px" }}>
            <span style={{ color: "var(--muted)" }}>Focus prediction model:</span>
            <span className={`d-badge ${data.system.focus_model_trained ? "allow" : "warn"}`}>
              {data.system.focus_model_trained ? "Trained & Active" : "Untrained"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}