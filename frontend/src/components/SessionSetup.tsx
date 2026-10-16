import { useEffect, useState } from "react";
import { api } from "../api";
import type { HistoryItem, StudySession } from "../api";
import type { AuthUser } from "../dashApi";
import type { ActiveSession } from "../App";

interface Props {
  student: AuthUser;
  active: ActiveSession | null;
  onStart: (s: ActiveSession) => void;
  onEnd: () => void;
}

const fromSession = (s: StudySession): ActiveSession => ({
  sessionId: s.id,
  topic: s.study_context?.topic ?? s.title,
  objectives: s.study_context?.learning_objectives ?? "",
  keywords: s.study_context?.keywords ?? [],
});

export default function SessionSetup({ student, active, onStart, onEnd }: Props) {
  const [recent, setRecent] = useState<StudySession[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [title, setTitle] = useState("");
  const [topic, setTopic] = useState("");
  const [objectives, setObjectives] = useState("");
  const [material, setMaterial] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api.sessions(student.id).then(setRecent).catch(() => setRecent([]));
  }, [student.id, active]);

  const sessionId = active?.sessionId;
  const refreshHistory = () => { if (sessionId) api.history(sessionId).then(setHistory).catch(() => setHistory([])); };
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    api.history(sessionId).then((h) => { if (!cancelled) setHistory(h); }).catch(() => undefined);
    return () => { cancelled = true; };
  }, [sessionId]);

  async function start() {
    setBusy(true); setError("");
    try {
      const created = await api.createSession({
        student_id: student.id, title: title.trim() || topic.trim(), topic: topic.trim(), learning_objectives: objectives.trim() || undefined,
      });
      let ctx = created.study_context;
      if (material) ctx = await api.uploadMaterial(created.id, material);
      onStart(fromSession({ ...created, study_context: ctx }));
      setTitle(""); setTopic(""); setObjectives(""); setMaterial(null);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }

  async function end() {
    if (!active) return;
    try { await api.endSession(active.sessionId); } catch (e) { setError((e as Error).message); return; }
    onEnd();
  }

  return (
    <div className="stack">
      {active && (
        <section className="panel highlight">
          <div className="result-head">
            <span className="badge ALLOW">Active Study Goal</span>
            <h2>{active.topic}</h2>
          </div>
          {active.objectives && (
            <p style={{ color: "var(--text-secondary)", fontSize: "14px", lineHeight: "1.5" }}>
              {active.objectives}
            </p>
          )}

          <h3>Key terms FocusAI is monitoring for</h3>
          <div className="chips">
            {active.keywords && active.keywords.length > 0 ? (
              active.keywords.map((k) => <span key={k} className="chip">{k}</span>)
            ) : (
              <span className="muted small">No specific keywords extracted. Whole-topic embeddings active.</span>
            )}
          </div>

          <p className="muted small" style={{ marginTop: "12px" }}>
            Session ID: <code>{active.sessionId}</code> · Connected Chrome extensions sync with this session automatically.
          </p>

          <div className="row" style={{ marginTop: "16px" }}>
            <button onClick={refreshHistory}>Refresh decisions</button>
            <button className="danger" onClick={end}>End session</button>
          </div>

          {history.length > 0 && (
            <div className="table-responsive" style={{ marginTop: "16px" }}>
              <table>
                <thead>
                  <tr>
                    <th>Decision</th>
                    <th>Relevance</th>
                    <th>Page / Source</th>
                  </tr>
                </thead>
                <tbody>
                  {history.slice(0, 10).map((h) => (
                    <tr key={h.id}>
                      <td><span className={`badge ${h.decision}`}>{h.decision}</span></td>
                      <td style={{ fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>{Math.round(h.relevance_score)}%</td>
                      <td className="ellipsis" title={h.url}>{h.title || h.url}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      <section className="panel">
        <h2>Start a Study Session</h2>
        <p className="muted small">Configuring session for <strong>{student.full_name}</strong> ({student.email})</p>

        <label htmlFor="topic-input">Study topic *</label>
        <input
          id="topic-input"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="e.g. Operating System Deadlocks"
        />

        <label htmlFor="obj-input">Learning objectives (optional)</label>
        <textarea
          id="obj-input"
          rows={3}
          value={objectives}
          onChange={(e) => setObjectives(e.target.value)}
          placeholder="e.g. Understand Banker's algorithm, Coffman conditions, and resource allocation graphs"
        />

        <label htmlFor="title-input">Session custom title (optional)</label>
        <input
          id="title-input"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Defaults to the study topic if left blank"
        />

        <label htmlFor="material-input">Course material syllabus or notes (optional PDF, DOCX, TXT, MD)</label>
        <input
          id="material-input"
          type="file"
          accept=".pdf,.docx,.txt,.md"
          onChange={(e) => setMaterial(e.target.files?.[0] ?? null)}
        />

        <div style={{ marginTop: "16px" }}>
          <button className="primary" disabled={!topic.trim() || busy} onClick={start}>
            {busy ? "Starting session..." : "Start study session"}
          </button>
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      {recent.filter((s) => s.id !== active?.sessionId).length > 0 && (
        <section className="panel">
          <h2>Recent Sessions</h2>
          <ul className="plain">
            {recent.filter((s) => s.id !== active?.sessionId).slice(0, 6).map((s) => (
              <li key={s.id}>
                <div>
                  <strong style={{ color: "var(--text)" }}>{s.title}</strong>
                  <span className="muted" style={{ fontSize: "13px", marginLeft: "8px" }}>· {s.status}</span>
                </div>
                <button className="ghost" onClick={() => onStart(fromSession(s))}>
                  Resume session
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}