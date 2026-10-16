import { useState } from "react";
import { api } from "../api";
import type { AnalyzeResponse } from "../api";
import type { ActiveSession } from "../App";
import Heatmap from "./Heatmap";

const SIGNAL_LABELS: Record<string, string> = {
  chunk_similarity: "Best-passage similarity",
  embedding_similarity: "Whole-page similarity",
  keyword_coverage: "Key-term coverage",
  title_similarity: "Title similarity",
  llm_score: "AI reviewer score",
};

export default function Analyzer({ active }: { active: ActiveSession | null }) {
  const [topic, setTopic] = useState("");
  const [objectives, setObjectives] = useState("");
  const [mode, setMode] = useState<"text" | "url">("text");
  const [url, setUrl] = useState("");
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  const ready = (active || topic.trim()) && (mode === "url" ? url.trim() : text.trim());

  async function onFile(file: File | undefined) {
    if (!file) return;
    setError("");
    try {
      const ex = await api.extractFile(file);
      if (ex.error) throw new Error(ex.error);
      setMode("text");
      setTitle(ex.title);
      setText(ex.cleaned_text);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function run() {
    setBusy(true);
    setError("");
    try {
      const ctx = active ? { session_id: active.sessionId } : { topic, learning_objectives: objectives };
      const body = mode === "url" ? { ...ctx, url: url.trim() } : { ...ctx, title, text };
      setResult(await api.analyze(body));
    } catch (e) {
      setResult(null);
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      <section className="panel">
        <h2>Analyze Content</h2>
        {active ? (
          <div className="d-hint" style={{ marginBottom: "14px", fontSize: "13px" }}>
            Evaluating against your active session: <strong style={{ color: "var(--accent)" }}>{active.topic}</strong>
          </div>
        ) : (
          <div style={{ marginBottom: "16px" }}>
            <p className="muted small">No active study session. Running a standalone test without saving to session history.</p>
            <label htmlFor="analyzer-topic">Study topic *</label>
            <input
              id="analyzer-topic"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g. Operating System Deadlocks"
            />
            <label htmlFor="analyzer-objectives">Learning objectives (optional)</label>
            <input
              id="analyzer-objectives"
              value={objectives}
              onChange={(e) => setObjectives(e.target.value)}
              placeholder="e.g. Understand Banker's algorithm and resource allocation"
            />
          </div>
        )}

        <div className="tabs small" role="tablist">
          <button
            className={mode === "text" ? "active" : ""}
            onClick={() => setMode("text")}
            role="tab"
            aria-selected={mode === "text"}
          >
            Paste text or document
          </button>
          <button
            className={mode === "url" ? "active" : ""}
            onClick={() => setMode("url")}
            role="tab"
            aria-selected={mode === "url"}
          >
            Web page / YouTube URL
          </button>
        </div>

        {mode === "url" ? (
          <div>
            <label htmlFor="analyzer-url">Target URL</label>
            <input
              id="analyzer-url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://en.wikipedia.org/wiki/Banker%27s_algorithm or YouTube watch URL"
            />
          </div>
        ) : (
          <div>
            <label htmlFor="analyzer-title">Content title (optional)</label>
            <input
              id="analyzer-title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. Chapter 7: Deadlock Mitigation"
            />
            <label htmlFor="analyzer-text">Content text</label>
            <textarea
              id="analyzer-text"
              rows={7}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste article text, notes, or lecture transcript..."
            />
            <label htmlFor="analyzer-file">...or upload a document (PDF, DOCX, TXT, MD)</label>
            <input
              id="analyzer-file"
              type="file"
              accept=".pdf,.docx,.txt,.md"
              onChange={(e) => onFile(e.target.files?.[0])}
            />
          </div>
        )}

        <div style={{ marginTop: "16px" }}>
          <button className="primary" disabled={!ready || busy} onClick={run}>
            {busy ? "Analyzing content..." : "Analyze relevance"}
          </button>
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      {result && (
        <section className="panel">
          <div className="result-head">
            <span className={`badge ${result.decision}`}>{result.decision}</span>
            <span className="big">{Math.round(result.relevance_score)}% relevant</span>
            <span className="muted" style={{ textTransform: "capitalize" }}>
              · {result.classification.replace(/_/g, " ").toLowerCase()}
            </span>
          </div>

          <div className="bar" aria-label={`Relevance score: ${Math.round(result.relevance_score)}%`}>
            <div className={`fill ${result.decision}`} style={{ width: `${result.relevance_score}%` }} />
          </div>

          <p style={{ fontSize: "14px", lineHeight: "1.6", color: "var(--text)" }}>
            {result.explanation}
          </p>

          {result.matching_concepts.length > 0 && (
            <>
              <h3>Matching concepts</h3>
              <div className="chips">
                {result.matching_concepts.map((c) => (
                  <span key={c} className="chip">{c}</span>
                ))}
              </div>
            </>
          )}

          <h3>Decision rationale</h3>
          <ul style={{ margin: "8px 0 0 18px", padding: 0, fontSize: "13px", color: "var(--text-secondary)", lineHeight: "1.6" }}>
            {result.reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>

          <h3>Signals breakdown</h3>
          <div className="signals">
            {Object.entries(SIGNAL_LABELS).map(([k, label]) => {
              const v = result.signals[k];
              return v == null ? null : (
                <div key={k}>
                  <span>{label}</span>
                  <div className="bar thin">
                    <div className="fill neutral" style={{ width: `${Math.max(0, Math.min(100, v))}%` }} />
                  </div>
                  <b>{Math.round(v)}%</b>
                </div>
              );
            })}
          </div>

          <h3>Sentence-level relevance heatmap</h3>
          <Heatmap sentences={result.heatmap} method={result.explanation_method} />

          <p className="muted small" style={{ marginTop: "14px" }}>
            {result.title && <>"{result.title}" · </>}
            {result.word_count} words analyzed · Embedding: <code>{result.embedding_backend}</code>
            {result.llm_used ? " + AI Reviewer" : ""}
          </p>
        </section>
      )}
    </div>
  );
}
