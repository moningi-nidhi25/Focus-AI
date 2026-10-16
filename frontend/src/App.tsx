import { useEffect, useState } from "react";
import { api } from "./api";
import type { EngineInfo } from "./api";
import { auth, clearToken, getToken } from "./dashApi";
import type { AuthUser } from "./dashApi";
import Analyzer from "./components/Analyzer";
import SessionSetup from "./components/SessionSetup";
import AuthScreen from "./components/AuthScreen";
import StudentDashboard from "./components/StudentDashboard";
import TeacherDashboard from "./components/TeacherDashboard";
import AdminDashboard from "./components/AdminDashboard";
import "./App.css";
import "./dashboard.css";

export interface ActiveSession { sessionId: string; topic: string; objectives: string; keywords: string[] }

const STORAGE_KEY = "focusai_active_session";

function loadActive(): ActiveSession | null {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null");
    return parsed?.sessionId ? { objectives: "", keywords: [], ...parsed } : null;
  } catch {
    return null;
  }
}

// Forget the local study session (and tell the extension) so the next person to log in starts clean.
function endLocalSession() {
  localStorage.removeItem(STORAGE_KEY);
  window.postMessage({ type: "FOCUSAI_SESSION_ENDED" }, window.location.origin);
}

function BackendStatus() {
  const [text, setText] = useState("Connecting to backend...");
  const [online, setOnline] = useState(false);

  useEffect(() => {
    api.health()
      .then((d) => { setText(`${d.service} is ${d.status}`); setOnline(true); })
      .catch(() => { setText("Backend unreachable"); setOnline(false); });
  }, []);

  return (
    <span className={`status ${online ? "up" : "down"}`}>
      <span className="status-dot" />
      {text}
    </span>
  );
}

function StudentShell({ user }: { user: AuthUser }) {
  const [tab, setTab] = useState<"session" | "analyze" | "dashboard">("session");
  const [active, setActive] = useState<ActiveSession | null>(loadActive);
  const [engine, setEngine] = useState<EngineInfo | null>(null);

  useEffect(() => {
    api.engine().then(setEngine).catch(() => setEngine(null));
  }, []);

  // Keep the Chrome extension in sync with the active study session.
  useEffect(() => {
    if (active) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(active));
      window.postMessage({ type: "FOCUSAI_SESSION", sessionId: active.sessionId, topic: active.topic }, window.location.origin);
    } else {
      localStorage.removeItem(STORAGE_KEY);
      window.postMessage({ type: "FOCUSAI_SESSION_ENDED" }, window.location.origin);
    }
  }, [active]);

  return (
    <>
      {engine && (
        <div className="engine-banner">
          <span>⚙️ Embeddings: <code>{engine.embedding_backend}</code></span>
          <span>·</span>
          <span>AI Reviewer: <strong>{engine.llm_enabled ? "Enabled" : "Disabled"}</strong></span>
          <span>·</span>
          <span>Thresholds: Allow ≥ {engine.allow_threshold}%, Warn ≥ {engine.warn_threshold}%</span>
        </div>
      )}
      <nav className="tabs" role="tablist">
        <button
          className={tab === "session" ? "active" : ""}
          onClick={() => setTab("session")}
          role="tab"
          aria-selected={tab === "session"}
        >
          Study session
        </button>
        <button
          className={tab === "analyze" ? "active" : ""}
          onClick={() => setTab("analyze")}
          role="tab"
          aria-selected={tab === "analyze"}
        >
          Analyze content
        </button>
        <button
          className={tab === "dashboard" ? "active" : ""}
          onClick={() => setTab("dashboard")}
          role="tab"
          aria-selected={tab === "dashboard"}
        >
          My dashboard
        </button>
      </nav>
      {tab === "session" ? (
        <SessionSetup student={user} active={active} onStart={setActive} onEnd={() => setActive(null)} />
      ) : tab === "analyze" ? (
        <Analyzer active={active} />
      ) : (
        <StudentDashboard />
      )}
    </>
  );
}

function App() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [checking, setChecking] = useState<boolean>(() => getToken() !== null);

  useEffect(() => {
    if (getToken() === null) return;
    auth.me()
      .then(setUser)
      .catch(() => clearToken())
      .finally(() => setChecking(false));
  }, []);

  useEffect(() => {
    const onLogout = () => { clearToken(); endLocalSession(); setUser(null); };
    window.addEventListener("focusai-logout", onLogout);
    return () => window.removeEventListener("focusai-logout", onLogout);
  }, []);

  function logout() {
    clearToken();
    endLocalSession();
    setUser(null);
  }

  return (
    <div className="app">
      <header>
        <div className="brand-wrapper">
          <div className="brand-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <circle cx="12" cy="12" r="4" />
              <line x1="21.17" y1="8" x2="12" y2="8" />
              <line x1="3.95" y1="6.06" x2="8.54" y2="14" />
              <line x1="10.88" y1="21.94" x2="15.46" y2="14" />
            </svg>
          </div>
          <h1>FocusAI</h1>
          <BackendStatus />
        </div>
        {user && (
          <div className="d-userbar">
            <div className="d-user-info">
              <div className="d-avatar">{user.full_name ? user.full_name.charAt(0).toUpperCase() : "U"}</div>
              <span>{user.full_name}</span>
              <em className="d-badge">{user.role}</em>
            </div>
            <button className="d-btn ghost" onClick={logout}>Log out</button>
          </div>
        )}
      </header>
      {checking ? (
        <div className="d-wrap">
          <div className="d-card" style={{ textAlign: "center", padding: "40px" }}>
            <p className="d-hint">Verifying authentication session...</p>
          </div>
        </div>
      ) : !user ? (
        <AuthScreen onAuth={setUser} />
      ) : user.role === "ADMIN" ? (
        <AdminDashboard />
      ) : user.role === "TEACHER" ? (
        <TeacherDashboard />
      ) : (
        <StudentShell user={user} />
      )}
    </div>
  );
}

export default App;