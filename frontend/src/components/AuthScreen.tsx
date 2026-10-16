import { useState } from "react";
import type { FormEvent } from "react";
import { auth } from "../dashApi";
import type { AuthUser } from "../dashApi";

export default function AuthScreen({ onAuth }: { onAuth: (u: AuthUser) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [role, setRole] = useState<"STUDENT" | "TEACHER">("STUDENT");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const user = mode === "login"
        ? await auth.login(email.trim(), password)
        : await auth.register(email.trim(), name.trim(), password, role);
      onAuth(user);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-wrapper">
      <form className="d-card auth-card" onSubmit={submit}>
        <div className="auth-header">
          <div className="brand-icon" style={{ margin: "0 auto 12px", width: "42px", height: "42px" }}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            </svg>
          </div>
          <h3>{mode === "login" ? "Welcome back to FocusAI" : "Create your account"}</h3>
          <p>{mode === "login" ? "Log in to track your study sessions and focus metrics." : "Sign up to start studying smarter with AI-powered focus tracking."}</p>
        </div>

        {mode === "register" && (
          <div>
            <label htmlFor="auth-name">Full name</label>
            <input
              id="auth-name"
              className="d-input"
              placeholder="e.g. Alex Johnson"
              value={name}
              required
              autoComplete="name"
              onChange={(e) => setName(e.target.value)}
            />
          </div>
        )}

        <div>
          <label htmlFor="auth-email">Email address</label>
          <input
            id="auth-email"
            className="d-input"
            type="email"
            placeholder="you@university.edu"
            value={email}
            required
            autoComplete="email"
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>

        <div>
          <label htmlFor="auth-password">Password</label>
          <input
            id="auth-password"
            className="d-input"
            type="password"
            placeholder="At least 8 characters"
            value={password}
            required
            minLength={8}
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>

        {mode === "register" && (
          <div>
            <label htmlFor="auth-role">Account role</label>
            <select
              id="auth-role"
              className="d-input"
              value={role}
              onChange={(e) => setRole(e.target.value as "STUDENT" | "TEACHER")}
            >
              <option value="STUDENT">Student (Self-study & session focus tracking)</option>
              <option value="TEACHER">Teacher / Instructor (Cohort management)</option>
            </select>
          </div>
        )}

        {error && <div className="d-error">{error}</div>}

        <button className="d-btn" disabled={busy} type="submit" style={{ marginTop: "8px", width: "100%", height: "42px" }}>
          {busy ? "Please wait..." : mode === "login" ? "Log in" : "Create account"}
        </button>

        <button
          className="d-btn ghost"
          type="button"
          onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }}
          style={{ width: "100%" }}
        >
          {mode === "login" ? "Need an account? Sign up" : "Already have an account? Log in"}
        </button>

        <p className="d-hint" style={{ textAlign: "center", marginTop: "12px" }}>
          Administrator accounts are provisioned directly by system administrators.
        </p>
      </form>
    </div>
  );
}