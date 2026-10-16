import { API_URL } from "./api";

export type Role = "STUDENT" | "TEACHER" | "ADMIN";
export interface AuthUser { id: string; email: string; full_name: string; role: Role }
export interface TrendPoint { date: string; avg_focus: number | null; study_minutes: number; distractions: number }
export interface Summary {
  students: number; snapshots: number; avg_focus: number | null; avg_productivity: number | null;
  study_minutes: number; distractions: number; tab_switches: number;
}
export interface CurvePoint { window: number; predicted: number; lower: number; upper: number }
export interface Prediction {
  session_id: string; current_focus: number; curve: CurvePoint[]; alert: boolean;
  alert_window: number | null; message: string; history_points: number; model_note: string;
}
export interface StudentOverview {
  days: number; summary: Summary; blocked_count: number; focus_change_vs_previous_week: number | null;
  trend: TrendPoint[];
  recent_sessions: { id: string; title: string; status: string; start_time: string; avg_focus: number | null }[];
  relevance_history: { title: string | null; url: string; relevance_score: number; decision: string; analyzed_at: string }[];
  focus_decay: { created_at: string; curve: CurvePoint[] } | null;
  insights: string[];
}
interface CohortBase { cohort_id: string; name: string; description: string | null; student_count: number; days: number }
export interface CohortSuppressed extends CohortBase { suppressed: true; message: string }
export interface CohortFull extends CohortBase {
  suppressed: false; active_students_7d: number; summary: Summary;
  avg_study_minutes_per_student: number | null; focus_change_vs_previous_week: number | null;
  trend: TrendPoint[]; focus_distribution: { range: string; students: number }[];
}
export type CohortAnalytics = CohortSuppressed | CohortFull;
export interface AdminOverview {
  days: number; total_students: number; total_teachers: number; active_teachers: number;
  total_cohorts: number; total_sessions: number; active_students_7d: number;
  platform_study_hours_all_time: number; summary: Summary; trend: TrendPoint[];
  system: { database: string; focus_model_trained: boolean };
}

const TOKEN_KEY = "focusai_token";
export const getToken = (): string | null => {
  try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
};
// Tell the Chrome extension (via sync.js) which login token to use.
const announceToken = () =>
  window.postMessage({ type: "FOCUSAI_TOKEN", token: getToken() ?? "" }, window.location.origin);
const setToken = (t: string) => { localStorage.setItem(TOKEN_KEY, t); announceToken(); };
export const clearToken = () => { localStorage.removeItem(TOKEN_KEY); announceToken(); };

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { ...(init.headers as Record<string, string> | undefined) };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  if (init.body && !headers["Content-Type"]) headers["Content-Type"] = "application/json";
  const res = await fetch(`${API_URL}/api${path}`, { ...init, headers });
  if (res.status === 401 && token) {
    clearToken();
    window.dispatchEvent(new Event("focusai-logout"));
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string"
        ? body.detail
        : Array.isArray(body.detail)
          ? body.detail.map((d: { msg?: string }) => d.msg).join("; ")
          : JSON.stringify(body.detail);
    } catch { /* keep status text */ }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

interface TokenResponse { access_token: string; user: AuthUser }

export const auth = {
  login: async (email: string, password: string): Promise<AuthUser> => {
    const r = await call<TokenResponse>("/auth/login", { method: "POST", body: JSON.stringify({ email, password }) });
    setToken(r.access_token);
    return r.user;
  },
  register: async (email: string, full_name: string, password: string, role: "STUDENT" | "TEACHER"): Promise<AuthUser> => {
    const r = await call<TokenResponse>("/auth/register", {
      method: "POST", body: JSON.stringify({ email, full_name, password, role }),
    });
    setToken(r.access_token);
    return r.user;
  },
  me: () => call<AuthUser>("/auth/me"),
};

export const dash = {
  student: (days: number) => call<StudentOverview>(`/dashboard/student/overview?days=${days}`),
  predict: (sessionId: string) => call<Prediction>(`/prediction/sessions/${sessionId}`, { method: "POST" }),
  cohorts: (days: number) => call<CohortAnalytics[]>(`/dashboard/teacher/cohorts?days=${days}`),
  createCohort: (name: string, description: string) =>
    call<{ id: string }>("/dashboard/teacher/cohorts", {
      method: "POST", body: JSON.stringify({ name, description: description || null }),
    }),
  addStudent: (cohortId: string, email: string) =>
    call<{ student_count: number }>(`/dashboard/teacher/cohorts/${cohortId}/students`, {
      method: "POST", body: JSON.stringify({ email }),
    }),
  admin: (days: number) => call<AdminOverview>(`/dashboard/admin/overview?days=${days}`),
};

export const fmtMinutes = (m: number): string =>
  m >= 60 ? `${Math.floor(m / 60)}h ${Math.round(m % 60)}m` : `${Math.round(m)}m`;
export const fmtDate = (iso: string): string =>
  new Date(/(Z|[+-]\d{2}:\d{2})$/i.test(iso) ? iso : iso + "Z").toLocaleString();
export const fmtChange = (v: number | null): string | undefined =>
  v === null ? undefined : `${v >= 0 ? "▲" : "▼"} ${Math.abs(v)} pts vs previous week`;