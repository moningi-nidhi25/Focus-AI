export const API_URL: string = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export interface User { id: string; email: string; full_name: string; role: string }
export interface StudyContext {
  id: string; session_id: string; topic: string; learning_objectives: string | null;
  keywords: string[]; has_course_material: boolean; has_embedding: boolean;
}
export interface StudySession {
  id: string; student_id: string; title: string; status: string; created_at: string;
  study_context: StudyContext | null;
}
export interface Sentence { text: string; score: number; level: "high" | "medium" | "low" }
export type Decision = "ALLOW" | "WARN" | "BLOCK";
export interface AnalyzeResponse {
  analysis_id: string | null; url: string | null; title: string; source_type: string;
  relevance_score: number; classification: string; decision: Decision;
  explanation: string; reasons: string[]; matching_concepts: string[];
  highlighted_sentences: Sentence[]; heatmap: Sentence[];
  signals: Record<string, number | null>;
  explanation_method: string; embedding_backend: string; llm_used: boolean;
  low_content: boolean; insufficient_content: boolean; word_count: number;
}
export interface EngineInfo {
  embedding_backend: string; dense_embeddings: boolean; llm_enabled: boolean;
  allow_threshold: number; warn_threshold: number; explanation_method: string;
}
export interface HistoryItem {
  id: string; url: string; title: string | null; relevance_score: number;
  classification: string | null; decision: Decision; analyzed_at: string;
}
export interface Extraction { title: string; cleaned_text: string; word_count: number; error?: string | null }
export interface AnalyzeRequest {
  session_id?: string; topic?: string; learning_objectives?: string;
  url?: string; title?: string; text?: string;
}

function authHeader(): Record<string, string> {
  try {
    const t = localStorage.getItem("focusai_token");
    return t ? { Authorization: `Bearer ${t}` } : {};
  } catch {
    return {};
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const hadToken = "Authorization" in authHeader();
  const res = await fetch(`${API_URL}/api${path}`, {
    ...init,
    headers: { ...authHeader(), ...((init?.headers as Record<string, string> | undefined) ?? {}) },
  });
  if (res.status === 401 && hadToken) {
    localStorage.removeItem("focusai_token");
    window.dispatchEvent(new Event("focusai-logout"));
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* keep status text */ }
    throw new Error(detail);
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

const json = (method: string, body: unknown): RequestInit => ({
  method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

function upload<T>(path: string, file: File): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  return request<T>(path, { method: "POST", body: form });
}

export const api = {
  health: () => request<{ status: string; service: string }>("/health"),
  engine: () => request<EngineInfo>("/relevance/engine"),
  students: () => request<User[]>("/users?role=STUDENT"),
  createStudent: (email: string, full_name: string) => request<User>("/users", json("POST", { email, full_name, role: "STUDENT" })),
  sessions: (studentId: string) => request<StudySession[]>(`/sessions?student_id=${studentId}`),
  createSession: (b: { student_id: string; title: string; topic: string; learning_objectives?: string }) =>
    request<StudySession>("/sessions", json("POST", b)),
  endSession: (id: string) => request<StudySession>(`/sessions/${id}`, json("PATCH", { status: "completed" })),
  uploadMaterial: (id: string, file: File) => upload<StudyContext>(`/sessions/${id}/context/material`, file),
  context: (id: string) => request<StudyContext>(`/sessions/${id}/context`),
  extractFile: (file: File) => upload<Extraction>("/content/extract/file", file),
  analyze: (b: AnalyzeRequest) => request<AnalyzeResponse>("/relevance/analyze", json("POST", b)),
  history: (id: string) => request<HistoryItem[]>(`/relevance/sessions/${id}/history`),
};
