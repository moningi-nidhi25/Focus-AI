// Runs on the FocusAI web app (localhost:5173): passes the active study session and login token to the extension.
(() => {
  const send = (sessionId, topic) => {
    try { chrome.runtime.sendMessage({ type: "SET_SESSION", sessionId, topic }); } catch (_) { /* extension reloaded */ }
  };
  const sendToken = (token) => {
    try { chrome.runtime.sendMessage({ type: "SET_TOKEN", token }); } catch (_) { /* extension reloaded */ }
  };
  window.addEventListener("message", (e) => {
    if (e.source !== window || !e.data) return;
    if (e.data.type === "FOCUSAI_SESSION") send(e.data.sessionId, e.data.topic);
    if (e.data.type === "FOCUSAI_SESSION_ENDED") send("", "");
    if (e.data.type === "FOCUSAI_TOKEN") sendToken(e.data.token || "");
  });
  try {
    const saved = JSON.parse(localStorage.getItem("focusai_active_session") || "null");
    if (saved && saved.sessionId) send(saved.sessionId, saved.topic);
    const token = localStorage.getItem("focusai_token");
    if (token) sendToken(token);
  } catch (_) { /* ignore */ }
})();