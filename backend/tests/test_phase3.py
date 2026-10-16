"""Phase 3: study context + content extraction."""
import io

from docx import Document

from app.services import content_extractor as ce


def _tiny_pdf(text: str) -> bytes:
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        None,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    objs[3] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


def test_session_creates_structured_context(client, session):
    ctx = session["study_context"]
    assert ctx["topic"] == "Operating System Deadlocks"
    assert "banker algorithm" in ctx["keywords"]
    assert ctx["has_course_material"] is False


def test_context_update_does_not_need_session_id_in_body(client, session):
    r = client.post(f"/api/sessions/{session['id']}/context", json={"topic": "Process Scheduling", "learning_objectives": "Round robin"})
    assert r.status_code == 200
    assert r.json()["topic"] == "Process Scheduling"
    assert any("round" in k for k in r.json()["keywords"])
    assert client.get(f"/api/sessions/{session['id']}/context").json()["topic"] == "Process Scheduling"


def test_unknown_session_404(client):
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.post("/api/sessions/nope/context", json={"topic": "x"}).status_code == 404


def test_completing_session_sets_end_time(client, session):
    r = client.patch(f"/api/sessions/{session['id']}", json={"status": "completed"})
    assert r.status_code == 200 and r.json()["end_time"] is not None


def test_course_material_upload_enriches_context(client, session):
    notes = b"Semaphores and mutexes. Semaphores protect critical sections. Mutexes ensure mutual exclusion. Semaphores count."
    r = client.post(f"/api/sessions/{session['id']}/context/material", files={"file": ("notes.txt", notes, "text/plain")})
    assert r.status_code == 200
    body = r.json()
    assert body["has_course_material"] is True
    assert "semaphores" in body["keywords"]


def test_extract_webpage_html():
    html = "<html><head><title>Raft</title></head><body><nav>Home</nav><article><h1>Raft Consensus</h1><p>Raft is a consensus algorithm designed to be easy to understand.</p></article><script>x()</script></body></html>"
    r = ce.extract_and_normalize(html=html, url="https://raft.github.io/")
    assert r["title"] == "Raft" and r["source_type"] == "webpage"
    assert "Raft Consensus." in r["cleaned_text"] and "x()" not in r["cleaned_text"] and "Home" not in r["cleaned_text"]


def test_extract_text_and_empty():
    assert ce.extract_and_normalize(text="  Deadlock  occurs\n\nwhen...  ", title="Notes")["cleaned_text"] == "Deadlock occurs when..."
    assert ce.extract_and_normalize()["word_count"] == 0


def test_youtube_id_formats():
    assert ce.extract_youtube_id("https://www.youtube.com/watch?v=abc123") == "abc123"
    assert ce.extract_youtube_id("https://youtu.be/abc123") == "abc123"
    assert ce.extract_youtube_id("https://www.youtube.com/shorts/abc123") == "abc123"
    assert ce.extract_youtube_id("https://example.com/watch?v=abc123") is None


def test_youtube_extraction_uses_transcript_and_title(monkeypatch):
    monkeypatch.setattr(ce, "_fetch_transcript", lambda vid: "Today we study deadlock detection in operating systems")
    monkeypatch.setattr(ce, "_youtube_page_meta", lambda vid: {"title": "", "description": "A lecture on deadlocks"})

    class FakeResp:
        status_code = 200
        def json(self): return {"title": "Deadlocks Explained"}

    monkeypatch.setattr(ce.requests, "get", lambda *a, **k: FakeResp())
    r = ce.extract_and_normalize(url="https://www.youtube.com/watch?v=abc123")
    assert r["source_type"] == "youtube" and r["title"] == "Deadlocks Explained"
    assert r["metadata"]["has_transcript"] is True
    assert "deadlock detection" in r["cleaned_text"] and "A lecture on deadlocks" in r["cleaned_text"]


def test_youtube_without_transcript_still_returns_title(monkeypatch):
    def boom(vid): raise RuntimeError("disabled")
    monkeypatch.setattr(ce, "_fetch_transcript", boom)
    monkeypatch.setattr(ce, "_youtube_page_meta", lambda vid: {"title": "Funny cats", "description": ""})
    monkeypatch.setattr(ce.requests, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    r = ce.extract_and_normalize(url="https://youtu.be/abc123")
    assert r["title"] == "Funny cats" and r["metadata"]["has_transcript"] is False


def test_private_urls_blocked():
    r = ce.extract_and_normalize(url="http://127.0.0.1:8000/")
    assert "private" in r["error"].lower() and r["cleaned_text"] == ""


def test_extract_file_docx_and_pdf_and_txt(client):
    d = Document(); d.add_paragraph("Deadlock occurs when processes wait forever."); buf = io.BytesIO(); d.save(buf)
    r = client.post("/api/content/extract/file", files={"file": ("n.docx", buf.getvalue())})
    assert r.status_code == 200 and "Deadlock occurs" in r.json()["cleaned_text"]

    r = client.post("/api/content/extract/file", files={"file": ("n.pdf", _tiny_pdf("Banker algorithm avoids unsafe states"))})
    assert r.status_code == 200 and "Banker algorithm" in r.json()["cleaned_text"]

    r = client.post("/api/content/extract/file", files={"file": ("n.md", b"# Notes\nmutual exclusion")})
    assert r.status_code == 200 and "mutual exclusion" in r.json()["cleaned_text"]


def test_extract_file_rejects_unsupported(client):
    r = client.post("/api/content/extract/file", files={"file": ("virus.exe", b"MZ")})
    assert r.status_code == 400


def test_new_columns_are_added_to_existing_tables(client):
    from sqlalchemy import inspect, text
    from app.utils.database import engine, ensure_columns

    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE study_contexts DROP COLUMN keywords"))
    assert "keywords" not in {c["name"] for c in inspect(engine).get_columns("study_contexts")}
    ensure_columns()
    assert "keywords" in {c["name"] for c in inspect(engine).get_columns("study_contexts")}
