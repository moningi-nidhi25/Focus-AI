import os
import tempfile
import uuid
from pathlib import Path

# Configure BEFORE the app is imported: isolated SQLite DB, offline embeddings, no LLM.
_tmp = tempfile.mkdtemp(prefix="focusai-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_tmp) / 'test.db'}"
os.environ["EMBEDDING_BACKEND"] = "lexical"
os.environ["LLM_API_KEY"] = ""
os.environ["JWT_SECRET"] = "test-only-secret-not-used-anywhere-else-0123456789"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def make_user(client):
    """Register a brand-new user of the given role; returns (user, auth_headers)."""
    def _make(role: str = "STUDENT"):
        email = f"{role.lower()}-{uuid.uuid4().hex[:8]}@test.edu"
        r = client.post("/api/auth/register", json={
            "email": email, "full_name": f"Test {role.title()}", "password": "TestPass123", "role": role,
        })
        assert r.status_code == 201, r.text
        body = r.json()
        return body["user"], {"Authorization": f"Bearer {body['access_token']}"}
    return _make


@pytest.fixture()
def student(client, make_user):
    """A fresh student; the shared client is logged in as them for this test."""
    user, headers = make_user("STUDENT")
    client.headers.update(headers)
    return user


@pytest.fixture(autouse=True)
def _logged_in_student(student):
    return student


@pytest.fixture()
def anon(client):
    """A client with no login at all."""
    return TestClient(app)


@pytest.fixture()
def session(client, student):
    r = client.post(
        "/api/sessions",
        json={
            "student_id": student["id"],
            "title": "Deadlocks",
            "topic": "Operating System Deadlocks",
            "learning_objectives": "Understand Banker's Algorithm and Coffman conditions for deadlock prevention.",
        },
    )
    assert r.status_code == 201
    return r.json()


@pytest.fixture(autouse=True)
def _clear_youtube_cache():
    from app.services.content_extractor import clear_youtube_cache

    clear_youtube_cache()
    yield
    clear_youtube_cache()