"""Authentication, role-based access control and privacy rules (roadmap Phase 10: RBAC testing)."""


def test_login_required_everywhere(anon):
    assert anon.get("/api/sessions").status_code == 401
    assert anon.get("/api/users").status_code == 401
    assert anon.get("/api/dashboard/student/overview").status_code == 401
    assert anon.post("/api/relevance/analyze", json={}).status_code == 401
    assert anon.post("/api/activity/events", json={}).status_code == 401
    assert anon.post("/api/content/extract", json={}).status_code == 401


def test_wrong_password_and_admin_signup_rejected(client, make_user):
    user, _ = make_user("STUDENT")
    r = client.post("/api/auth/login", json={"email": user["email"], "password": "wrong-password"})
    assert r.status_code == 401
    r = client.post("/api/auth/register", json={
        "email": "evil@test.edu", "full_name": "Evil", "password": "TestPass123", "role": "ADMIN"})
    assert r.status_code == 422  # admins can only be created by script


def test_student_cannot_use_admin_or_teacher_apis(client):
    assert client.get("/api/users").status_code == 403
    assert client.get("/api/dashboard/admin/overview").status_code == 403
    assert client.get("/api/dashboard/teacher/cohorts").status_code == 403


def test_teacher_cannot_read_student_sessions(client, make_user):
    _, teacher = make_user("TEACHER")
    assert client.get("/api/sessions", headers=teacher).status_code == 403
    assert client.get("/api/dashboard/student/overview", headers=teacher).status_code == 403


def test_students_cannot_touch_each_others_sessions(client, session, make_user):
    _, other = make_user("STUDENT")
    sid = session["id"]
    for path in (f"/api/sessions/{sid}", f"/api/analytics/sessions/{sid}/summary",
                 f"/api/relevance/sessions/{sid}/history"):
        assert client.get(path, headers=other).status_code == 404, path
    assert client.post(f"/api/prediction/sessions/{sid}", headers=other).status_code == 404
    r = client.post("/api/activity/events", headers=other,
                    json={"session_id": sid, "events": [{"event_type": "click", "metadata": {}}]})
    assert r.status_code == 404
    r = client.post("/api/sessions", headers=other,
                    json={"student_id": session["student_id"], "title": "x", "topic": "x"})
    assert r.status_code == 403  # cannot create a session for someone else


def test_small_cohort_is_suppressed_and_anonymous(client, student, make_user):
    _, teacher = make_user("TEACHER")
    r = client.post("/api/dashboard/teacher/cohorts", headers=teacher, json={"name": "Tiny", "description": None})
    assert r.status_code == 201
    cid = r.json()["id"]
    r = client.post(f"/api/dashboard/teacher/cohorts/{cid}/students", headers=teacher, json={"email": student["email"]})
    assert r.status_code == 200 and r.json()["student_count"] == 1

    rows = client.get("/api/dashboard/teacher/cohorts", headers=teacher).json()
    tiny = next(c for c in rows if c["cohort_id"] == cid)
    assert tiny["suppressed"] is True  # fewer than 3 students: no analytics at all
    assert student["email"] not in str(tiny) and student["full_name"] not in str(tiny)


def test_teacher_cannot_see_another_teachers_cohort(client, make_user):
    _, t1 = make_user("TEACHER")
    _, t2 = make_user("TEACHER")
    cid = client.post("/api/dashboard/teacher/cohorts", headers=t1, json={"name": "Mine"}).json()["id"]
    assert client.get(f"/api/dashboard/teacher/cohorts/{cid}/analytics", headers=t2).status_code == 404