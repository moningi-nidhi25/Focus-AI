import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Ensure .env is loaded before anything reads configuration
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from app.api import api_router  # noqa: E402
from app.models import StudySession, User  # noqa: E402
from app.utils.database import get_db, init_db  # noqa: E402


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize database tables (and add any new columns) on startup."""
    init_db()
    yield


app = FastAPI(title=os.getenv("APP_NAME", "FocusAI API"), lifespan=lifespan)

# The React dev server and the Chrome extension (chrome-extension://<id>) call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"^chrome-extension://[a-z]{32}$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/")
def root():
    return {"message": "FocusAI Backend is running!"}


@app.get("/api/health")
def health_check(db: Session = Depends(get_db)):
    db_status, users_count, sessions_count = "connected", 0, 0
    try:
        users_count = db.query(User).count()
        sessions_count = db.query(StudySession).count()
    except Exception as e:
        db_status = f"error: {e}"
    return {
        "status": "healthy",
        "service": "FocusAI Backend",
        "database": {"status": db_status, "total_users": users_count, "total_study_sessions": sessions_count},
    }
