import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.models import StudySession, User, UserRole
from app.utils.database import get_db
from app.utils.security import decode_token

bearer = HTTPBearer(auto_error=False)


def current_user(creds: HTTPAuthorizationCredentials = Depends(bearer), db: Session = Depends(get_db)) -> User:
    unauthorized = HTTPException(status_code=401, detail="Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    if not creds:
        raise unauthorized
    try:
        payload = decode_token(creds.credentials)
    except jwt.PyJWTError:
        raise unauthorized
    user = db.get(User, payload.get("sub"))
    if not user:
        raise unauthorized
    return user


def require_roles(*roles):
    """The role is read from the database, never trusted from the token."""
    def dependency(user: User = Depends(current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="You do not have access to this resource")
        return user
    return dependency


def owned_session_or_404(db: Session, user: User, session_id: str) -> StudySession:
    """A student may only touch their own sessions. Someone else's id looks exactly like a missing one."""
    session = db.get(StudySession, session_id) if session_id else None
    if not session or session.student_id != user.id:
        raise HTTPException(status_code=404, detail="Study session not found")
    return session


def owned_session(session_id: str, user: User = Depends(require_roles(UserRole.STUDENT)),
                  db: Session = Depends(get_db)) -> StudySession:
    """Dependency for routes with {session_id} in the path."""
    return owned_session_or_404(db, user, session_id)