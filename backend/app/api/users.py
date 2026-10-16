from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.models.user import User, UserRole
from app.schemas.user import UserCreate, UserRead
from app.utils.database import get_db

# Only administrators may list users or create accounts here. Everyone else signs up via /auth/register.
router = APIRouter(prefix="/users", tags=["Users"], dependencies=[Depends(require_roles(UserRole.ADMIN))])


@router.get("", response_model=List[UserRead])
def list_users(role: Optional[UserRole] = None, db: Session = Depends(get_db)):
    q = db.query(User)
    if role:
        q = q.filter(User.role == role)
    return q.order_by(User.created_at).limit(200).all()


@router.post("", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def create_user(user_in: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user_in.email).first():
        raise HTTPException(status_code=409, detail="A user with this email already exists")
    user = User(**user_in.model_dump())
    db.add(user)
    db.commit()
    db.refresh(user)
    return user