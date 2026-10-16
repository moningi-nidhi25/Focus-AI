from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.models import Cohort, CohortMembership, User, UserRole
from app.schemas.auth import AddStudentRequest, CohortCreate
from app.services import dashboard_service
from app.utils.database import get_db

router = APIRouter(prefix="/dashboard", tags=["Dashboards"])

DAYS = Query(14, ge=7, le=90)


def _own_cohort(db: Session, teacher: User, cohort_id: str) -> Cohort:
    cohort = db.get(Cohort, cohort_id)
    if not cohort or cohort.teacher_id != teacher.id:
        raise HTTPException(status_code=404, detail="Cohort not found")
    return cohort


@router.get("/student/overview")
def student_overview(days: int = DAYS, user: User = Depends(require_roles(UserRole.STUDENT)),
                     db: Session = Depends(get_db)):
    return dashboard_service.student_overview(db, user, days)


@router.get("/teacher/cohorts")
def list_cohorts(days: int = DAYS, teacher: User = Depends(require_roles(UserRole.TEACHER)),
                 db: Session = Depends(get_db)):
    cohorts = db.query(Cohort).filter(Cohort.teacher_id == teacher.id).order_by(Cohort.created_at).all()
    return [dashboard_service.cohort_analytics(db, c, days) for c in cohorts]


@router.post("/teacher/cohorts", status_code=status.HTTP_201_CREATED)
def create_cohort(body: CohortCreate, teacher: User = Depends(require_roles(UserRole.TEACHER)),
                  db: Session = Depends(get_db)):
    cohort = Cohort(name=body.name.strip(), description=body.description, teacher_id=teacher.id)
    db.add(cohort)
    db.commit()
    db.refresh(cohort)
    return {"id": cohort.id, "name": cohort.name, "description": cohort.description}


@router.post("/teacher/cohorts/{cohort_id}/students")
def add_student(cohort_id: str, body: AddStudentRequest, teacher: User = Depends(require_roles(UserRole.TEACHER)),
                db: Session = Depends(get_db)):
    cohort = _own_cohort(db, teacher, cohort_id)
    student = db.query(User).filter(User.email == body.email.lower(), User.role == UserRole.STUDENT).first()
    if not student:
        raise HTTPException(status_code=404, detail="No student account with that email")
    exists = db.query(CohortMembership).filter(CohortMembership.cohort_id == cohort.id,
                                               CohortMembership.student_id == student.id).first()
    if not exists:
        db.add(CohortMembership(cohort_id=cohort.id, student_id=student.id))
        db.commit()
    count = db.query(CohortMembership).filter(CohortMembership.cohort_id == cohort.id).count()
    return {"student_count": count}


@router.get("/teacher/cohorts/{cohort_id}/analytics")
def cohort_analytics(cohort_id: str, days: int = DAYS, teacher: User = Depends(require_roles(UserRole.TEACHER)),
                     db: Session = Depends(get_db)):
    return dashboard_service.cohort_analytics(db, _own_cohort(db, teacher, cohort_id), days)


@router.get("/admin/overview")
def admin_overview(days: int = DAYS, _: User = Depends(require_roles(UserRole.ADMIN)),
                   db: Session = Depends(get_db)):
    return dashboard_service.admin_overview(db, days)