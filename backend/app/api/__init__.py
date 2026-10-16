from fastapi import APIRouter
from app.api.study_sessions import router as sessions_router
from app.api.content_extraction import router as content_router
from app.api.relevance import router as relevance_router
from app.api.users import router as users_router
from app.api.activity import activity_router, analytics_router
from app.api.prediction import router as prediction_router
from app.api.auth import router as auth_router
from app.api.dashboard import router as dashboard_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(users_router)
api_router.include_router(sessions_router)
api_router.include_router(content_router)
api_router.include_router(relevance_router)
api_router.include_router(activity_router)
api_router.include_router(analytics_router)
api_router.include_router(prediction_router)
api_router.include_router(dashboard_router)

__all__ = ["api_router"]