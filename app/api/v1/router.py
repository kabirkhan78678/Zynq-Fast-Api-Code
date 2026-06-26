from fastapi import APIRouter
from app.api.v1.endpoints import search, tags

api_router = APIRouter()

# Grouping all endpoint routers under unified prefix tags
api_router.include_router(search.router, prefix="/search", tags=["Search"])
api_router.include_router(tags.router, prefix="/tags", tags=["Tags"])
