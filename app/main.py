from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.router import api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup CORS middleware for local development/external callers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this configuration for production environments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bind the aggregated v1 router under the configured prefix path
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["General"], summary="Root Health Check")
def root():
    """Welcome and API root status check."""
    return {
        "message": f"Welcome to the {settings.PROJECT_NAME}!",
        "status": "healthy",
        "docs": "/docs"
    }
