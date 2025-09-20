"""
Main FastAPI application.
Entry point for the API server.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time

from app.core import settings, setup_logging
from app.models.database import init_db
from app.api import router

# Setup logging
logger = setup_logging(
    level="DEBUG" if settings.debug else "INFO", log_file="logs/app.log"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle management.
    Runs on startup and shutdown.
    """
    # Startup
    logger.info("Starting Coin Detection API")
    init_db()  # Create database tables
    logger.info("Database initialized")

    yield  # Application runs

    # Shutdown
    logger.info("Shutting down Coin Detection API")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API for detecting circular objects (coins) in images",
    lifespan=lifespan,
)

# Add CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Include API routes
app.include_router(router, prefix=settings.api_prefix, tags=["coins"])


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.app_version}


# Root endpoint
@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
