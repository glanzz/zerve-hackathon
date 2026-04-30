"""
src/api/app.py — FastAPI application factory.

Usage:
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

Or run directly:
  python src/api/app.py
"""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from src.api.middleware import register_middleware
from src.api.routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    _app = FastAPI(
        title="User Intelligence API",
        description=(
            "Predicts 7-day subscription upgrade probability for Zerve users. "
            "Powered by LightGBM trained on behavioural event features. "
            "OOF AUC ≈ 0.83 | Test AUC ≈ 0.90"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Middleware (CORS, timing, global exception handler)
    register_middleware(_app)

    # Routes
    _app.include_router(router)

    return _app


# Module-level app instance (used by uvicorn)
app = create_app()


if __name__ == "__main__":
    from config.config import CFG
    uvicorn.run(
        "src.api.app:app",
        host=CFG.API_HOST,
        port=CFG.API_PORT,
        reload=False,
        log_level="info",
    )
