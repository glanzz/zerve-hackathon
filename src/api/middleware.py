"""
src/api/middleware.py — FastAPI middleware stack.

Registers:
  1. CORSMiddleware  — allow all origins (internal/demo use)
  2. X-Process-Time — response header with wall-clock ms
  3. Global exception handler — JSON {error, type} on any uncaught exception
"""
from __future__ import annotations

import time
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


def register_middleware(app: FastAPI) -> None:
    """Attach all middleware and exception handlers to *app*."""

    # ── 1. CORS ───────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── 2. Request timing — adds X-Process-Time header (milliseconds) ─────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        _t0 = time.perf_counter()
        _response = await call_next(request)
        _elapsed_ms = (time.perf_counter() - _t0) * 1000
        _response.headers["X-Process-Time"] = f"{_elapsed_ms:.2f}ms"
        return _response

    # ── 3. Global exception handler ───────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "type": type(exc).__name__,
                "path": str(request.url.path),
            },
        )
