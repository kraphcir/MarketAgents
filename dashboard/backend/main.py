"""FastAPI application for Polymarket Agent Dashboard"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from .routes import router

app = FastAPI(
    title="Polymarket Agent Dashboard",
    description="Dashboard for viewing Polymarket predictions and market data",
    version="1.0.0"
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Frontend directory
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/")
async def serve_frontend():
    """Serve the main React app"""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/styles.css")
async def serve_styles():
    """Serve CSS"""
    return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css")


@app.get("/app.js")
async def serve_js():
    """Serve JavaScript"""
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")
