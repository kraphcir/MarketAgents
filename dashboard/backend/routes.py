"""API routes for the dashboard"""

from fastapi import APIRouter, Query
from typing import List
from . import csv_reader

router = APIRouter()


@router.get("/api/stats")
async def get_stats():
    """Get dashboard statistics"""
    return csv_reader.get_stats()


@router.get("/api/markets")
async def get_markets(limit: int = Query(default=100, le=500)):
    """Get recent markets"""
    return csv_reader.read_markets(limit=limit)


@router.get("/api/predictions")
async def get_predictions(limit: int = Query(default=50, le=200)):
    """Get recent predictions"""
    return csv_reader.read_predictions(limit=limit)


@router.get("/api/consensus")
async def get_consensus_picks(limit: int = Query(default=20, le=100)):
    """Get top consensus picks"""
    return csv_reader.read_consensus_picks(limit=limit)


@router.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}
