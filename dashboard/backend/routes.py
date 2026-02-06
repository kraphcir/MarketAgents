"""API routes for the dashboard"""

from fastapi import APIRouter, Query
from typing import List
from . import csv_reader
from . import kalshi_csv_reader

router = APIRouter()


@router.get("/api/stats")
async def get_stats():
    """Get combined dashboard statistics"""
    stats = csv_reader.get_stats()
    stats.update(kalshi_csv_reader.get_kalshi_stats())
    stats.update(kalshi_csv_reader.get_arbitrage_stats())
    stats.update(kalshi_csv_reader.get_crypto_stats())
    return stats


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


@router.get("/api/kalshi/markets")
async def get_kalshi_markets(limit: int = Query(default=100, le=500)):
    """Get recent Kalshi markets"""
    return kalshi_csv_reader.read_kalshi_markets(limit=limit)


@router.get("/api/kalshi/predictions")
async def get_kalshi_predictions(limit: int = Query(default=50, le=200)):
    """Get recent Kalshi predictions"""
    return kalshi_csv_reader.read_kalshi_predictions(limit=limit)


@router.get("/api/kalshi/consensus")
async def get_kalshi_consensus_picks(limit: int = Query(default=20, le=100)):
    """Get Kalshi consensus picks"""
    return kalshi_csv_reader.read_kalshi_consensus_picks(limit=limit)


@router.get("/api/arbitrage/opportunities")
async def get_arbitrage_opportunities(limit: int = Query(default=50, le=200)):
    """Get arbitrage opportunities"""
    return kalshi_csv_reader.read_arbitrage_opportunities(limit=limit)


@router.get("/api/arbitrage/history")
async def get_arbitrage_history(limit: int = Query(default=20, le=100)):
    """Get arbitrage scan history"""
    return kalshi_csv_reader.read_arbitrage_history(limit=limit)


@router.get("/api/crypto/prices")
async def get_crypto_prices():
    """Get current crypto prices"""
    return kalshi_csv_reader.read_crypto_prices()


@router.get("/api/crypto/markets")
async def get_crypto_markets(limit: int = Query(default=100, le=500)):
    """Get Kalshi crypto markets"""
    return kalshi_csv_reader.read_crypto_markets(limit=limit)


@router.get("/api/crypto/opportunities")
async def get_crypto_opportunities(limit: int = Query(default=50, le=200)):
    """Get crypto hedge opportunities"""
    return kalshi_csv_reader.read_crypto_opportunities(limit=limit)


@router.get("/api/crypto/history")
async def get_crypto_history(limit: int = Query(default=20, le=100)):
    """Get crypto hedge scan history"""
    return kalshi_csv_reader.read_crypto_history(limit=limit)


@router.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}
