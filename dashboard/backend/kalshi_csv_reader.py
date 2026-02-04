"""CSV reading utilities for Kalshi and arbitrage dashboard data"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Data paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
KALSHI_DATA_DIR = PROJECT_ROOT / "src" / "data" / "kalshi"
ARBITRAGE_DATA_DIR = PROJECT_ROOT / "src" / "data" / "arbitrage"

KALSHI_MARKETS_CSV = KALSHI_DATA_DIR / "markets.csv"
KALSHI_PREDICTIONS_CSV = KALSHI_DATA_DIR / "predictions.csv"
KALSHI_CONSENSUS_CSV = KALSHI_DATA_DIR / "consensus_picks.csv"
ARBITRAGE_OPPORTUNITIES_CSV = ARBITRAGE_DATA_DIR / "opportunities.csv"
ARBITRAGE_HISTORY_CSV = ARBITRAGE_DATA_DIR / "history.csv"


def read_kalshi_markets(limit: int = 100) -> List[Dict[str, Any]]:
    """Read Kalshi markets from CSV, most recent first"""
    if not KALSHI_MARKETS_CSV.exists():
        return []
    try:
        df = pd.read_csv(KALSHI_MARKETS_CSV)
        df = df.sort_values('first_seen', ascending=False).head(limit)
        return df.fillna('').to_dict('records')
    except Exception as e:
        print(f"Error reading Kalshi markets: {e}")
        return []


def read_kalshi_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    """Read Kalshi predictions from CSV, most recent first"""
    if not KALSHI_PREDICTIONS_CSV.exists():
        return []
    try:
        df = pd.read_csv(KALSHI_PREDICTIONS_CSV)
        df = df.sort_values('analysis_timestamp', ascending=False).head(limit)
        return df.fillna('').to_dict('records')
    except Exception as e:
        print(f"Error reading Kalshi predictions: {e}")
        return []


def read_kalshi_consensus_picks(limit: int = 20) -> List[Dict[str, Any]]:
    """Read Kalshi consensus picks from CSV, most recent first"""
    if not KALSHI_CONSENSUS_CSV.exists():
        return []
    try:
        df = pd.read_csv(KALSHI_CONSENSUS_CSV)
        df = df.sort_values('timestamp', ascending=False).head(limit)
        return df.fillna('').to_dict('records')
    except Exception as e:
        print(f"Error reading Kalshi consensus picks: {e}")
        return []


def read_arbitrage_opportunities(limit: int = 50) -> List[Dict[str, Any]]:
    """Read arbitrage opportunities from CSV, most recent first"""
    if not ARBITRAGE_OPPORTUNITIES_CSV.exists():
        return []
    try:
        df = pd.read_csv(ARBITRAGE_OPPORTUNITIES_CSV)
        df = df.sort_values('timestamp', ascending=False).head(limit)
        return df.fillna('').to_dict('records')
    except Exception as e:
        print(f"Error reading arbitrage opportunities: {e}")
        return []


def read_arbitrage_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Read arbitrage scan history from CSV, most recent first"""
    if not ARBITRAGE_HISTORY_CSV.exists():
        return []
    try:
        df = pd.read_csv(ARBITRAGE_HISTORY_CSV)
        df = df.sort_values('timestamp', ascending=False).head(limit)
        return df.fillna('').to_dict('records')
    except Exception as e:
        print(f"Error reading arbitrage history: {e}")
        return []


def get_kalshi_stats() -> Dict[str, Any]:
    """Get Kalshi-specific stats"""
    kalshi_markets = 0
    kalshi_predictions = 0
    kalshi_picks = 0

    try:
        if KALSHI_MARKETS_CSV.exists():
            kalshi_markets = len(pd.read_csv(KALSHI_MARKETS_CSV))
        if KALSHI_PREDICTIONS_CSV.exists():
            kalshi_predictions = len(pd.read_csv(KALSHI_PREDICTIONS_CSV))
        if KALSHI_CONSENSUS_CSV.exists():
            kalshi_picks = len(pd.read_csv(KALSHI_CONSENSUS_CSV))
    except Exception as e:
        print(f"Error getting Kalshi stats: {e}")

    return {
        "kalshi_markets": kalshi_markets,
        "kalshi_predictions": kalshi_predictions,
        "kalshi_consensus_picks": kalshi_picks,
    }


def get_arbitrage_stats() -> Dict[str, Any]:
    """Get arbitrage-specific stats"""
    opportunities = 0
    profitable = 0
    best_spread = 0

    try:
        if ARBITRAGE_OPPORTUNITIES_CSV.exists():
            df = pd.read_csv(ARBITRAGE_OPPORTUNITIES_CSV)
            opportunities = len(df)
            if len(df) > 0:
                profitable = len(df[df['net_profit_cents'] > 0])
                best_spread = float(df['spread_cents'].max())
    except Exception as e:
        print(f"Error getting arbitrage stats: {e}")

    return {
        "arbitrage_opportunities": opportunities,
        "arbitrage_profitable": profitable,
        "arbitrage_best_spread_cents": best_spread,
    }
