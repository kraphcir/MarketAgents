"""CSV reading utilities for Kalshi and arbitrage dashboard data"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Data paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
KALSHI_DATA_DIR = PROJECT_ROOT / "src" / "data" / "kalshi"
ARBITRAGE_DATA_DIR = PROJECT_ROOT / "src" / "data" / "arbitrage"
CRYPTO_HEDGE_DATA_DIR = PROJECT_ROOT / "src" / "data" / "crypto_hedge"

KALSHI_MARKETS_CSV = KALSHI_DATA_DIR / "markets.csv"
KALSHI_PREDICTIONS_CSV = KALSHI_DATA_DIR / "predictions.csv"
KALSHI_CONSENSUS_CSV = KALSHI_DATA_DIR / "consensus_picks.csv"
ARBITRAGE_OPPORTUNITIES_CSV = ARBITRAGE_DATA_DIR / "opportunities.csv"
ARBITRAGE_HISTORY_CSV = ARBITRAGE_DATA_DIR / "history.csv"
CRYPTO_PRICES_CSV = CRYPTO_HEDGE_DATA_DIR / "prices.csv"
CRYPTO_MARKETS_CSV = CRYPTO_HEDGE_DATA_DIR / "kalshi_crypto_markets.csv"
CRYPTO_OPPORTUNITIES_CSV = CRYPTO_HEDGE_DATA_DIR / "opportunities.csv"
CRYPTO_HISTORY_CSV = CRYPTO_HEDGE_DATA_DIR / "history.csv"


def _safe_read_csv(filepath: Path) -> Optional[pd.DataFrame]:
    """Safely read a CSV file, returning None if file is empty or doesn't exist"""
    if not filepath.exists():
        return None
    # Check if file has content (more than just newlines/whitespace)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return None
    except Exception:
        return None
    # Now try to parse the CSV
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return None
        return df
    except pd.errors.EmptyDataError:
        return None
    except Exception:
        return None


def read_kalshi_markets(limit: int = 100) -> List[Dict[str, Any]]:
    """Read Kalshi markets from CSV, most recent first"""
    df = _safe_read_csv(KALSHI_MARKETS_CSV)
    if df is None:
        return []
    try:
        df = df.sort_values('first_seen', ascending=False).head(limit)
        return df.fillna('').to_dict('records')
    except Exception as e:
        print(f"Error reading Kalshi markets: {e}")
        return []


def read_kalshi_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    """Read Kalshi predictions from CSV, most recent first"""
    df = _safe_read_csv(KALSHI_PREDICTIONS_CSV)
    if df is None:
        return []
    try:
        df = df.sort_values('analysis_timestamp', ascending=False).head(limit)
        return df.fillna('').to_dict('records')
    except Exception as e:
        print(f"Error reading Kalshi predictions: {e}")
        return []


def read_kalshi_consensus_picks(limit: int = 20) -> List[Dict[str, Any]]:
    """Read Kalshi consensus picks from CSV, most recent first"""
    df = _safe_read_csv(KALSHI_CONSENSUS_CSV)
    if df is None:
        return []
    try:
        df = df.sort_values('timestamp', ascending=False).head(limit)
        return df.fillna('').to_dict('records')
    except Exception as e:
        print(f"Error reading Kalshi consensus picks: {e}")
        return []


def read_arbitrage_opportunities(limit: int = 50) -> List[Dict[str, Any]]:
    """Read arbitrage opportunities from CSV, most recent first"""
    df = _safe_read_csv(ARBITRAGE_OPPORTUNITIES_CSV)
    if df is None:
        return []
    try:
        df = df.sort_values('timestamp', ascending=False).head(limit)
        return df.fillna('').to_dict('records')
    except Exception as e:
        print(f"Error reading arbitrage opportunities: {e}")
        return []


def read_arbitrage_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Read arbitrage scan history from CSV, most recent first"""
    df = _safe_read_csv(ARBITRAGE_HISTORY_CSV)
    if df is None:
        return []
    try:
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

    df = _safe_read_csv(KALSHI_MARKETS_CSV)
    if df is not None:
        kalshi_markets = len(df)
    df = _safe_read_csv(KALSHI_PREDICTIONS_CSV)
    if df is not None:
        kalshi_predictions = len(df)
    df = _safe_read_csv(KALSHI_CONSENSUS_CSV)
    if df is not None:
        kalshi_picks = len(df)

    return {
        "kalshi_markets": kalshi_markets,
        "kalshi_predictions": kalshi_predictions,
        "kalshi_consensus_picks": kalshi_picks,
    }


def get_arbitrage_stats() -> Dict[str, Any]:
    """Get arbitrage-specific stats"""
    opportunities = 0
    profitable = 0
    best_spread = 0.0

    df = _safe_read_csv(ARBITRAGE_OPPORTUNITIES_CSV)
    if df is not None:
        opportunities = len(df)
        if 'net_profit_cents' in df.columns:
            profitable = len(df[df['net_profit_cents'] > 0])
        if 'spread_cents' in df.columns:
            best_spread = float(df['spread_cents'].max())

    return {
        "arbitrage_opportunities": opportunities,
        "arbitrage_profitable": profitable,
        "arbitrage_best_spread_cents": best_spread,
    }


def read_crypto_prices() -> List[Dict[str, Any]]:
    """Read current crypto prices from CSV"""
    df = _safe_read_csv(CRYPTO_PRICES_CSV)
    if df is None:
        return []
    return df.fillna('').to_dict('records')


def read_crypto_markets(limit: int = 100) -> List[Dict[str, Any]]:
    """Read Kalshi crypto markets from CSV"""
    df = _safe_read_csv(CRYPTO_MARKETS_CSV)
    if df is None:
        return []
    return df.head(limit).fillna('').to_dict('records')


def read_crypto_opportunities(limit: int = 50) -> List[Dict[str, Any]]:
    """Read crypto hedge opportunities from CSV, sorted by profit"""
    df = _safe_read_csv(CRYPTO_OPPORTUNITIES_CSV)
    if df is None:
        return []
    try:
        df = df.sort_values('expected_profit_pct', ascending=False).head(limit)
        return df.fillna('').to_dict('records')
    except Exception as e:
        print(f"Error reading crypto opportunities: {e}")
        return []


def read_crypto_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Read crypto hedge scan history from CSV"""
    df = _safe_read_csv(CRYPTO_HISTORY_CSV)
    if df is None:
        return []
    try:
        df = df.sort_values('timestamp', ascending=False).head(limit)
        return df.fillna('').to_dict('records')
    except Exception as e:
        print(f"Error reading crypto history: {e}")
        return []


def get_crypto_stats() -> Dict[str, Any]:
    """Get crypto hedge stats"""
    prices_count = 0
    markets_count = 0
    opportunities_count = 0
    best_profit = 0.0

    df = _safe_read_csv(CRYPTO_PRICES_CSV)
    if df is not None:
        prices_count = len(df)
    df = _safe_read_csv(CRYPTO_MARKETS_CSV)
    if df is not None:
        markets_count = len(df)
    df = _safe_read_csv(CRYPTO_OPPORTUNITIES_CSV)
    if df is not None:
        opportunities_count = len(df)
        if 'expected_profit_pct' in df.columns:
            best_profit = float(df['expected_profit_pct'].max())

    return {
        "crypto_prices_tracked": prices_count,
        "crypto_markets": markets_count,
        "crypto_opportunities": opportunities_count,
        "crypto_best_profit_pct": best_profit,
    }
