"""CSV reading utilities for the dashboard"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Data paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data" / "polymarket"

MARKETS_CSV = DATA_DIR / "markets.csv"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
CONSENSUS_PICKS_CSV = DATA_DIR / "consensus_picks.csv"


def read_markets(limit: int = 100) -> List[Dict[str, Any]]:
    """Read markets from CSV, most recent first"""
    if not MARKETS_CSV.exists():
        return []
    try:
        df = pd.read_csv(MARKETS_CSV)
        df = df.sort_values('first_seen', ascending=False).head(limit)
        return df.fillna('').to_dict('records')
    except Exception as e:
        print(f"Error reading markets: {e}")
        return []


def read_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    """Read predictions from CSV, most recent first"""
    if not PREDICTIONS_CSV.exists():
        return []
    try:
        df = pd.read_csv(PREDICTIONS_CSV)
        df = df.sort_values('analysis_timestamp', ascending=False).head(limit)
        return df.fillna('').to_dict('records')
    except Exception as e:
        print(f"Error reading predictions: {e}")
        return []


def read_consensus_picks(limit: int = 20) -> List[Dict[str, Any]]:
    """Read consensus picks from CSV, most recent first"""
    if not CONSENSUS_PICKS_CSV.exists():
        return []
    try:
        df = pd.read_csv(CONSENSUS_PICKS_CSV)
        df = df.sort_values('timestamp', ascending=False).head(limit)
        return df.fillna('').to_dict('records')
    except Exception as e:
        print(f"Error reading consensus picks: {e}")
        return []


def get_stats() -> Dict[str, Any]:
    """Get dashboard statistics"""
    markets_count = 0
    predictions_count = 0
    picks_count = 0
    latest_run = None

    try:
        if MARKETS_CSV.exists():
            markets_count = len(pd.read_csv(MARKETS_CSV))

        if PREDICTIONS_CSV.exists():
            df = pd.read_csv(PREDICTIONS_CSV)
            predictions_count = len(df)
            if len(df) > 0:
                latest_run = df['analysis_run_id'].iloc[-1]

        if CONSENSUS_PICKS_CSV.exists():
            picks_count = len(pd.read_csv(CONSENSUS_PICKS_CSV))
    except Exception as e:
        print(f"Error getting stats: {e}")

    return {
        "total_markets": markets_count,
        "total_predictions": predictions_count,
        "total_consensus_picks": picks_count,
        "latest_run_id": latest_run,
        "last_updated": datetime.now().isoformat()
    }
