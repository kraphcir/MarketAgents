"""CSV reading utilities for the dashboard"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Data paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data" / "polymarket"

MARKETS_CSV = DATA_DIR / "markets.csv"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
CONSENSUS_PICKS_CSV = DATA_DIR / "consensus_picks.csv"


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


def read_markets(limit: int = 100) -> List[Dict[str, Any]]:
    """Read markets from CSV, most recent first"""
    df = _safe_read_csv(MARKETS_CSV)
    if df is None:
        return []
    try:
        df = df.sort_values('first_seen', ascending=False).head(limit)
        return df.fillna('').to_dict('records')
    except Exception as e:
        print(f"Error reading markets: {e}")
        return []


def read_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    """Read predictions from CSV, most recent first"""
    df = _safe_read_csv(PREDICTIONS_CSV)
    if df is None:
        return []
    try:
        df = df.sort_values('analysis_timestamp', ascending=False).head(limit)
        return df.fillna('').to_dict('records')
    except Exception as e:
        print(f"Error reading predictions: {e}")
        return []


def read_consensus_picks(limit: int = 20) -> List[Dict[str, Any]]:
    """Read consensus picks from CSV, most recent first"""
    df = _safe_read_csv(CONSENSUS_PICKS_CSV)
    if df is None:
        return []
    try:
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

    df = _safe_read_csv(MARKETS_CSV)
    if df is not None:
        markets_count = len(df)

    df = _safe_read_csv(PREDICTIONS_CSV)
    if df is not None:
        predictions_count = len(df)
        if 'analysis_run_id' in df.columns and len(df) > 0:
            latest_run = df['analysis_run_id'].iloc[-1]

    df = _safe_read_csv(CONSENSUS_PICKS_CSV)
    if df is not None:
        picks_count = len(df)

    return {
        "total_markets": markets_count,
        "total_predictions": predictions_count,
        "total_consensus_picks": picks_count,
        "latest_run_id": latest_run,
        "last_updated": datetime.now().isoformat()
    }
