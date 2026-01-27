#!/usr/bin/env python3
"""
Polymarket Agent Dashboard

Usage:
    python dashboard/run.py

The dashboard will be available at http://localhost:8000
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    import uvicorn

    print("\n" + "=" * 60)
    print("  Polymarket Agent Dashboard")
    print("=" * 60)
    print("\n  Starting server at http://localhost:8000")
    print("  Press Ctrl+C to stop\n")

    uvicorn.run(
        "dashboard.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
