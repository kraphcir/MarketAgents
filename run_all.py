#!/usr/bin/env python3
"""
Run all agents and the dashboard in parallel.

Usage:
    python run_all.py              # Run everything
    python run_all.py --fresh      # Clear old data and start fresh
    python run_all.py --no-poly    # Skip Polymarket agent
    python run_all.py --no-kalshi  # Skip Kalshi agent
    python run_all.py --no-arb     # Skip Arbitrage agent
    python run_all.py --no-dash    # Skip Dashboard

Dashboard: http://localhost:8000
"""

import sys
import os
import time
import signal
import argparse
import subprocess
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# Color codes for terminal output
COLORS = {
    'poly':  '\033[96m',   # cyan
    'kalshi': '\033[93m',  # yellow
    'arb':   '\033[95m',   # magenta
    'crypto': '\033[94m',  # blue
    'dash':  '\033[92m',   # green
    'sys':   '\033[97m',   # white
    'err':   '\033[91m',   # red
    'reset': '\033[0m',
}


def log(tag, msg):
    color = COLORS.get(tag, COLORS['sys'])
    label = tag.upper().ljust(6)
    print(f"{color}[{label}]{COLORS['reset']} {msg}")


def clear_data_directories():
    """Clear all CSV data from previous runs."""
    data_dirs = [
        PROJECT_ROOT / 'src' / 'data' / 'polymarket',
        PROJECT_ROOT / 'src' / 'data' / 'kalshi',
        PROJECT_ROOT / 'src' / 'data' / 'arbitrage',
        PROJECT_ROOT / 'src' / 'data' / 'crypto_hedge',
    ]

    cleared_count = 0
    for data_dir in data_dirs:
        if data_dir.exists():
            for csv_file in data_dir.glob('*.csv'):
                try:
                    csv_file.unlink()
                    cleared_count += 1
                    log('sys', f'  Removed: {csv_file.relative_to(PROJECT_ROOT)}')
                except Exception as e:
                    log('err', f'  Failed to remove {csv_file}: {e}')

    return cleared_count


def stream_output(proc, tag):
    """Stream subprocess stdout/stderr with a colored prefix."""
    try:
        for line in iter(proc.stdout.readline, ''):
            if line:
                log(tag, line.rstrip())
    except (ValueError, OSError):
        pass  # pipe closed


def main():
    parser = argparse.ArgumentParser(description="Run all prediction market agents and dashboard")
    parser.add_argument('--fresh', action='store_true', help='Clear old data before starting')
    parser.add_argument('--no-poly', action='store_true', help='Skip Polymarket agent')
    parser.add_argument('--no-kalshi', action='store_true', help='Skip Kalshi agent')
    parser.add_argument('--no-arb', action='store_true', help='Skip Arbitrage agent')
    parser.add_argument('--no-crypto', action='store_true', help='Skip Crypto Hedge agent')
    parser.add_argument('--no-dash', action='store_true', help='Skip Dashboard')
    args = parser.parse_args()

    python = sys.executable
    processes = []
    threads = []

    print(f"\n{COLORS['sys']}{'='*70}")
    print("  Prediction Market Agent Suite")
    print(f"{'='*70}{COLORS['reset']}\n")

    # Clear old data if --fresh flag is set
    if args.fresh:
        log('sys', 'Clearing old data from previous runs...')
        cleared = clear_data_directories()
        if cleared > 0:
            log('sys', f'Cleared {cleared} CSV file(s)')
        else:
            log('sys', 'No old data files found')
        print()

    # Determine what to run
    components = []
    if not args.no_poly:
        components.append(('poly', 'Polymarket Agent', [python, str(PROJECT_ROOT / 'src' / 'agents' / 'polymarket_agent.py')]))
    if not args.no_kalshi:
        components.append(('kalshi', 'Kalshi Agent', [python, str(PROJECT_ROOT / 'src' / 'agents' / 'kalshi_agent.py')]))
    if not args.no_arb:
        components.append(('arb', 'Arbitrage Agent', [python, str(PROJECT_ROOT / 'src' / 'agents' / 'arbitrage_agent.py')]))
    if not args.no_crypto:
        components.append(('crypto', 'Crypto Hedge Agent', [python, str(PROJECT_ROOT / 'src' / 'agents' / 'crypto_hedge_agent.py')]))
    if not args.no_dash:
        components.append(('dash', 'Dashboard', [python, str(PROJECT_ROOT / 'dashboard' / 'run.py')]))

    if not components:
        print("Nothing to run (all components disabled).")
        return

    # Print summary
    log('sys', 'Starting components:')
    for tag, name, _ in components:
        log(tag, f'  {name}')
    if not args.no_dash:
        log('dash', '  Dashboard at http://localhost:8000')
    print()

    # Launch each component as a subprocess
    for tag, name, cmd in components:
        log(tag, f'Starting {name}...')
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, 'PYTHONUNBUFFERED': '1'},
            )
            processes.append((tag, name, proc))

            # Stream output in a background thread
            t = threading.Thread(target=stream_output, args=(proc, tag), daemon=True)
            t.start()
            threads.append(t)

            # Small delay between launches to avoid port/init conflicts
            time.sleep(1)

        except Exception as e:
            log('err', f'Failed to start {name}: {e}')

    if not processes:
        log('err', 'No processes started successfully.')
        return

    log('sys', f'\nAll {len(processes)} components running. Press Ctrl+C to stop all.\n')

    # Wait for Ctrl+C then clean up
    def shutdown(signum=None, frame=None):
        print()
        log('sys', 'Shutting down all components...')
        for tag, name, proc in processes:
            try:
                proc.terminate()
                log(tag, f'Stopped {name}')
            except Exception:
                pass

        # Give processes a moment to exit, then force kill
        time.sleep(2)
        for tag, name, proc in processes:
            if proc.poll() is None:
                try:
                    proc.kill()
                    log('err', f'Force killed {name}')
                except Exception:
                    pass

        log('sys', 'All components stopped.')
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Monitor processes - restart if any crash
    try:
        while True:
            for tag, name, proc in processes:
                ret = proc.poll()
                if ret is not None:
                    log('err', f'{name} exited with code {ret}')
            time.sleep(5)
    except (KeyboardInterrupt, SystemExit):
        shutdown()


if __name__ == '__main__':
    main()
