#!/usr/bin/env python3
"""
Crypto Hedge Agent - Live ETH/BTC price tracking with Kalshi hedge opportunity detection.

Tracks real-time cryptocurrency prices and identifies hedge opportunities
in Kalshi crypto prediction markets.

Usage:
    python src/agents/crypto_hedge_agent.py
"""

import os
import sys
import time
import json
import threading
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

import requests
import pandas as pd
from termcolor import cprint
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
load_dotenv(PROJECT_ROOT / '.env')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Price feed settings
PRICE_UPDATE_INTERVAL = 10  # Fetch prices every 10 seconds
PRICE_HISTORY_SIZE = 360  # Keep 1 hour of price history (at 10s intervals)

# Kalshi settings
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_POLL_INTERVAL = 60  # Fetch Kalshi markets every 60 seconds

# Hedge detection settings
HEDGE_SCAN_INTERVAL = 30  # Scan for hedge opportunities every 30 seconds
MIN_HEDGE_PROFIT_PCT = 1.0  # Minimum profit percentage to flag (after fees)
KALSHI_FEE = 0.01  # 1% Kalshi fee

# Crypto keywords for filtering Kalshi markets
CRYPTO_KEYWORDS = [
    'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency'
]

# Price target patterns to extract from market titles
PRICE_PATTERNS = [
    r'\$?([\d,]+(?:\.\d+)?)\s*(?:k|K)?\s*(?:or\s+(?:more|higher|above))',
    r'(?:above|over|exceed|reach|hit)\s*\$?([\d,]+(?:\.\d+)?)\s*(?:k|K)?',
    r'\$?([\d,]+(?:\.\d+)?)\s*(?:k|K)?\s*(?:or\s+(?:less|lower|below))',
    r'(?:below|under|drop|fall)\s*\$?([\d,]+(?:\.\d+)?)\s*(?:k|K)?',
    r'between\s*\$?([\d,]+(?:\.\d+)?)\s*(?:k|K)?\s*(?:and|-)\s*\$?([\d,]+(?:\.\d+)?)\s*(?:k|K)?',
]

# Output paths
DATA_DIR = PROJECT_ROOT / 'src' / 'data' / 'crypto_hedge'

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PriceData:
    """Current price data for a cryptocurrency."""
    symbol: str
    price_usd: float
    price_24h_change_pct: float
    volume_24h: float
    market_cap: float
    timestamp: datetime

@dataclass
class KalshiCryptoMarket:
    """Kalshi market related to crypto."""
    ticker: str
    event_ticker: str
    title: str
    subtitle: str
    crypto_asset: str  # 'BTC', 'ETH', or 'CRYPTO'
    target_price: Optional[float]
    direction: str  # 'above', 'below', 'between', 'unknown'
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    last_price: float
    volume_24h: int
    open_interest: int
    close_time: datetime

@dataclass
class HedgeOpportunity:
    """A potential hedge opportunity."""
    id: str
    crypto_asset: str
    current_price: float
    kalshi_market: KalshiCryptoMarket
    target_price: float
    direction: str
    probability_implied: float  # From Kalshi mid price
    hedge_type: str  # 'bullish_hedge', 'bearish_hedge', 'range_hedge'
    expected_profit_pct: float
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    strategy_description: str
    timestamp: datetime


# =============================================================================
# CRYPTO PRICE FEED
# =============================================================================

class CryptoPriceFeed:
    """Fetches live crypto prices from CoinGecko (free, no API key)."""

    COINGECKO_API = "https://api.coingecko.com/api/v3"

    COIN_IDS = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum'
    }

    def __init__(self):
        self.prices: Dict[str, PriceData] = {}
        self.price_history: Dict[str, deque] = {
            'BTC': deque(maxlen=PRICE_HISTORY_SIZE),
            'ETH': deque(maxlen=PRICE_HISTORY_SIZE)
        }
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the price feed thread."""
        self._running = True
        self._thread = threading.Thread(target=self._price_loop, daemon=True)
        self._thread.start()
        cprint("[+] Crypto price feed started", "green")

    def stop(self):
        """Stop the price feed thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        cprint("[*] Crypto price feed stopped", "cyan")

    def _price_loop(self):
        """Main price fetching loop."""
        while self._running:
            try:
                self._fetch_prices()
            except Exception as e:
                cprint(f"[!] Price fetch error: {e}", "yellow")
            time.sleep(PRICE_UPDATE_INTERVAL)

    def _fetch_prices(self):
        """Fetch current prices from CoinGecko."""
        coin_ids = ','.join(self.COIN_IDS.values())
        url = f"{self.COINGECKO_API}/simple/price"
        params = {
            'ids': coin_ids,
            'vs_currencies': 'usd',
            'include_24hr_change': 'true',
            'include_24hr_vol': 'true',
            'include_market_cap': 'true'
        }

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        now = datetime.now(timezone.utc)

        with self._lock:
            for symbol, coin_id in self.COIN_IDS.items():
                if coin_id in data:
                    coin_data = data[coin_id]
                    price_data = PriceData(
                        symbol=symbol,
                        price_usd=coin_data.get('usd', 0),
                        price_24h_change_pct=coin_data.get('usd_24h_change', 0),
                        volume_24h=coin_data.get('usd_24h_vol', 0),
                        market_cap=coin_data.get('usd_market_cap', 0),
                        timestamp=now
                    )
                    self.prices[symbol] = price_data
                    self.price_history[symbol].append((now, price_data.price_usd))

    def get_price(self, symbol: str) -> Optional[PriceData]:
        """Get current price for a symbol."""
        with self._lock:
            return self.prices.get(symbol)

    def get_price_history(self, symbol: str) -> List[Tuple[datetime, float]]:
        """Get price history for a symbol."""
        with self._lock:
            return list(self.price_history.get(symbol, []))

    def get_all_prices(self) -> Dict[str, PriceData]:
        """Get all current prices."""
        with self._lock:
            return dict(self.prices)


# =============================================================================
# KALSHI CRYPTO MARKET FETCHER
# =============================================================================

class KalshiCryptoFetcher:
    """Fetches crypto-related markets from Kalshi."""

    def __init__(self):
        self.markets: List[KalshiCryptoMarket] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.api_calls = 0

    def start(self):
        """Start the Kalshi polling thread."""
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        cprint("[+] Kalshi crypto market fetcher started", "green")

    def stop(self):
        """Stop the polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        cprint("[*] Kalshi crypto market fetcher stopped", "cyan")

    def _poll_loop(self):
        """Main polling loop."""
        while self._running:
            try:
                self._fetch_crypto_markets()
            except Exception as e:
                cprint(f"[!] Kalshi fetch error: {e}", "yellow")
            time.sleep(KALSHI_POLL_INTERVAL)

    def _fetch_crypto_markets(self):
        """Fetch all crypto-related markets from Kalshi."""
        all_markets = []
        cursor = None

        while True:
            url = f"{KALSHI_API_BASE}/markets"
            params = {
                'status': 'open',
                'limit': 200
            }
            if cursor:
                params['cursor'] = cursor

            resp = requests.get(url, params=params, timeout=30)
            self.api_calls += 1
            resp.raise_for_status()
            data = resp.json()

            markets = data.get('markets', [])
            if not markets:
                break

            # Filter for crypto markets
            for m in markets:
                title_lower = (m.get('title', '') + ' ' + m.get('subtitle', '')).lower()

                # Check if crypto-related
                is_crypto = any(kw in title_lower for kw in CRYPTO_KEYWORDS)
                if not is_crypto:
                    continue

                # Determine which crypto asset
                if 'bitcoin' in title_lower or 'btc' in title_lower:
                    crypto_asset = 'BTC'
                elif 'ethereum' in title_lower or 'eth' in title_lower:
                    crypto_asset = 'ETH'
                else:
                    crypto_asset = 'CRYPTO'

                # Extract price target and direction
                target_price, direction = self._extract_price_target(m.get('title', ''), m.get('subtitle', ''))

                # Parse close time
                close_time_str = m.get('close_time', '')
                try:
                    close_time = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
                except:
                    close_time = datetime.now(timezone.utc)

                market = KalshiCryptoMarket(
                    ticker=m.get('ticker', ''),
                    event_ticker=m.get('event_ticker', ''),
                    title=m.get('title', ''),
                    subtitle=m.get('subtitle', ''),
                    crypto_asset=crypto_asset,
                    target_price=target_price,
                    direction=direction,
                    yes_bid=m.get('yes_bid', 0) / 100 if m.get('yes_bid') else 0,
                    yes_ask=m.get('yes_ask', 0) / 100 if m.get('yes_ask') else 0,
                    no_bid=m.get('no_bid', 0) / 100 if m.get('no_bid') else 0,
                    no_ask=m.get('no_ask', 0) / 100 if m.get('no_ask') else 0,
                    last_price=m.get('last_price', 0) / 100 if m.get('last_price') else 0,
                    volume_24h=m.get('volume_24h', 0),
                    open_interest=m.get('open_interest', 0),
                    close_time=close_time
                )
                all_markets.append(market)

            cursor = data.get('cursor')
            if not cursor:
                break

        with self._lock:
            self.markets = all_markets

        cprint(f"[*] Fetched {len(all_markets)} crypto markets from Kalshi", "cyan")

    def _extract_price_target(self, title: str, subtitle: str) -> Tuple[Optional[float], str]:
        """Extract price target and direction from market title."""
        full_text = f"{title} {subtitle}".lower()

        # Try each pattern
        for pattern in PRICE_PATTERNS:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                # Extract the price value
                price_str = match.group(1).replace(',', '')
                try:
                    price = float(price_str)
                    # Handle 'k' suffix (thousands)
                    if 'k' in full_text[match.start():match.end()].lower():
                        price *= 1000
                    # Determine direction
                    if any(w in full_text for w in ['above', 'over', 'exceed', 'reach', 'hit', 'or more', 'or higher']):
                        direction = 'above'
                    elif any(w in full_text for w in ['below', 'under', 'drop', 'fall', 'or less', 'or lower']):
                        direction = 'below'
                    elif 'between' in full_text:
                        direction = 'between'
                    else:
                        direction = 'unknown'
                    return price, direction
                except ValueError:
                    continue

        return None, 'unknown'

    def get_markets(self) -> List[KalshiCryptoMarket]:
        """Get all crypto markets."""
        with self._lock:
            return list(self.markets)

    def get_markets_for_asset(self, asset: str) -> List[KalshiCryptoMarket]:
        """Get markets for a specific crypto asset."""
        with self._lock:
            return [m for m in self.markets if m.crypto_asset == asset]


# =============================================================================
# HEDGE OPPORTUNITY DETECTOR
# =============================================================================

class HedgeDetector:
    """Detects hedge opportunities between spot prices and Kalshi markets."""

    def __init__(self, price_feed: CryptoPriceFeed, kalshi_fetcher: KalshiCryptoFetcher):
        self.price_feed = price_feed
        self.kalshi_fetcher = kalshi_fetcher
        self.opportunities: List[HedgeOpportunity] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the hedge detection thread."""
        self._running = True
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()
        cprint("[+] Hedge detector started", "green")

    def stop(self):
        """Stop the detection thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        cprint("[*] Hedge detector stopped", "cyan")

    def _scan_loop(self):
        """Main scanning loop."""
        while self._running:
            try:
                self._scan_for_opportunities()
            except Exception as e:
                cprint(f"[!] Hedge scan error: {e}", "yellow")
                import traceback
                traceback.print_exc()
            time.sleep(HEDGE_SCAN_INTERVAL)

    def _scan_for_opportunities(self):
        """Scan for hedge opportunities."""
        opportunities = []
        now = datetime.now(timezone.utc)

        for asset in ['BTC', 'ETH']:
            price_data = self.price_feed.get_price(asset)
            if not price_data:
                continue

            current_price = price_data.price_usd
            markets = self.kalshi_fetcher.get_markets_for_asset(asset)

            for market in markets:
                if not market.target_price or market.direction == 'unknown':
                    continue

                # Skip if no liquidity
                if market.yes_ask == 0 and market.yes_bid == 0:
                    continue

                # Calculate implied probability from mid price
                if market.yes_bid > 0 and market.yes_ask > 0:
                    implied_prob = (market.yes_bid + market.yes_ask) / 2
                elif market.last_price > 0:
                    implied_prob = market.last_price
                else:
                    continue

                # Analyze hedge opportunity based on direction
                opp = self._analyze_hedge(
                    asset=asset,
                    current_price=current_price,
                    market=market,
                    implied_prob=implied_prob,
                    now=now
                )

                if opp and opp.expected_profit_pct >= MIN_HEDGE_PROFIT_PCT:
                    opportunities.append(opp)

        # Sort by expected profit
        opportunities.sort(key=lambda x: x.expected_profit_pct, reverse=True)

        with self._lock:
            self.opportunities = opportunities

        if opportunities:
            cprint(f"[+] Found {len(opportunities)} hedge opportunities", "green")

    def _analyze_hedge(
        self,
        asset: str,
        current_price: float,
        market: KalshiCryptoMarket,
        implied_prob: float,
        now: datetime
    ) -> Optional[HedgeOpportunity]:
        """Analyze a potential hedge opportunity."""

        target = market.target_price
        direction = market.direction

        # Calculate price distance as percentage
        price_distance_pct = abs(current_price - target) / current_price * 100

        # Time to expiry in days
        time_to_expiry = (market.close_time - now).total_seconds() / 86400
        if time_to_expiry <= 0:
            return None

        # Determine hedge type and calculate expected profit
        if direction == 'above':
            if current_price > target:
                # Price already above target - YES is likely to resolve YES
                # Hedge: Buy YES if cheap, lock in profit if price stays above
                hedge_type = 'bullish_hedge'

                # Calculate profit: if YES resolves, we get $1 per contract
                # Cost is the ask price
                cost = market.yes_ask if market.yes_ask > 0 else implied_prob
                gross_profit = 1.0 - cost
                net_profit = gross_profit * (1 - KALSHI_FEE)
                expected_profit_pct = (net_profit / cost) * 100 if cost > 0 else 0

                strategy = f"Price ${current_price:,.0f} already above ${target:,.0f}. " \
                          f"Buy YES at {cost:.2f} - likely resolves YES for ${1-cost:.2f} profit."

            else:
                # Price below target - NO is currently favored
                # Hedge: Buy NO to profit if price stays below
                hedge_type = 'bearish_hedge'

                cost = market.no_ask if market.no_ask > 0 else (1 - implied_prob)
                gross_profit = 1.0 - cost
                net_profit = gross_profit * (1 - KALSHI_FEE)
                expected_profit_pct = (net_profit / cost) * 100 if cost > 0 else 0

                strategy = f"Price ${current_price:,.0f} below ${target:,.0f}. " \
                          f"Buy NO at {cost:.2f} - profits if price stays below target."

        elif direction == 'below':
            if current_price < target:
                # Price already below target - YES likely
                hedge_type = 'bearish_hedge'

                cost = market.yes_ask if market.yes_ask > 0 else implied_prob
                gross_profit = 1.0 - cost
                net_profit = gross_profit * (1 - KALSHI_FEE)
                expected_profit_pct = (net_profit / cost) * 100 if cost > 0 else 0

                strategy = f"Price ${current_price:,.0f} below ${target:,.0f}. " \
                          f"Buy YES at {cost:.2f} - likely resolves YES for ${1-cost:.2f} profit."

            else:
                # Price above target - NO is favored
                hedge_type = 'bullish_hedge'

                cost = market.no_ask if market.no_ask > 0 else (1 - implied_prob)
                gross_profit = 1.0 - cost
                net_profit = gross_profit * (1 - KALSHI_FEE)
                expected_profit_pct = (net_profit / cost) * 100 if cost > 0 else 0

                strategy = f"Price ${current_price:,.0f} above ${target:,.0f}. " \
                          f"Buy NO at {cost:.2f} - profits if price stays above."

        else:
            return None

        # Determine risk level based on price distance and time
        if price_distance_pct > 20 and time_to_expiry > 7:
            risk_level = 'LOW'
        elif price_distance_pct > 10 or time_to_expiry > 3:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'

        # Skip if expected profit is too low
        if expected_profit_pct < MIN_HEDGE_PROFIT_PCT:
            return None

        return HedgeOpportunity(
            id=f"{asset}-{market.ticker}-{now.strftime('%Y%m%d%H%M%S')}",
            crypto_asset=asset,
            current_price=current_price,
            kalshi_market=market,
            target_price=target,
            direction=direction,
            probability_implied=implied_prob,
            hedge_type=hedge_type,
            expected_profit_pct=expected_profit_pct,
            risk_level=risk_level,
            strategy_description=strategy,
            timestamp=now
        )

    def get_opportunities(self) -> List[HedgeOpportunity]:
        """Get all current opportunities."""
        with self._lock:
            return list(self.opportunities)

    def get_opportunities_by_asset(self, asset: str) -> List[HedgeOpportunity]:
        """Get opportunities for a specific asset."""
        with self._lock:
            return [o for o in self.opportunities if o.crypto_asset == asset]


# =============================================================================
# CSV OUTPUT
# =============================================================================

class CSVWriter:
    """Writes hedge data to CSV files."""

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write_prices(self, prices: Dict[str, PriceData]):
        """Write current prices to CSV."""
        if not prices:
            return

        rows = []
        for symbol, data in prices.items():
            rows.append({
                'symbol': symbol,
                'price_usd': data.price_usd,
                'change_24h_pct': data.price_24h_change_pct,
                'volume_24h': data.volume_24h,
                'market_cap': data.market_cap,
                'timestamp': data.timestamp.isoformat()
            })

        df = pd.DataFrame(rows)
        with self._lock:
            df.to_csv(DATA_DIR / 'prices.csv', index=False)

    def write_markets(self, markets: List[KalshiCryptoMarket]):
        """Write Kalshi crypto markets to CSV."""
        if not markets:
            return

        rows = []
        for m in markets:
            rows.append({
                'ticker': m.ticker,
                'event_ticker': m.event_ticker,
                'title': m.title,
                'subtitle': m.subtitle,
                'crypto_asset': m.crypto_asset,
                'target_price': m.target_price,
                'direction': m.direction,
                'yes_bid': m.yes_bid,
                'yes_ask': m.yes_ask,
                'no_bid': m.no_bid,
                'no_ask': m.no_ask,
                'last_price': m.last_price,
                'volume_24h': m.volume_24h,
                'open_interest': m.open_interest,
                'close_time': m.close_time.isoformat()
            })

        df = pd.DataFrame(rows)
        with self._lock:
            df.to_csv(DATA_DIR / 'kalshi_crypto_markets.csv', index=False)

    def write_opportunities(self, opportunities: List[HedgeOpportunity]):
        """Write hedge opportunities to CSV."""
        rows = []
        for o in opportunities:
            rows.append({
                'id': o.id,
                'crypto_asset': o.crypto_asset,
                'current_price': o.current_price,
                'target_price': o.target_price,
                'direction': o.direction,
                'kalshi_ticker': o.kalshi_market.ticker,
                'kalshi_title': o.kalshi_market.title,
                'yes_bid': o.kalshi_market.yes_bid,
                'yes_ask': o.kalshi_market.yes_ask,
                'no_bid': o.kalshi_market.no_bid,
                'no_ask': o.kalshi_market.no_ask,
                'implied_probability': o.probability_implied,
                'hedge_type': o.hedge_type,
                'expected_profit_pct': o.expected_profit_pct,
                'risk_level': o.risk_level,
                'strategy': o.strategy_description,
                'close_time': o.kalshi_market.close_time.isoformat(),
                'timestamp': o.timestamp.isoformat()
            })

        df = pd.DataFrame(rows)
        with self._lock:
            df.to_csv(DATA_DIR / 'opportunities.csv', index=False)

    def write_history(self, scan_count: int, opportunities_found: int, api_calls: int):
        """Append scan history."""
        row = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'scan_count': scan_count,
            'opportunities_found': opportunities_found,
            'api_calls': api_calls
        }

        history_file = DATA_DIR / 'history.csv'
        with self._lock:
            if history_file.exists():
                df = pd.read_csv(history_file)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            else:
                df = pd.DataFrame([row])
            df.to_csv(history_file, index=False)


# =============================================================================
# MAIN AGENT
# =============================================================================

class CryptoHedgeAgent:
    """Main agent that coordinates all components."""

    def __init__(self):
        self.price_feed = CryptoPriceFeed()
        self.kalshi_fetcher = KalshiCryptoFetcher()
        self.hedge_detector = HedgeDetector(self.price_feed, self.kalshi_fetcher)
        self.csv_writer = CSVWriter()
        self._running = False
        self.scan_count = 0

    def start(self):
        """Start all components."""
        cprint("\n" + "=" * 70, "cyan")
        cprint("  Crypto Hedge Agent - ETH/BTC Price Tracking + Kalshi Hedge Detection", "cyan")
        cprint("=" * 70 + "\n", "cyan")

        self._running = True

        # Start all components
        self.price_feed.start()
        time.sleep(2)  # Wait for initial prices

        self.kalshi_fetcher.start()
        time.sleep(2)  # Wait for initial market fetch

        self.hedge_detector.start()

        cprint("[+] All components started\n", "green")

        # Run main loop
        self._main_loop()

    def stop(self):
        """Stop all components."""
        self._running = False
        self.hedge_detector.stop()
        self.kalshi_fetcher.stop()
        self.price_feed.stop()
        cprint("\n[*] Crypto Hedge Agent stopped", "cyan")

    def _main_loop(self):
        """Main display and CSV writing loop."""
        last_csv_write = 0

        while self._running:
            try:
                # Display current status
                self._display_status()

                # Write to CSV periodically (every 30 seconds)
                now = time.time()
                if now - last_csv_write >= 30:
                    self._write_csv_output()
                    last_csv_write = now
                    self.scan_count += 1

                time.sleep(10)

            except KeyboardInterrupt:
                break
            except Exception as e:
                cprint(f"[!] Main loop error: {e}", "yellow")
                time.sleep(5)

    def _display_status(self):
        """Display current prices and opportunities."""
        prices = self.price_feed.get_all_prices()
        markets = self.kalshi_fetcher.get_markets()
        opportunities = self.hedge_detector.get_opportunities()

        # Clear screen and display header
        print("\033[2J\033[H", end="")  # Clear screen
        cprint("=" * 70, "cyan")
        cprint("  CRYPTO HEDGE AGENT - Live Monitor", "cyan")
        cprint("=" * 70, "cyan")
        print()

        # Display prices
        cprint("📊 LIVE PRICES", "white", attrs=["bold"])
        cprint("-" * 40, "white")
        for symbol, data in prices.items():
            change_color = "green" if data.price_24h_change_pct >= 0 else "red"
            change_str = f"+{data.price_24h_change_pct:.2f}%" if data.price_24h_change_pct >= 0 else f"{data.price_24h_change_pct:.2f}%"
            cprint(f"  {symbol}: ${data.price_usd:,.2f}  ", "white", end="")
            cprint(f"({change_str})", change_color)
        print()

        # Display Kalshi market count
        cprint(f"📈 KALSHI CRYPTO MARKETS: {len(markets)}", "white", attrs=["bold"])
        btc_markets = len([m for m in markets if m.crypto_asset == 'BTC'])
        eth_markets = len([m for m in markets if m.crypto_asset == 'ETH'])
        cprint(f"  BTC: {btc_markets} | ETH: {eth_markets}", "white")
        print()

        # Display hedge opportunities
        cprint(f"💰 HEDGE OPPORTUNITIES: {len(opportunities)}", "white", attrs=["bold"])
        cprint("-" * 70, "white")

        if opportunities:
            for i, opp in enumerate(opportunities[:10]):  # Show top 10
                risk_color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}.get(opp.risk_level, "white")

                cprint(f"\n  [{i+1}] {opp.crypto_asset} - {opp.hedge_type.upper()}", "cyan", attrs=["bold"])
                cprint(f"      Market: {opp.kalshi_market.title[:50]}...", "white")
                cprint(f"      Current: ${opp.current_price:,.0f} | Target: ${opp.target_price:,.0f} ({opp.direction})", "white")
                cprint(f"      Expected Profit: ", "white", end="")
                cprint(f"{opp.expected_profit_pct:.1f}%", "green", end="")
                cprint(f" | Risk: ", "white", end="")
                cprint(f"{opp.risk_level}", risk_color)
                cprint(f"      Strategy: {opp.strategy_description[:70]}...", "blue")
        else:
            cprint("  No opportunities found matching criteria", "yellow")

        print()
        cprint("-" * 70, "white")
        cprint(f"  Scans: {self.scan_count} | API Calls: {self.kalshi_fetcher.api_calls} | Press Ctrl+C to stop", "white")

    def _write_csv_output(self):
        """Write all data to CSV files."""
        prices = self.price_feed.get_all_prices()
        markets = self.kalshi_fetcher.get_markets()
        opportunities = self.hedge_detector.get_opportunities()

        self.csv_writer.write_prices(prices)
        self.csv_writer.write_markets(markets)
        self.csv_writer.write_opportunities(opportunities)
        self.csv_writer.write_history(
            self.scan_count,
            len(opportunities),
            self.kalshi_fetcher.api_calls
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    agent = CryptoHedgeAgent()
    try:
        agent.start()
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()


if __name__ == '__main__':
    main()
