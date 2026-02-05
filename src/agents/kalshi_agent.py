"""
Kalshi Prediction Market Agent - REST polling + AI analysis.
Mirrors the Polymarket agent architecture but uses Kalshi's public REST API.
NO ACTUAL TRADING - just predictions and analysis.
"""

import os
import sys
import time
import json
import re
import requests
import pandas as pd
import threading
from datetime import datetime, timedelta
from pathlib import Path
from termcolor import cprint

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.model_factory import ModelFactory
from src.agents.polymarket_agent import IGNORE_CRYPTO_KEYWORDS, IGNORE_SPORTS_KEYWORDS

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Kalshi API (public - no auth needed for market reads)
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Polling intervals (REST-based since Kalshi WS requires auth)
POLL_INTERVAL_SECONDS = 60          # Fetch markets every 60 seconds
REQUEST_DELAY = 0.1                 # Delay between paginated requests (rate limit safety)

# Market filtering (relaxed to collect more markets)
MIN_VOLUME_24H = 1000               # Minimum 24h volume in dollars
MIN_OPEN_INTEREST = 500             # Minimum open interest
IGNORE_PRICE_THRESHOLD = 0.01      # Ignore near-resolution markets
MAX_DAYS_TO_CLOSE = 365             # Ignore markets closing > 365 days out
MIN_DAYS_TO_CLOSE = 0               # Include all markets regardless of close time

# Analysis behavior (same cadence as Polymarket agent)
ANALYSIS_CHECK_INTERVAL_SECONDS = 300  # Check every 5 minutes
NEW_MARKETS_FOR_ANALYSIS = 3          # Trigger analysis with 3 fresh markets
MARKETS_TO_ANALYZE = 3                # Number of markets to send to AI
MARKETS_TO_DISPLAY = 20               # Markets to show in status
REANALYSIS_HOURS = 8                  # Re-analyze after this many hours

# AI Configuration
USE_SWARM_MODE = True
AI_MODEL_PROVIDER = "xai"
AI_MODEL_NAME = "grok-2-fast-reasoning"
SEND_PRICE_INFO_TO_AI = False
TOP_MARKETS_COUNT = 5

# System prompt for AI market analysis
MARKET_ANALYSIS_SYSTEM_PROMPT = """You are a prediction market expert analyzing Kalshi prediction markets.
For each market, provide your prediction in this exact format:

MARKET [number]: [decision]
Reasoning: [brief 1-2 sentence explanation]

Decision must be one of: YES, NO, or NO_TRADE
- YES means you would bet on the "Yes" outcome
- NO means you would bet on the "No" outcome
- NO_TRADE means you would not take a position

Be concise and focused on the most promising opportunities."""

# Consensus AI prompt
CONSENSUS_AI_PROMPT_TEMPLATE = """You are analyzing predictions from multiple AI models on Kalshi prediction markets.

MARKET REFERENCE:
{market_reference}

ALL AI RESPONSES:
{all_responses}

Based on ALL of these AI responses, identify the TOP {top_count} MARKETS that have the STRONGEST CONSENSUS across all models.

Rules:
- Look for markets where most AIs agree on the same side (YES, NO, or NO_TRADE)
- Ignore markets with split opinions
- Focus on clear, strong agreement
- DO NOT use any reasoning or thinking - just analyze the responses
- Provide exactly {top_count} markets ranked by consensus strength

Format your response EXACTLY like this:

TOP {top_count} CONSENSUS PICKS:

1. Market [number]: [market title]
   Side: [YES/NO/NO_TRADE]
   Consensus: [X out of Y models agreed]
   Link: [kalshi URL from market reference]
   Reasoning: [1 sentence why this is a strong pick]

2. Market [number]: [market title]
   Side: [YES/NO/NO_TRADE]
   Consensus: [X out of Y models agreed]
   Link: [kalshi URL from market reference]
   Reasoning: [1 sentence why this is a strong pick]

[Continue for all {top_count} markets...]
"""

# Data paths
DATA_FOLDER = os.path.join(project_root, "src/data/kalshi")
MARKETS_CSV = os.path.join(DATA_FOLDER, "markets.csv")
PREDICTIONS_CSV = os.path.join(DATA_FOLDER, "predictions.csv")
CONSENSUS_PICKS_CSV = os.path.join(DATA_FOLDER, "consensus_picks.csv")


# ==============================================================================
# Kalshi Agent
# ==============================================================================

class KalshiAgent:
    """Agent that tracks Kalshi markets via REST polling and provides AI predictions"""

    def __init__(self):
        """Initialize the Kalshi agent"""
        cprint("\n" + "="*80, "cyan")
        cprint("[*] Kalshi Prediction Market Agent - Initializing", "cyan", attrs=['bold'])
        cprint("="*80, "cyan")

        # Create data folder
        os.makedirs(DATA_FOLDER, exist_ok=True)

        # Thread-safe lock for CSV access
        self.csv_lock = threading.Lock()

        # Track analysis state
        self.last_analyzed_count = 0
        self.last_analysis_run_timestamp = None

        # Polling stats
        self.total_markets_fetched = 0
        self.filtered_markets_count = 0
        self.ignored_crypto_count = 0
        self.ignored_sports_count = 0
        self.poll_count = 0
        self.api_errors = 0

        # Initialize AI models
        if USE_SWARM_MODE:
            cprint("[AI] Using SWARM MODE - Multiple AI models", "green")
            try:
                from src.agents.swarm_agent import SwarmAgent
                self.swarm = SwarmAgent()
                cprint("[+] Swarm agent loaded successfully", "green")
            except Exception as e:
                cprint(f"[X] Failed to load swarm agent: {e}", "red")
                cprint("[i] Falling back to single model mode", "yellow")
                self.swarm = None
                self.model = ModelFactory().get_model(AI_MODEL_PROVIDER, AI_MODEL_NAME)
        else:
            cprint(f"[AI] Using single model: {AI_MODEL_PROVIDER}/{AI_MODEL_NAME}", "green")
            self.model = ModelFactory().get_model(AI_MODEL_PROVIDER, AI_MODEL_NAME)
            self.swarm = None

        # Initialize DataFrames
        self.markets_df = self._load_markets()
        self.predictions_df = self._load_predictions()

        cprint(f"[#] Loaded {len(self.markets_df)} existing markets from CSV", "cyan")
        cprint(f"[~] Loaded {len(self.predictions_df)} existing predictions from CSV", "cyan")

        if len(self.predictions_df) > 0:
            unique_runs = self.predictions_df['analysis_run_id'].nunique()
            cprint(f"   └─ {unique_runs} historical analysis runs", "cyan")

        cprint("[+] Initialization complete!\n", "green")

    # ── Data Loading ──────────────────────────────────────────────────────────

    def _load_markets(self):
        """Load existing markets from CSV or create empty DataFrame"""
        if os.path.exists(MARKETS_CSV):
            try:
                df = pd.read_csv(MARKETS_CSV)
                cprint(f"[+] Loaded existing markets from {MARKETS_CSV}", "green")
                return df
            except Exception as e:
                cprint(f"[!] Error loading CSV: {e}", "yellow")

        return pd.DataFrame(columns=[
            'timestamp', 'ticker', 'event_ticker', 'title', 'category',
            'yes_bid', 'yes_ask', 'no_bid', 'no_ask', 'last_price',
            'volume_24h', 'open_interest', 'status', 'close_time',
            'first_seen', 'last_updated', 'last_analyzed'
        ])

    def _load_predictions(self):
        """Load existing predictions from CSV or create empty DataFrame"""
        if os.path.exists(PREDICTIONS_CSV):
            try:
                df = pd.read_csv(PREDICTIONS_CSV)
                cprint(f"[+] Loaded existing predictions from {PREDICTIONS_CSV}", "green")
                return df
            except Exception as e:
                cprint(f"[!] Error loading predictions CSV: {e}", "yellow")

        return pd.DataFrame(columns=[
            'analysis_timestamp', 'analysis_run_id', 'market_title', 'market_ticker',
            'claude_prediction', 'openai_prediction', 'groq_prediction',
            'gemini_prediction', 'deepseek_prediction', 'xai_prediction',
            'ollama_prediction', 'consensus_prediction', 'num_models_responded',
            'market_link'
        ])

    def _save_markets(self):
        """Save markets DataFrame to CSV (thread-safe, silent)"""
        try:
            with self.csv_lock:
                self.markets_df.to_csv(MARKETS_CSV, index=False)
        except Exception as e:
            cprint(f"[X] Error saving CSV: {e}", "red")

    def _save_predictions(self):
        """Save predictions DataFrame to CSV (thread-safe)"""
        try:
            with self.csv_lock:
                self.predictions_df.to_csv(PREDICTIONS_CSV, index=False)
            cprint(f"[S] Saved {len(self.predictions_df)} predictions to CSV", "green")
        except Exception as e:
            cprint(f"[X] Error saving predictions CSV: {e}", "red")

    # ── Kalshi API Methods ────────────────────────────────────────────────────

    def fetch_markets(self, status="open", limit=200) -> list:
        """Fetch markets from Kalshi REST API with cursor-based pagination.

        Args:
            status: Market status filter (open, closed, settled)
            limit: Max markets per page (max 200)

        Returns:
            List of market dicts from the API
        """
        all_markets = []
        cursor = None
        page = 0

        try:
            while True:
                params = {
                    'status': status,
                    'limit': min(limit, 200),
                }
                if cursor:
                    params['cursor'] = cursor

                response = requests.get(
                    f"{KALSHI_API_BASE}/markets",
                    params=params,
                    timeout=30,
                    headers={'Accept': 'application/json'}
                )
                response.raise_for_status()
                data = response.json()

                markets = data.get('markets', [])
                if not markets:
                    break

                all_markets.extend(markets)
                page += 1

                # Check for next page cursor
                cursor = data.get('cursor')
                if not cursor:
                    break

                # Rate limit safety
                time.sleep(REQUEST_DELAY)

                # Safety cap to prevent infinite pagination
                if page >= 50:
                    cprint(f"[!] Hit pagination cap at {len(all_markets)} markets", "yellow")
                    break

            self.total_markets_fetched += len(all_markets)
            return all_markets

        except requests.exceptions.RequestException as e:
            self.api_errors += 1
            cprint(f"[X] Kalshi API error: {e}", "red")
            return []

    def fetch_event(self, event_ticker: str) -> dict:
        """Fetch event details from Kalshi API."""
        try:
            response = requests.get(
                f"{KALSHI_API_BASE}/events/{event_ticker}",
                params={'with_nested_markets': 'true'},
                timeout=15,
                headers={'Accept': 'application/json'}
            )
            response.raise_for_status()
            return response.json().get('event', {})
        except Exception as e:
            cprint(f"[X] Error fetching event {event_ticker}: {e}", "red")
            return {}

    # ── Filtering ─────────────────────────────────────────────────────────────

    def is_near_resolution(self, yes_bid, yes_ask) -> bool:
        """Check if market is near resolution based on bid/ask prices."""
        try:
            bid = float(yes_bid) if yes_bid else 0.5
            ask = float(yes_ask) if yes_ask else 0.5
            midpoint = (bid + ask) / 2
            return midpoint <= IGNORE_PRICE_THRESHOLD or midpoint >= (1.0 - IGNORE_PRICE_THRESHOLD)
        except (ValueError, TypeError):
            return False

    def should_ignore_market(self, title: str) -> tuple:
        """Check if market should be ignored based on category keywords.

        Returns:
            tuple: (should_ignore: bool, reason: str or None)
        """
        title_lower = title.lower()

        for keyword in IGNORE_CRYPTO_KEYWORDS:
            if keyword in title_lower:
                return (True, f"crypto ({keyword})")

        for keyword in IGNORE_SPORTS_KEYWORDS:
            if keyword in title_lower:
                return (True, f"sports ({keyword})")

        return (False, None)

    def is_eligible_market(self, market: dict) -> bool:
        """Check if a Kalshi market passes all filters.

        Args:
            market: Raw market dict from Kalshi API

        Returns:
            True if market should be tracked
        """
        title = market.get('yes_sub_title', '') or market.get('ticker', '')

        # Category filter
        should_ignore, reason = self.should_ignore_market(title)
        if should_ignore:
            if 'crypto' in (reason or ''):
                self.ignored_crypto_count += 1
            elif 'sports' in (reason or ''):
                self.ignored_sports_count += 1
            return False

        # Price filter - near resolution
        yes_bid = market.get('yes_bid_dollars', '0.50')
        yes_ask = market.get('yes_ask_dollars', '0.50')
        if self.is_near_resolution(yes_bid, yes_ask):
            return False

        # Volume filter
        try:
            volume_24h = float(market.get('volume_24h_fp', '0') or '0')
            if volume_24h < MIN_VOLUME_24H:
                return False
        except (ValueError, TypeError):
            return False

        # Open interest filter
        try:
            open_interest = float(market.get('open_interest_fp', '0') or '0')
            if open_interest < MIN_OPEN_INTEREST:
                return False
        except (ValueError, TypeError):
            return False

        # Close time filter
        close_time_str = market.get('close_time')
        if close_time_str:
            try:
                close_time = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
                now = datetime.now(close_time.tzinfo) if close_time.tzinfo else datetime.now()
                days_to_close = (close_time - now).days
                if days_to_close < MIN_DAYS_TO_CLOSE or days_to_close > MAX_DAYS_TO_CLOSE:
                    return False
            except (ValueError, TypeError):
                pass

        # Status must be active/open
        status = market.get('status', '')
        if status not in ('active', 'open', ''):
            return False

        self.filtered_markets_count += 1
        return True

    # ── Market Processing ─────────────────────────────────────────────────────

    def process_markets(self, markets: list):
        """Process fetched Kalshi markets and update the DataFrame.

        Args:
            markets: List of market dicts from Kalshi API
        """
        if not markets:
            return

        new_markets = 0
        updated_markets = 0

        for market in markets:
            try:
                ticker = market.get('ticker', '')
                if not ticker:
                    continue

                # Extract market data
                event_ticker = market.get('event_ticker', '')
                title = market.get('yes_sub_title', '') or ticker
                category = market.get('category', '')
                yes_bid = float(market.get('yes_bid_dollars', '0') or '0')
                yes_ask = float(market.get('yes_ask_dollars', '0') or '0')
                no_bid = float(market.get('no_bid_dollars', '0') or '0')
                no_ask = float(market.get('no_ask_dollars', '0') or '0')
                last_price = float(market.get('last_price_dollars', '0') or '0')
                volume_24h = float(market.get('volume_24h_fp', '0') or '0')
                open_interest = float(market.get('open_interest_fp', '0') or '0')
                status = market.get('status', '')
                close_time = market.get('close_time', '')
                now_iso = datetime.now().isoformat()

                # Check if market already exists
                if ticker in self.markets_df['ticker'].values:
                    # Update existing market with fresh data
                    mask = self.markets_df['ticker'] == ticker
                    self.markets_df.loc[mask, 'timestamp'] = now_iso
                    self.markets_df.loc[mask, 'yes_bid'] = yes_bid
                    self.markets_df.loc[mask, 'yes_ask'] = yes_ask
                    self.markets_df.loc[mask, 'no_bid'] = no_bid
                    self.markets_df.loc[mask, 'no_ask'] = no_ask
                    self.markets_df.loc[mask, 'last_price'] = last_price
                    self.markets_df.loc[mask, 'volume_24h'] = volume_24h
                    self.markets_df.loc[mask, 'open_interest'] = open_interest
                    self.markets_df.loc[mask, 'status'] = status
                    self.markets_df.loc[mask, 'last_updated'] = now_iso
                    updated_markets += 1
                    continue

                # Add new market
                new_market = {
                    'timestamp': now_iso,
                    'ticker': ticker,
                    'event_ticker': event_ticker,
                    'title': title,
                    'category': category,
                    'yes_bid': yes_bid,
                    'yes_ask': yes_ask,
                    'no_bid': no_bid,
                    'no_ask': no_ask,
                    'last_price': last_price,
                    'volume_24h': volume_24h,
                    'open_interest': open_interest,
                    'status': status,
                    'close_time': close_time,
                    'first_seen': now_iso,
                    'last_updated': now_iso,
                    'last_analyzed': None,
                }

                self.markets_df = pd.concat([
                    self.markets_df,
                    pd.DataFrame([new_market])
                ], ignore_index=True)

                new_markets += 1
                cprint(f"[+] NEW: {title[:70]}", "green")

            except Exception as e:
                cprint(f"[!] Error processing market: {e}", "yellow")
                continue

        # Save if changed
        if new_markets > 0 or updated_markets > 0:
            self._save_markets()
            if updated_markets > 0:
                cprint(f"[~] Updated {updated_markets} existing markets with fresh data", "cyan")

    # ── Display ───────────────────────────────────────────────────────────────

    def display_recent_markets(self):
        """Display the most recent markets from CSV"""
        if len(self.markets_df) == 0:
            cprint("\n[#] No markets in database yet", "yellow")
            return

        cprint("\n" + "="*80, "cyan")
        cprint(f"[#] Most Recent {min(MARKETS_TO_DISPLAY, len(self.markets_df))} Kalshi Markets", "cyan", attrs=['bold'])
        cprint("="*80, "cyan")

        recent = self.markets_df.tail(MARKETS_TO_DISPLAY)

        for idx, row in recent.iterrows():
            title = row['title'][:60] + "..." if len(str(row['title'])) > 60 else row['title']
            yes_bid = row.get('yes_bid', 0)
            yes_ask = row.get('yes_ask', 0)
            volume = row.get('volume_24h', 0)

            cprint(f"\n[$] Bid: ${yes_bid:.2f} / Ask: ${yes_ask:.2f}  |  Vol: ${volume:,.0f}", "yellow")
            cprint(f"[-] {title}", "white")
            cprint(f"[L] https://kalshi.com/markets/{row['ticker']}", "cyan")

        cprint("\n" + "="*80, "cyan")
        cprint(f"Total Kalshi markets tracked: {len(self.markets_df)}", "green", attrs=['bold'])
        cprint("="*80 + "\n", "cyan")

    # ── AI Predictions ────────────────────────────────────────────────────────

    def get_ai_predictions(self):
        """Get AI predictions for recent Kalshi markets"""
        if len(self.markets_df) == 0:
            cprint("\n[!] No markets to analyze yet", "yellow")
            return

        markets_to_analyze = self.markets_df.tail(MARKETS_TO_ANALYZE)

        analysis_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_timestamp = datetime.now().isoformat()

        cprint("\n" + "="*80, "magenta")
        cprint(f"[AI] AI Analysis - Analyzing {len(markets_to_analyze)} Kalshi markets", "magenta", attrs=['bold'])
        cprint(f"[#] Analysis Run ID: {analysis_run_id}", "magenta")
        cprint(f"[$] Price info to AI: {'[+] ENABLED' if SEND_PRICE_INFO_TO_AI else '[X] DISABLED'}", "green" if SEND_PRICE_INFO_TO_AI else "yellow")
        cprint("="*80, "magenta")

        # Build prompt
        if SEND_PRICE_INFO_TO_AI:
            markets_text = "\n\n".join([
                f"Market {i+1}:\n"
                f"Title: {row['title']}\n"
                f"YES Bid: ${row['yes_bid']:.2f} / YES Ask: ${row['yes_ask']:.2f}\n"
                f"Last Price: ${row['last_price']:.2f} ({row['last_price']*100:.1f}% implied)\n"
                f"24h Volume: ${row['volume_24h']:,.0f}\n"
                f"Link: https://kalshi.com/markets/{row['ticker']}"
                for i, (_, row) in enumerate(markets_to_analyze.iterrows())
            ])
        else:
            markets_text = "\n\n".join([
                f"Market {i+1}:\n"
                f"Title: {row['title']}\n"
                f"24h Volume: ${row['volume_24h']:,.0f}\n"
                f"Link: https://kalshi.com/markets/{row['ticker']}"
                for i, (_, row) in enumerate(markets_to_analyze.iterrows())
            ])

        user_prompt = f"""Analyze these {len(markets_to_analyze)} Kalshi prediction markets and provide your predictions:

{markets_text}

Provide predictions for each market in the specified format."""

        if USE_SWARM_MODE and self.swarm:
            cprint("\n[~] Getting predictions from AI swarm (120s timeout per model)...\n", "cyan")

            swarm_result = self.swarm.query(
                prompt=user_prompt,
                system_prompt=MARKET_ANALYSIS_SYSTEM_PROMPT
            )

            if not swarm_result or not swarm_result.get('responses'):
                cprint("[X] No responses from swarm - all models failed or timed out", "red")
                return

            successful_responses = [
                name for name, data in swarm_result.get('responses', {}).items()
                if data.get('success')
            ]

            if not successful_responses:
                cprint("[X] All AI models failed - no predictions available", "red")
                return

            cprint(f"\n[+] Received {len(successful_responses)}/{len(swarm_result['responses'])} successful responses!\n", "green", attrs=['bold'])

            # Display individual responses
            cprint("="*80, "yellow")
            cprint("[AI] Individual AI Predictions", "yellow", attrs=['bold'])
            cprint("="*80, "yellow")

            for model_name, model_data in swarm_result.get('responses', {}).items():
                if model_data.get('success'):
                    response_time = model_data.get('response_time', 0)
                    cprint(f"\n{'='*80}", "cyan")
                    cprint(f"[+] {model_name.upper()} ({response_time:.1f}s)", "cyan", attrs=['bold'])
                    cprint(f"{'='*80}", "cyan")
                    cprint(model_data.get('response', 'No response'), "white")
                else:
                    error = model_data.get('error', 'Unknown error')
                    cprint(f"\n[X] {model_name.upper()} - FAILED: {error}", "red", attrs=['bold'])

            # Calculate consensus
            consensus_text = self._calculate_consensus(swarm_result, markets_to_analyze)

            cprint("\n" + "="*80, "green")
            cprint("[>] CONSENSUS ANALYSIS", "green", attrs=['bold'])
            cprint(f"Based on {len(successful_responses)} AI models", "green")
            cprint("="*80, "green")
            cprint(consensus_text, "white")
            cprint("="*80 + "\n", "green")

            # Get top consensus picks
            self._get_top_consensus_picks(swarm_result, markets_to_analyze)

            # Save predictions
            try:
                self._save_swarm_predictions(
                    analysis_run_id=analysis_run_id,
                    analysis_timestamp=analysis_timestamp,
                    markets=markets_to_analyze,
                    swarm_result=swarm_result
                )
                cprint(f"\n[F] Predictions saved to: {PREDICTIONS_CSV}", "cyan", attrs=['bold'])
            except Exception as e:
                cprint(f"[X] Error saving predictions: {e}", "red")
                import traceback
                traceback.print_exc()

            # Mark markets as analyzed
            self._mark_markets_analyzed(markets_to_analyze, analysis_timestamp)
        else:
            # Single model mode
            cprint(f"\n[AI] Getting predictions from {AI_MODEL_PROVIDER}/{AI_MODEL_NAME}...\n", "cyan")

            try:
                response = self.model.generate_response(
                    system_prompt=MARKET_ANALYSIS_SYSTEM_PROMPT,
                    user_content=user_prompt,
                    temperature=0.7
                )

                cprint("="*80, "green")
                cprint("[>] AI PREDICTION", "green", attrs=['bold'])
                cprint("="*80, "green")
                cprint(response.content, "white")
                cprint("="*80 + "\n", "green")

                prediction_summary = response.content.split('\n')[0][:200] if response.content else 'No response'
                prediction_record = {
                    'analysis_timestamp': analysis_timestamp,
                    'analysis_run_id': analysis_run_id,
                    'market_title': f"Analyzed {len(markets_to_analyze)} markets",
                    'market_ticker': 'batch_analysis',
                    'claude_prediction': 'N/A',
                    'openai_prediction': 'N/A',
                    'groq_prediction': 'N/A',
                    'gemini_prediction': 'N/A',
                    'deepseek_prediction': 'N/A',
                    'xai_prediction': prediction_summary if AI_MODEL_PROVIDER == 'xai' else 'N/A',
                    'ollama_prediction': 'N/A',
                    'consensus_prediction': prediction_summary,
                    'num_models_responded': 1,
                    'market_link': ''
                }

                self.predictions_df = pd.concat([
                    self.predictions_df,
                    pd.DataFrame([prediction_record])
                ], ignore_index=True)
                self._save_predictions()

                self._mark_markets_analyzed(markets_to_analyze, analysis_timestamp)

            except Exception as e:
                cprint(f"[X] Error getting prediction: {e}", "red")

    # ── Consensus Methods ─────────────────────────────────────────────────────

    def _calculate_consensus(self, swarm_result, markets_df):
        """Calculate consensus from swarm responses."""
        try:
            market_votes = {}
            model_predictions = {}

            for provider, data in swarm_result["responses"].items():
                if not data["success"]:
                    continue

                response_text = data["response"]
                model_predictions[provider] = response_text

                lines = response_text.strip().split('\n')
                for line in lines:
                    line_upper = line.upper()

                    if 'MARKET' in line_upper and ':' in line:
                        try:
                            market_part = line_upper.split('MARKET')[1].split(':')[0].strip()
                            market_num = int(''.join(filter(str.isdigit, market_part)))

                            if market_num < 1 or market_num > len(markets_df):
                                continue

                            if market_num not in market_votes:
                                market_votes[market_num] = {"YES": 0, "NO": 0, "NO_TRADE": 0}

                            if 'NO_TRADE' in line_upper or 'NO TRADE' in line_upper:
                                market_votes[market_num]["NO_TRADE"] += 1
                            elif 'YES' in line_upper:
                                market_votes[market_num]["YES"] += 1
                            elif 'NO' in line_upper:
                                market_votes[market_num]["NO"] += 1
                        except:
                            continue

            total_models = len(model_predictions)

            if total_models == 0:
                return "No valid model responses to analyze"

            consensus_text = f"Analyzed responses from {total_models} AI models\n\n"

            if market_votes:
                consensus_text += "MARKET CONSENSUS:\n"
                consensus_text += "="*80 + "\n\n"

                markets_list = list(markets_df.iterrows())

                for market_num in sorted(market_votes.keys()):
                    votes = market_votes[market_num]
                    total_votes = sum(votes.values())

                    if total_votes == 0:
                        continue

                    majority = max(votes, key=votes.get)
                    majority_count = votes[majority]
                    confidence = int((majority_count / total_votes) * 100)

                    if 1 <= market_num <= len(markets_list):
                        idx, row = markets_list[market_num - 1]
                        market_title = row['title']
                        market_ticker = row['ticker']
                        market_link = f"https://kalshi.com/markets/{market_ticker}"

                        display_title = market_title[:70] + "..." if len(str(market_title)) > 70 else market_title

                        consensus_text += f"Market {market_num}: {majority} ({confidence}% consensus)\n"
                        consensus_text += f"  [-] {display_title}\n"
                        consensus_text += f"  [L] {market_link}\n"
                        consensus_text += f"  Votes: YES: {votes['YES']} | NO: {votes['NO']} | NO_TRADE: {votes['NO_TRADE']}\n\n"
                    else:
                        consensus_text += f"Market {market_num}: {majority} ({confidence}% consensus)\n"
                        consensus_text += f"  YES: {votes['YES']} | NO: {votes['NO']} | NO_TRADE: {votes['NO_TRADE']}\n\n"
            else:
                consensus_text += "[!] Could not extract structured market predictions from responses\n"

            consensus_text += "\nRESPONDED MODELS:\n"
            consensus_text += "="*60 + "\n"
            for model_name in model_predictions.keys():
                consensus_text += f"  [+] {model_name}\n"

            failed_models = [
                provider for provider, data in swarm_result["responses"].items()
                if not data["success"]
            ]
            if failed_models:
                consensus_text += "\nFAILED/TIMEOUT MODELS:\n"
                consensus_text += "="*60 + "\n"
                for model_name in failed_models:
                    error = swarm_result["responses"][model_name].get("error", "Unknown")
                    consensus_text += f"  [X] {model_name}: {error}\n"

            return consensus_text

        except Exception as e:
            cprint(f"[X] Error calculating consensus: {e}", "red")
            import traceback
            traceback.print_exc()
            return f"Error calculating consensus: {str(e)}"

    def _get_top_consensus_picks(self, swarm_result, markets_df):
        """Use consensus AI to identify top markets with strongest agreement."""
        try:
            cprint("\n" + "="*80, "yellow")
            cprint(f"[AI] Running Consensus AI to identify top {TOP_MARKETS_COUNT} picks...", "yellow", attrs=['bold'])
            cprint("="*80 + "\n", "yellow")

            all_responses_text = ""
            for model_name, model_data in swarm_result.get('responses', {}).items():
                if model_data.get('success'):
                    all_responses_text += f"\n{'='*60}\n"
                    all_responses_text += f"{model_name.upper()} PREDICTIONS:\n"
                    all_responses_text += f"{'='*60}\n"
                    all_responses_text += model_data.get('response', '') + "\n"

            markets_list = list(markets_df.iterrows())
            market_reference = "\n".join([
                f"Market {i+1}: {row['title']}\nLink: https://kalshi.com/markets/{row['ticker']}"
                for i, (_, row) in enumerate(markets_list)
            ])

            consensus_prompt = CONSENSUS_AI_PROMPT_TEMPLATE.format(
                market_reference=market_reference,
                all_responses=all_responses_text,
                top_count=TOP_MARKETS_COUNT
            )

            consensus_model = ModelFactory().get_model('claude', 'claude-sonnet-4-5')

            cprint("[.] Analyzing all responses for strongest consensus...\n", "cyan")

            response = consensus_model.generate_response(
                system_prompt="You are a consensus analyzer that identifies the strongest agreements across multiple AI predictions. Be concise and clear.",
                user_content=consensus_prompt,
                temperature=0.3,
                max_tokens=1000
            )

            cprint("\n" + "="*80, "white", "on_blue", attrs=['bold'])
            cprint(f"[#1] TOP {TOP_MARKETS_COUNT} CONSENSUS PICKS - KALSHI AGENT", "white", "on_blue", attrs=['bold'])
            cprint("="*80, "white", "on_blue", attrs=['bold'])
            cprint("", "white")
            cprint(response.content, "cyan", attrs=['bold'])
            cprint("\n" + "="*80, "white", "on_blue", attrs=['bold'])
            cprint("="*80 + "\n", "white", "on_blue", attrs=['bold'])

            self._save_consensus_picks_to_csv(response.content, markets_df)

        except Exception as e:
            cprint(f"[X] Error getting top consensus picks: {e}", "red")
            import traceback
            traceback.print_exc()

    def _save_consensus_picks_to_csv(self, consensus_response, markets_df):
        """Save top consensus picks to dedicated CSV (append-only)."""
        try:
            cprint("\n[S] Saving top consensus picks to CSV...", "cyan")

            picks = []
            lines = consensus_response.split('\n')

            current_pick = {}
            for line in lines:
                line = line.strip()

                market_match = re.match(r'(\d+)\.\s+Market\s+(\d+):\s+(.+)', line)
                if market_match:
                    if current_pick:
                        picks.append(current_pick)

                    rank = market_match.group(1)
                    market_num = int(market_match.group(2))
                    title = market_match.group(3)

                    current_pick = {
                        'rank': rank,
                        'market_number': market_num,
                        'market_title': title
                    }

                elif line.startswith('Side:'):
                    current_pick['side'] = line.replace('Side:', '').strip()

                elif line.startswith('Consensus:'):
                    consensus_text = line.replace('Consensus:', '').strip()
                    current_pick['consensus'] = consensus_text
                    consensus_match = re.search(r'(\d+)\s+out\s+of\s+(\d+)', consensus_text)
                    if consensus_match:
                        current_pick['consensus_count'] = int(consensus_match.group(1))
                        current_pick['total_models'] = int(consensus_match.group(2))

                elif line.startswith('Link:'):
                    current_pick['link'] = line.replace('Link:', '').strip()

                elif line.startswith('Reasoning:'):
                    current_pick['reasoning'] = line.replace('Reasoning:', '').strip()

            if current_pick:
                picks.append(current_pick)

            if not picks:
                cprint("[!] Could not parse consensus picks from response", "yellow")
                return

            timestamp = datetime.now().isoformat()
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            records = []
            for pick in picks:
                record = {
                    'timestamp': timestamp,
                    'run_id': run_id,
                    'rank': pick.get('rank', ''),
                    'market_number': pick.get('market_number', ''),
                    'market_title': pick.get('market_title', ''),
                    'side': pick.get('side', ''),
                    'consensus': pick.get('consensus', ''),
                    'consensus_count': pick.get('consensus_count', ''),
                    'total_models': pick.get('total_models', ''),
                    'reasoning': pick.get('reasoning', ''),
                    'link': pick.get('link', '')
                }
                records.append(record)

            if os.path.exists(CONSENSUS_PICKS_CSV):
                consensus_df = pd.read_csv(CONSENSUS_PICKS_CSV)
            else:
                consensus_df = pd.DataFrame(columns=[
                    'timestamp', 'run_id', 'rank', 'market_number', 'market_title',
                    'side', 'consensus', 'consensus_count', 'total_models', 'reasoning',
                    'link'
                ])

            consensus_df = pd.concat([
                consensus_df,
                pd.DataFrame(records)
            ], ignore_index=True)

            with self.csv_lock:
                consensus_df.to_csv(CONSENSUS_PICKS_CSV, index=False)

            cprint(f"[+] Saved {len(records)} consensus picks to CSV", "green")
            cprint(f"[F] Consensus picks CSV: {CONSENSUS_PICKS_CSV}", "cyan", attrs=['bold'])
            cprint(f"[#] Total consensus picks in history: {len(consensus_df)}", "cyan")

        except Exception as e:
            cprint(f"[X] Error saving consensus picks: {e}", "red")
            import traceback
            traceback.print_exc()

    def _save_swarm_predictions(self, analysis_run_id, analysis_timestamp, markets, swarm_result):
        """Save swarm predictions to CSV database (one row per market)."""
        try:
            cprint("\n[S] Saving predictions to database...", "cyan")

            market_predictions = {}

            for model_name, model_data in swarm_result.get('responses', {}).items():
                if not model_data.get('success'):
                    continue

                response = model_data.get('response', '')
                lines = response.strip().split('\n')

                for line in lines:
                    line_upper = line.upper()

                    if 'MARKET' in line_upper and ':' in line:
                        try:
                            market_part = line_upper.split('MARKET')[1].split(':')[0].strip()
                            market_num = int(''.join(filter(str.isdigit, market_part)))

                            if market_num < 1 or market_num > len(markets):
                                continue

                            if market_num not in market_predictions:
                                market_predictions[market_num] = {}

                            if 'NO_TRADE' in line_upper or 'NO TRADE' in line_upper:
                                market_predictions[market_num][model_name] = 'NO_TRADE'
                            elif 'YES' in line_upper:
                                market_predictions[market_num][model_name] = 'YES'
                            elif 'NO' in line_upper:
                                market_predictions[market_num][model_name] = 'NO'
                        except:
                            continue

            markets_list = list(markets.iterrows())
            new_records = []

            for market_num, predictions in market_predictions.items():
                if 1 <= market_num <= len(markets_list):
                    idx, row = markets_list[market_num - 1]
                    market_title = row['title']
                    market_ticker = row['ticker']
                    market_link = f"https://kalshi.com/markets/{market_ticker}"

                    votes = {"YES": 0, "NO": 0, "NO_TRADE": 0}
                    for pred in predictions.values():
                        if pred in votes:
                            votes[pred] += 1

                    majority = max(votes, key=votes.get)
                    total = sum(votes.values())
                    confidence = int((votes[majority] / total) * 100) if total > 0 else 0
                    consensus = f"{majority} ({confidence}%)"

                    record = {
                        'analysis_timestamp': analysis_timestamp,
                        'analysis_run_id': analysis_run_id,
                        'market_title': market_title,
                        'market_ticker': market_ticker,
                        'claude_prediction': predictions.get('claude', 'N/A'),
                        'openai_prediction': predictions.get('openai', 'N/A'),
                        'groq_prediction': predictions.get('groq', 'N/A'),
                        'gemini_prediction': predictions.get('gemini', 'N/A'),
                        'deepseek_prediction': predictions.get('deepseek', 'N/A'),
                        'xai_prediction': predictions.get('xai', 'N/A'),
                        'ollama_prediction': predictions.get('ollama', 'N/A'),
                        'consensus_prediction': consensus,
                        'num_models_responded': len(predictions),
                        'market_link': market_link
                    }
                    new_records.append(record)

            if new_records:
                self.predictions_df = pd.concat([
                    self.predictions_df,
                    pd.DataFrame(new_records)
                ], ignore_index=True)
                self._save_predictions()
                cprint(f"[+] Saved {len(new_records)} market predictions (run {analysis_run_id})", "green")
            else:
                cprint("[!] No structured predictions found to save", "yellow")

        except Exception as e:
            cprint(f"[X] Error saving predictions: {e}", "red")
            import traceback
            traceback.print_exc()

    def _mark_markets_analyzed(self, markets, analysis_timestamp):
        """Mark markets as analyzed with timestamp."""
        try:
            cprint("\n[T] Marking markets as analyzed...", "cyan")

            analyzed_tickers = markets['ticker'].tolist()

            for ticker in analyzed_tickers:
                mask = self.markets_df['ticker'] == ticker
                self.markets_df.loc[mask, 'last_analyzed'] = analysis_timestamp

            self._save_markets()

            cprint(f"[+] Marked {len(analyzed_tickers)} markets with analysis timestamp", "green")
            cprint(f"   Next re-analysis eligible after: {REANALYSIS_HOURS}h", "cyan")

        except Exception as e:
            cprint(f"[X] Error marking markets as analyzed: {e}", "red")
            import traceback
            traceback.print_exc()

    # ── Main Loops ────────────────────────────────────────────────────────────

    def poll_loop(self):
        """REST polling loop - fetches Kalshi markets at regular intervals."""
        cprint("\n[#] POLL THREAD STARTED", "cyan", attrs=['bold'])
        cprint(f"[*] Fetching markets every {POLL_INTERVAL_SECONDS} seconds\n", "cyan")

        while True:
            try:
                self.poll_count += 1
                markets = self.fetch_markets()
                filtered = [m for m in markets if self.is_eligible_market(m)]
                self.process_markets(filtered)

                time.sleep(POLL_INTERVAL_SECONDS)

            except KeyboardInterrupt:
                break
            except Exception as e:
                cprint(f"[X] Error in poll loop: {e}", "red")
                time.sleep(POLL_INTERVAL_SECONDS)

    def status_display_loop(self):
        """Display status updates every 30 seconds."""
        cprint("\n[#] STATUS DISPLAY THREAD STARTED", "cyan", attrs=['bold'])

        while True:
            try:
                time.sleep(30)

                total_markets = len(self.markets_df)

                now = datetime.now()
                cutoff_time = now - timedelta(hours=REANALYSIS_HOURS)
                fresh_eligible_count = 0

                for idx, row in self.markets_df.iterrows():
                    last_analyzed = row.get('last_analyzed')
                    last_updated = row.get('last_updated')

                    is_eligible = False
                    if pd.isna(last_analyzed) or last_analyzed is None:
                        is_eligible = True
                    else:
                        try:
                            analyzed_time = pd.to_datetime(last_analyzed)
                            if analyzed_time < cutoff_time:
                                is_eligible = True
                        except:
                            is_eligible = True

                    has_fresh_data = False
                    if self.last_analysis_run_timestamp is None:
                        has_fresh_data = not pd.isna(last_updated) and last_updated is not None
                    else:
                        try:
                            if not pd.isna(last_updated) and last_updated is not None:
                                update_time = pd.to_datetime(last_updated)
                                last_run_time = pd.to_datetime(self.last_analysis_run_timestamp)
                                if update_time > last_run_time:
                                    has_fresh_data = True
                        except:
                            pass

                    if is_eligible and has_fresh_data:
                        fresh_eligible_count += 1

                cprint(f"\n{'='*60}", "cyan")
                cprint(f"[#] Kalshi Agent Status @ {datetime.now().strftime('%H:%M:%S')}", "cyan", attrs=['bold'])
                cprint(f"{'='*60}", "cyan")
                cprint(f"   Poll count: {self.poll_count}", "white")
                cprint(f"   Total markets fetched (all time): {self.total_markets_fetched}", "white")
                cprint(f"   API errors: {self.api_errors}", "red" if self.api_errors > 0 else "white")
                cprint(f"   Ignored crypto: {self.ignored_crypto_count}", "red")
                cprint(f"   Ignored sports: {self.ignored_sports_count}", "red")
                cprint(f"   Filtered markets (passed all checks): {self.filtered_markets_count}", "yellow")
                cprint(f"   Total markets in database: {total_markets}", "white")
                cprint(f"   Fresh eligible markets: {fresh_eligible_count}", "yellow" if fresh_eligible_count < NEW_MARKETS_FOR_ANALYSIS else "green", attrs=['bold'])

                if fresh_eligible_count >= NEW_MARKETS_FOR_ANALYSIS:
                    cprint(f"   [+] Ready for analysis! (Have {fresh_eligible_count}, need {NEW_MARKETS_FOR_ANALYSIS})", "green", attrs=['bold'])
                else:
                    cprint(f"   [.] Collecting... (Have {fresh_eligible_count}, need {NEW_MARKETS_FOR_ANALYSIS})", "yellow")

                cprint(f"{'='*60}\n", "cyan")

            except KeyboardInterrupt:
                break
            except Exception as e:
                cprint(f"[X] Error in status display loop: {e}", "red")

    def analysis_cycle(self):
        """Check if we have enough eligible markets and run AI analysis."""
        cprint("\n" + "="*80, "magenta")
        cprint("[AI] KALSHI ANALYSIS CYCLE CHECK", "magenta", attrs=['bold'])
        cprint(f"[T] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "magenta")
        cprint("="*80 + "\n", "magenta")

        with self.csv_lock:
            self.markets_df = self._load_markets()

        total_markets = len(self.markets_df)

        if total_markets == 0:
            cprint(f"\n[.] No markets in database yet! Polling is collecting...", "yellow", attrs=['bold'])
            return

        now = datetime.now()
        cutoff_time = now - timedelta(hours=REANALYSIS_HOURS)

        fresh_eligible_count = 0
        for idx, row in self.markets_df.iterrows():
            last_analyzed = row.get('last_analyzed')
            last_updated = row.get('last_updated')

            is_eligible = False
            if pd.isna(last_analyzed) or last_analyzed is None:
                is_eligible = True
            else:
                try:
                    analyzed_time = pd.to_datetime(last_analyzed)
                    if analyzed_time < cutoff_time:
                        is_eligible = True
                except:
                    is_eligible = True

            has_fresh_data = False
            if self.last_analysis_run_timestamp is None:
                has_fresh_data = not pd.isna(last_updated) and last_updated is not None
            else:
                try:
                    if not pd.isna(last_updated) and last_updated is not None:
                        update_time = pd.to_datetime(last_updated)
                        last_run_time = pd.to_datetime(self.last_analysis_run_timestamp)
                        if update_time > last_run_time:
                            has_fresh_data = True
                except:
                    pass

            if is_eligible and has_fresh_data:
                fresh_eligible_count += 1

        is_first_run = (self.last_analysis_run_timestamp is None)

        cprint(f"[#] Market Analysis Status:", "cyan", attrs=['bold'])
        cprint(f"   Total markets in database: {total_markets}", "white")
        cprint(f"   Fresh eligible markets: {fresh_eligible_count}", "yellow" if fresh_eligible_count < NEW_MARKETS_FOR_ANALYSIS else "green", attrs=['bold'])

        if is_first_run:
            cprint(f"[>] FIRST ANALYSIS RUN - will analyze whatever markets we have", "yellow", attrs=['bold'])
        else:
            if fresh_eligible_count >= NEW_MARKETS_FOR_ANALYSIS:
                cprint(f"   [+] REQUIREMENT MET - Running analysis!", "green", attrs=['bold'])
            else:
                cprint(f"   [X] Need {NEW_MARKETS_FOR_ANALYSIS - fresh_eligible_count} more fresh eligible markets", "yellow", attrs=['bold'])

        should_analyze = (is_first_run and total_markets > 0) or (fresh_eligible_count >= NEW_MARKETS_FOR_ANALYSIS)

        if should_analyze:
            if is_first_run:
                cprint(f"\n[+] First run with {total_markets} markets! Running initial AI analysis...\n", "green", attrs=['bold'])
            else:
                cprint(f"\n[+] {fresh_eligible_count} fresh eligible markets! Running AI analysis...\n", "green", attrs=['bold'])

            self.display_recent_markets()
            self.get_ai_predictions()

            self.last_analysis_run_timestamp = datetime.now().isoformat()
            self.last_analyzed_count = total_markets
            cprint(f"\n[S] Updated analysis tracker: {self.last_analyzed_count} markets in database", "green")
        else:
            needed = NEW_MARKETS_FOR_ANALYSIS - fresh_eligible_count
            cprint(f"\n[.] Need {needed} more fresh eligible markets before next analysis", "yellow")

        cprint("\n" + "="*80, "green")
        cprint("[+] Kalshi analysis check complete!", "green", attrs=['bold'])
        cprint("="*80 + "\n", "green")

    def analysis_loop(self):
        """Continuously check for new markets to analyze."""
        cprint("\n[AI] KALSHI ANALYSIS THREAD STARTED", "magenta", attrs=['bold'])
        cprint(f"[AI] Running first analysis NOW, then checking every {ANALYSIS_CHECK_INTERVAL_SECONDS}s\n", "magenta")

        while True:
            try:
                self.analysis_cycle()

                next_check = datetime.now() + timedelta(seconds=ANALYSIS_CHECK_INTERVAL_SECONDS)
                cprint(f"[T] Next analysis check at: {next_check.strftime('%H:%M:%S')}\n", "magenta")

                time.sleep(ANALYSIS_CHECK_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                break
            except Exception as e:
                cprint(f"[X] Error in analysis loop: {e}", "red")
                import traceback
                traceback.print_exc()
                time.sleep(ANALYSIS_CHECK_INTERVAL_SECONDS)


def main():
    """Kalshi Agent Main - REST polling + AI analysis threads"""
    cprint("\n" + "="*80, "cyan")
    cprint("[*] Kalshi Prediction Market Agent - REST Polling Edition", "cyan", attrs=['bold'])
    cprint("="*80, "cyan")
    cprint(f"[>] API: {KALSHI_API_BASE}", "cyan")
    cprint(f"[T] Poll interval: {POLL_INTERVAL_SECONDS}s", "yellow")
    cprint(f"[$] Min 24h volume: ${MIN_VOLUME_24H:,}", "yellow")
    cprint(f"[$] Min open interest: ${MIN_OPEN_INTEREST:,}", "yellow")
    cprint(f"[X] Ignoring prices within {IGNORE_PRICE_THRESHOLD:.2f} of $0 or $1", "yellow")
    cprint(f"[X] Filtering crypto ({len(IGNORE_CRYPTO_KEYWORDS)} keywords) + sports ({len(IGNORE_SPORTS_KEYWORDS)} keywords)", "red")
    cprint(f"[T] Market close window: {MIN_DAYS_TO_CLOSE}-{MAX_DAYS_TO_CLOSE} days", "yellow")
    cprint("", "yellow")
    cprint(f"[AI] AI Mode: {'SWARM (multiple models)' if USE_SWARM_MODE else 'Single Model'}", "yellow")
    cprint(f"[$] Price Info to AI: {'ENABLED' if SEND_PRICE_INFO_TO_AI else 'DISABLED'}", "green" if SEND_PRICE_INFO_TO_AI else "yellow")
    cprint(f"[AI] Analysis: every {ANALYSIS_CHECK_INTERVAL_SECONDS}s, triggers at {NEW_MARKETS_FOR_ANALYSIS} fresh markets", "magenta")
    cprint("", "yellow")
    cprint("[F] Data Files:", "cyan", attrs=['bold'])
    cprint(f"   Markets: {MARKETS_CSV}", "white")
    cprint(f"   Predictions: {PREDICTIONS_CSV}", "white")
    cprint(f"   Consensus: {CONSENSUS_PICKS_CSV}", "white")
    cprint("="*80 + "\n", "cyan")

    # Initialize agent
    agent = KalshiAgent()

    # Initial market fetch
    cprint("\n" + "="*80, "yellow")
    cprint("[H] Fetching initial Kalshi markets...", "yellow", attrs=['bold'])
    cprint("="*80, "yellow")

    markets = agent.fetch_markets()
    filtered = [m for m in markets if agent.is_eligible_market(m)]
    agent.process_markets(filtered)
    cprint(f"[+] Database populated with {len(agent.markets_df)} markets ({len(filtered)} passed filters out of {len(markets)} total)", "green")
    cprint("="*80 + "\n", "yellow")

    # Start threads
    poll_thread = threading.Thread(target=agent.poll_loop, daemon=True, name="KalshiPoll")
    status_thread = threading.Thread(target=agent.status_display_loop, daemon=True, name="KalshiStatus")
    analysis_thread = threading.Thread(target=agent.analysis_loop, daemon=True, name="KalshiAnalysis")

    try:
        cprint("[>] Starting Kalshi agent threads...\n", "green", attrs=['bold'])
        poll_thread.start()
        status_thread.start()
        analysis_thread.start()

        cprint("[+] Kalshi Agent running! Press Ctrl+C to stop.\n", "green", attrs=['bold'])
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        cprint("\n\n" + "="*80, "yellow")
        cprint("[!] Kalshi Agent stopped by user", "yellow", attrs=['bold'])
        cprint("="*80 + "\n", "yellow")
        sys.exit(0)


if __name__ == "__main__":
    main()
