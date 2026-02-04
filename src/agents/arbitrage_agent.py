"""
Arbitrage Agent - Cross-platform opportunity detection between Polymarket and Kalshi.
Fuzzy-matches markets, validates equivalence with AI, and calculates price spreads.
NO ACTUAL TRADING - detection and analysis only.
"""

import os
import sys
import time
import re
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

try:
    from thefuzz import fuzz
except ImportError:
    cprint("[!] thefuzz not installed. Run: pip install thefuzz python-Levenshtein", "red")
    sys.exit(1)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Arbitrage detection thresholds
SPREAD_THRESHOLD_CENTS = 5          # Minimum spread in cents to flag (0.05)
STRONG_OPPORTUNITY_CENTS = 10       # "Strong" opportunity threshold (0.10)
FUZZY_MATCH_THRESHOLD = 65          # Minimum fuzz ratio for title matching (0-100)
AI_EQUIVALENCE_THRESHOLD = 0.8      # Min AI confidence that markets are equivalent

# Fee structure (conservative estimates)
POLYMARKET_FEE_PERCENT = 2.0        # Polymarket trading fee
KALSHI_FEE_PERCENT = 1.0            # Kalshi fee on profit

# Scanning
SCAN_INTERVAL_SECONDS = 600         # Check for arbitrage every 10 minutes
MAX_MARKETS_TO_COMPARE = 200        # Cap to prevent combinatorial explosion

# AI validation
USE_AI_VALIDATION = True            # Use AI to confirm market equivalence
AI_VALIDATION_MODEL = "claude"
AI_VALIDATION_MODEL_NAME = "claude-sonnet-4-5"

# Data paths
POLYMARKET_DATA = os.path.join(project_root, "src/data/polymarket")
KALSHI_DATA = os.path.join(project_root, "src/data/kalshi")
DATA_FOLDER = os.path.join(project_root, "src/data/arbitrage")
OPPORTUNITIES_CSV = os.path.join(DATA_FOLDER, "opportunities.csv")
MATCHED_MARKETS_CSV = os.path.join(DATA_FOLDER, "matched_markets.csv")
HISTORY_CSV = os.path.join(DATA_FOLDER, "history.csv")

# Common words to strip for better fuzzy matching
STRIP_WORDS = {
    'will', 'the', 'be', 'by', 'on', 'in', 'a', 'an', 'to', 'of',
    'before', 'after', 'during', 'this', 'that', 'is', 'are', 'was',
    'has', 'have', 'does', 'do', 'can', 'could', 'would', 'should',
}

# ==============================================================================
# Arbitrage Agent
# ==============================================================================

class ArbitrageAgent:
    """Cross-platform arbitrage detector between Polymarket and Kalshi"""

    def __init__(self):
        cprint("\n" + "="*80, "cyan")
        cprint("[*] Arbitrage Agent - Polymarket vs Kalshi", "cyan", attrs=['bold'])
        cprint("="*80, "cyan")

        os.makedirs(DATA_FOLDER, exist_ok=True)

        self.csv_lock = threading.Lock()

        # Matched markets cache (avoid re-validating with AI)
        self.matched_cache = self._load_matched_cache()

        # AI model for equivalence validation
        if USE_AI_VALIDATION:
            try:
                self.ai_model = ModelFactory().get_model(AI_VALIDATION_MODEL, AI_VALIDATION_MODEL_NAME)
                cprint(f"[AI] Loaded {AI_VALIDATION_MODEL}/{AI_VALIDATION_MODEL_NAME} for equivalence checks", "green")
            except Exception as e:
                cprint(f"[!] Failed to load AI model: {e}. Disabling AI validation.", "yellow")
                self.ai_model = None
        else:
            self.ai_model = None

        cprint("[+] Initialization complete!\n", "green")

    # ── Data Loading ──────────────────────────────────────────────────────────

    def _load_polymarket_markets(self) -> pd.DataFrame:
        """Load Polymarket markets from CSV."""
        csv_path = os.path.join(POLYMARKET_DATA, "markets.csv")
        if not os.path.exists(csv_path):
            cprint("[!] Polymarket markets.csv not found - run polymarket_agent.py first", "yellow")
            return pd.DataFrame()
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            cprint(f"[X] Error loading Polymarket markets: {e}", "red")
            return pd.DataFrame()

    def _load_kalshi_markets(self) -> pd.DataFrame:
        """Load Kalshi markets from CSV."""
        csv_path = os.path.join(KALSHI_DATA, "markets.csv")
        if not os.path.exists(csv_path):
            cprint("[!] Kalshi markets.csv not found - run kalshi_agent.py first", "yellow")
            return pd.DataFrame()
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            cprint(f"[X] Error loading Kalshi markets: {e}", "red")
            return pd.DataFrame()

    def _load_matched_cache(self) -> dict:
        """Load cached market pair validations to avoid re-querying AI."""
        if not os.path.exists(MATCHED_MARKETS_CSV):
            return {}
        try:
            df = pd.read_csv(MATCHED_MARKETS_CSV)
            cache = {}
            for _, row in df.iterrows():
                key = self._cache_key(row.get('polymarket_title', ''), row.get('kalshi_title', ''))
                cache[key] = {
                    'is_equivalent': row.get('is_equivalent', False),
                    'ai_equivalence_score': float(row.get('ai_equivalence_score', 0)),
                    'ai_reasoning': row.get('ai_reasoning', ''),
                }
            cprint(f"[+] Loaded {len(cache)} cached market pair validations", "green")
            return cache
        except Exception as e:
            cprint(f"[!] Error loading matched cache: {e}", "yellow")
            return {}

    def _cache_key(self, poly_title: str, kalshi_title: str) -> str:
        """Generate a stable cache key from two titles."""
        return f"{str(poly_title).strip().lower()}||{str(kalshi_title).strip().lower()}"

    def _save_matched_cache(self, pairs: list):
        """Save matched market pairs to CSV cache."""
        try:
            if os.path.exists(MATCHED_MARKETS_CSV):
                existing = pd.read_csv(MATCHED_MARKETS_CSV)
            else:
                existing = pd.DataFrame(columns=[
                    'timestamp', 'polymarket_title', 'kalshi_title',
                    'polymarket_slug', 'kalshi_ticker',
                    'fuzzy_match_score', 'ai_equivalence_score',
                    'ai_reasoning', 'is_equivalent'
                ])

            new_df = pd.concat([existing, pd.DataFrame(pairs)], ignore_index=True)
            with self.csv_lock:
                new_df.to_csv(MATCHED_MARKETS_CSV, index=False)
        except Exception as e:
            cprint(f"[X] Error saving matched cache: {e}", "red")

    def _save_opportunities(self, opportunities: list):
        """Save arbitrage opportunities to CSV (append)."""
        try:
            if os.path.exists(OPPORTUNITIES_CSV):
                existing = pd.read_csv(OPPORTUNITIES_CSV)
            else:
                existing = pd.DataFrame(columns=[
                    'timestamp', 'scan_id',
                    'polymarket_title', 'kalshi_title',
                    'polymarket_slug', 'kalshi_ticker',
                    'fuzzy_match_score', 'ai_equivalence_score',
                    'polymarket_yes_price', 'kalshi_yes_bid', 'kalshi_yes_ask',
                    'spread_cents', 'spread_direction',
                    'gross_profit_cents', 'estimated_fees_cents', 'net_profit_cents',
                    'risk_level', 'strategy',
                    'polymarket_link', 'kalshi_link'
                ])

            new_df = pd.concat([existing, pd.DataFrame(opportunities)], ignore_index=True)
            with self.csv_lock:
                new_df.to_csv(OPPORTUNITIES_CSV, index=False)
            cprint(f"[S] Saved {len(opportunities)} opportunities to CSV", "green")
        except Exception as e:
            cprint(f"[X] Error saving opportunities: {e}", "red")

    def _save_scan_history(self, scan_record: dict):
        """Save scan summary to history CSV."""
        try:
            if os.path.exists(HISTORY_CSV):
                existing = pd.read_csv(HISTORY_CSV)
            else:
                existing = pd.DataFrame(columns=[
                    'timestamp', 'scan_id',
                    'total_polymarket_markets', 'total_kalshi_markets',
                    'candidate_pairs_found', 'ai_validated_pairs',
                    'opportunities_above_threshold', 'best_spread_cents',
                    'scan_duration_seconds'
                ])

            new_df = pd.concat([existing, pd.DataFrame([scan_record])], ignore_index=True)
            with self.csv_lock:
                new_df.to_csv(HISTORY_CSV, index=False)
        except Exception as e:
            cprint(f"[X] Error saving scan history: {e}", "red")

    # ── Market Matching ───────────────────────────────────────────────────────

    def normalize_title(self, title: str) -> str:
        """Normalize a market title for fuzzy comparison."""
        if not title or not isinstance(title, str):
            return ""
        t = title.lower().strip()
        # Remove punctuation except hyphens
        t = re.sub(r'[^\w\s\-]', '', t)
        # Remove common filler words
        words = t.split()
        words = [w for w in words if w not in STRIP_WORDS]
        return ' '.join(words)

    def fuzzy_match_markets(self, poly_df: pd.DataFrame, kalshi_df: pd.DataFrame) -> list:
        """Find candidate market pairs using fuzzy title matching.

        Args:
            poly_df: Polymarket markets DataFrame
            kalshi_df: Kalshi markets DataFrame

        Returns:
            List of candidate pair dicts sorted by match score descending
        """
        # Cap the number of markets to prevent combinatorial explosion
        poly_subset = poly_df.tail(MAX_MARKETS_TO_COMPARE)
        kalshi_subset = kalshi_df.tail(MAX_MARKETS_TO_COMPARE)

        candidates = []

        for p_idx, p_row in poly_subset.iterrows():
            poly_title = str(p_row.get('title', ''))
            poly_norm = self.normalize_title(poly_title)
            if not poly_norm:
                continue

            for k_idx, k_row in kalshi_subset.iterrows():
                kalshi_title = str(k_row.get('title', ''))
                kalshi_norm = self.normalize_title(kalshi_title)
                if not kalshi_norm:
                    continue

                # Use token_sort_ratio for order-independent matching
                score = fuzz.token_sort_ratio(poly_norm, kalshi_norm)

                if score >= FUZZY_MATCH_THRESHOLD:
                    candidates.append({
                        'poly_idx': p_idx,
                        'kalshi_idx': k_idx,
                        'poly_title': poly_title,
                        'kalshi_title': kalshi_title,
                        'fuzzy_score': score,
                        'poly_row': p_row,
                        'kalshi_row': k_row,
                    })

        # Sort by score descending, deduplicate (keep best match per pair)
        candidates.sort(key=lambda x: x['fuzzy_score'], reverse=True)

        # Deduplicate: each polymarket market matches at most one kalshi market
        seen_poly = set()
        seen_kalshi = set()
        deduped = []
        for c in candidates:
            if c['poly_idx'] not in seen_poly and c['kalshi_idx'] not in seen_kalshi:
                deduped.append(c)
                seen_poly.add(c['poly_idx'])
                seen_kalshi.add(c['kalshi_idx'])

        return deduped

    def ai_validate_equivalence(self, poly_title: str, kalshi_title: str) -> tuple:
        """Use AI to determine if two markets are asking the same question.

        Returns:
            (is_equivalent: bool, confidence: float, reasoning: str)
        """
        if not self.ai_model:
            # No AI model available - rely on fuzzy score alone
            return (True, 0.5, "AI validation disabled")

        prompt = f"""You are evaluating whether two prediction markets from different platforms are asking the same question with the same resolution criteria.

Market A (Polymarket): {poly_title}
Market B (Kalshi): {kalshi_title}

Are these markets asking the SAME question and would they resolve the same way?

Consider:
- Are they about the same event/outcome?
- Do they have compatible resolution criteria?
- Could different date cutoffs or definitions cause different results?

Respond in EXACTLY this format:
VERDICT: EQUIVALENT or NOT_EQUIVALENT
CONFIDENCE: [0.0 to 1.0]
REASONING: [one sentence explanation]"""

        try:
            response = self.ai_model.generate_response(
                system_prompt="You compare prediction markets across platforms. Be precise about resolution criteria differences.",
                user_content=prompt,
                temperature=0.2,
                max_tokens=200
            )

            text = response.content.strip()

            # Parse verdict
            is_equivalent = 'EQUIVALENT' in text.upper() and 'NOT_EQUIVALENT' not in text.upper()

            # Parse confidence
            confidence = 0.5
            conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', text, re.IGNORECASE)
            if conf_match:
                confidence = float(conf_match.group(1))
                confidence = max(0.0, min(1.0, confidence))

            # Parse reasoning
            reasoning = ""
            reason_match = re.search(r'REASONING:\s*(.+)', text, re.IGNORECASE)
            if reason_match:
                reasoning = reason_match.group(1).strip()

            return (is_equivalent, confidence, reasoning)

        except Exception as e:
            cprint(f"[!] AI validation error: {e}", "yellow")
            return (False, 0.0, f"AI error: {str(e)}")

    # ── Price Analysis ────────────────────────────────────────────────────────

    def calculate_spread(self, poly_price: float, kalshi_yes_bid: float,
                         kalshi_yes_ask: float) -> list:
        """Calculate all possible arbitrage strategies between a matched pair.

        Args:
            poly_price: Polymarket YES price (0-1)
            kalshi_yes_bid: Kalshi best YES bid
            kalshi_yes_ask: Kalshi best YES ask

        Returns:
            List of viable strategy dicts with spread/profit info
        """
        strategies = []
        fee_rate = (POLYMARKET_FEE_PERCENT + KALSHI_FEE_PERCENT) / 100

        # Strategy 1: Buy YES on Kalshi, sell YES on Polymarket
        # Profitable if Kalshi ask < Polymarket price
        if kalshi_yes_ask > 0 and poly_price > kalshi_yes_ask:
            spread = poly_price - kalshi_yes_ask
            spread_cents = round(spread * 100, 2)
            gross = spread_cents
            fees = round(fee_rate * 100, 2)  # fees on a $1 contract
            net = round(gross - fees, 2)

            strategies.append({
                'spread_cents': spread_cents,
                'spread_direction': 'BUY_KALSHI_YES',
                'gross_profit_cents': gross,
                'estimated_fees_cents': fees,
                'net_profit_cents': net,
                'strategy': f"Buy YES on Kalshi @ ${kalshi_yes_ask:.2f}, implied sell on Polymarket @ ${poly_price:.2f}",
            })

        # Strategy 2: Buy YES on Polymarket, sell YES on Kalshi
        # Profitable if Polymarket price < Kalshi bid
        if kalshi_yes_bid > 0 and kalshi_yes_bid > poly_price:
            spread = kalshi_yes_bid - poly_price
            spread_cents = round(spread * 100, 2)
            gross = spread_cents
            fees = round(fee_rate * 100, 2)
            net = round(gross - fees, 2)

            strategies.append({
                'spread_cents': spread_cents,
                'spread_direction': 'BUY_POLY_YES',
                'gross_profit_cents': gross,
                'estimated_fees_cents': fees,
                'net_profit_cents': net,
                'strategy': f"Buy YES on Polymarket @ ${poly_price:.2f}, implied sell on Kalshi @ ${kalshi_yes_bid:.2f}",
            })

        # Strategy 3: Cross-platform hedge (guaranteed profit regardless of outcome)
        # Buy YES on Kalshi + Buy NO on Polymarket
        # Profit if: kalshi_yes_ask + (1 - poly_price) < 1.00
        poly_no_cost = 1.0 - poly_price
        total_cost_3 = kalshi_yes_ask + poly_no_cost
        if kalshi_yes_ask > 0 and total_cost_3 < 1.0:
            spread = 1.0 - total_cost_3
            spread_cents = round(spread * 100, 2)
            gross = spread_cents
            fees = round(fee_rate * 100, 2)
            net = round(gross - fees, 2)

            strategies.append({
                'spread_cents': spread_cents,
                'spread_direction': 'HEDGE_KALSHI_YES_POLY_NO',
                'gross_profit_cents': gross,
                'estimated_fees_cents': fees,
                'net_profit_cents': net,
                'strategy': f"Hedge: Buy YES Kalshi @ ${kalshi_yes_ask:.2f} + Buy NO Polymarket @ ${poly_no_cost:.2f} = ${total_cost_3:.2f} (guaranteed ${spread:.2f} profit)",
            })

        # Strategy 4: Reverse hedge
        # Buy YES on Polymarket + Buy NO on Kalshi
        # Profit if: poly_price + (1 - kalshi_yes_bid) < 1.00
        kalshi_no_cost = 1.0 - kalshi_yes_bid if kalshi_yes_bid > 0 else 1.0
        total_cost_4 = poly_price + kalshi_no_cost
        if kalshi_yes_bid > 0 and total_cost_4 < 1.0:
            spread = 1.0 - total_cost_4
            spread_cents = round(spread * 100, 2)
            gross = spread_cents
            fees = round(fee_rate * 100, 2)
            net = round(gross - fees, 2)

            strategies.append({
                'spread_cents': spread_cents,
                'spread_direction': 'HEDGE_POLY_YES_KALSHI_NO',
                'gross_profit_cents': gross,
                'estimated_fees_cents': fees,
                'net_profit_cents': net,
                'strategy': f"Hedge: Buy YES Polymarket @ ${poly_price:.2f} + Buy NO Kalshi @ ${kalshi_no_cost:.2f} = ${total_cost_4:.2f} (guaranteed ${spread:.2f} profit)",
            })

        return strategies

    def assess_risk(self, fuzzy_score: float, ai_confidence: float,
                    poly_volume: float, kalshi_volume: float) -> str:
        """Assess risk level of an arbitrage opportunity."""
        if fuzzy_score > 85 and ai_confidence > 0.9 and poly_volume > 50000 and kalshi_volume > 50000:
            return "LOW"
        elif fuzzy_score > 75 and ai_confidence > 0.8 and poly_volume > 10000 and kalshi_volume > 10000:
            return "MEDIUM"
        else:
            return "HIGH"

    # ── Main Scan ─────────────────────────────────────────────────────────────

    def run_scan(self) -> dict:
        """Run a full arbitrage scan cycle.

        Returns:
            Summary dict with scan results
        """
        scan_start = time.time()
        scan_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = datetime.now().isoformat()

        cprint("\n" + "="*80, "white", "on_magenta", attrs=['bold'])
        cprint(f"[ARB] ARBITRAGE SCAN - {scan_id}", "white", "on_magenta", attrs=['bold'])
        cprint("="*80, "white", "on_magenta", attrs=['bold'])

        # 1. Load markets from both platforms
        poly_df = self._load_polymarket_markets()
        kalshi_df = self._load_kalshi_markets()

        if poly_df.empty or kalshi_df.empty:
            cprint("\n[!] Need markets from BOTH platforms to detect arbitrage", "yellow", attrs=['bold'])
            if poly_df.empty:
                cprint("   [X] Polymarket: No data - run polymarket_agent.py", "red")
            else:
                cprint(f"   [+] Polymarket: {len(poly_df)} markets", "green")
            if kalshi_df.empty:
                cprint("   [X] Kalshi: No data - run kalshi_agent.py", "red")
            else:
                cprint(f"   [+] Kalshi: {len(kalshi_df)} markets", "green")
            return {'status': 'no_data'}

        cprint(f"\n[#] Polymarket markets: {len(poly_df)}", "cyan")
        cprint(f"[#] Kalshi markets: {len(kalshi_df)}", "cyan")

        # 2. Fuzzy match markets
        cprint(f"\n[~] Fuzzy matching titles (threshold: {FUZZY_MATCH_THRESHOLD})...", "cyan")
        candidates = self.fuzzy_match_markets(poly_df, kalshi_df)
        cprint(f"[+] Found {len(candidates)} candidate pairs", "green")

        if not candidates:
            cprint("\n[!] No matching markets found between platforms", "yellow")
            scan_duration = round(time.time() - scan_start, 1)
            self._save_scan_history({
                'timestamp': timestamp, 'scan_id': scan_id,
                'total_polymarket_markets': len(poly_df),
                'total_kalshi_markets': len(kalshi_df),
                'candidate_pairs_found': 0, 'ai_validated_pairs': 0,
                'opportunities_above_threshold': 0, 'best_spread_cents': 0,
                'scan_duration_seconds': scan_duration
            })
            return {'status': 'no_matches', 'candidates': 0}

        # 3. Validate equivalence (AI or cache)
        validated_pairs = []
        new_cache_entries = []
        ai_calls = 0

        cprint(f"\n[AI] Validating market equivalence...", "cyan")

        for candidate in candidates:
            poly_title = candidate['poly_title']
            kalshi_title = candidate['kalshi_title']
            cache_key = self._cache_key(poly_title, kalshi_title)

            # Check cache first
            if cache_key in self.matched_cache:
                cached = self.matched_cache[cache_key]
                if cached['is_equivalent']:
                    candidate['ai_equivalence_score'] = cached['ai_equivalence_score']
                    candidate['ai_reasoning'] = cached['ai_reasoning']
                    validated_pairs.append(candidate)
                    cprint(f"  [C] Cache hit: {poly_title[:40]}... <-> {kalshi_title[:40]}...", "white")
                continue

            # AI validation
            if USE_AI_VALIDATION and self.ai_model:
                is_eq, confidence, reasoning = self.ai_validate_equivalence(poly_title, kalshi_title)
                ai_calls += 1

                cache_entry = {
                    'timestamp': timestamp,
                    'polymarket_title': poly_title,
                    'kalshi_title': kalshi_title,
                    'polymarket_slug': candidate['poly_row'].get('event_slug', ''),
                    'kalshi_ticker': candidate['kalshi_row'].get('ticker', ''),
                    'fuzzy_match_score': candidate['fuzzy_score'],
                    'ai_equivalence_score': confidence,
                    'ai_reasoning': reasoning,
                    'is_equivalent': is_eq and confidence >= AI_EQUIVALENCE_THRESHOLD,
                }
                new_cache_entries.append(cache_entry)
                self.matched_cache[cache_key] = {
                    'is_equivalent': cache_entry['is_equivalent'],
                    'ai_equivalence_score': confidence,
                    'ai_reasoning': reasoning,
                }

                if is_eq and confidence >= AI_EQUIVALENCE_THRESHOLD:
                    candidate['ai_equivalence_score'] = confidence
                    candidate['ai_reasoning'] = reasoning
                    validated_pairs.append(candidate)
                    cprint(f"  [+] EQUIVALENT ({confidence:.0%}): {poly_title[:35]}... <-> {kalshi_title[:35]}...", "green")
                else:
                    cprint(f"  [X] Not equivalent ({confidence:.0%}): {poly_title[:35]}... <-> {kalshi_title[:35]}...", "red")
            else:
                # No AI - accept based on fuzzy score alone
                candidate['ai_equivalence_score'] = candidate['fuzzy_score'] / 100
                candidate['ai_reasoning'] = "Fuzzy match only (AI disabled)"
                validated_pairs.append(candidate)

        # Save new cache entries
        if new_cache_entries:
            self._save_matched_cache(new_cache_entries)
            cprint(f"[S] Cached {len(new_cache_entries)} new pair validations (used {ai_calls} AI calls)", "cyan")

        cprint(f"\n[+] {len(validated_pairs)} validated equivalent pairs", "green", attrs=['bold'])

        if not validated_pairs:
            scan_duration = round(time.time() - scan_start, 1)
            self._save_scan_history({
                'timestamp': timestamp, 'scan_id': scan_id,
                'total_polymarket_markets': len(poly_df),
                'total_kalshi_markets': len(kalshi_df),
                'candidate_pairs_found': len(candidates),
                'ai_validated_pairs': 0,
                'opportunities_above_threshold': 0, 'best_spread_cents': 0,
                'scan_duration_seconds': scan_duration
            })
            return {'status': 'no_validated', 'candidates': len(candidates)}

        # 4. Calculate spreads and find opportunities
        cprint(f"\n[~] Calculating price spreads...", "cyan")

        all_opportunities = []
        best_spread = 0

        for pair in validated_pairs:
            poly_row = pair['poly_row']
            kalshi_row = pair['kalshi_row']

            poly_price = float(poly_row.get('price', 0))
            kalshi_yes_bid = float(kalshi_row.get('yes_bid', 0))
            kalshi_yes_ask = float(kalshi_row.get('yes_ask', 0))

            if poly_price <= 0 or (kalshi_yes_bid <= 0 and kalshi_yes_ask <= 0):
                continue

            strategies = self.calculate_spread(poly_price, kalshi_yes_bid, kalshi_yes_ask)

            for strat in strategies:
                if strat['spread_cents'] < SPREAD_THRESHOLD_CENTS:
                    continue

                poly_volume = float(poly_row.get('size_usd', 0))
                kalshi_volume = float(kalshi_row.get('volume_24h', 0))
                risk = self.assess_risk(
                    pair['fuzzy_score'],
                    pair.get('ai_equivalence_score', 0.5),
                    poly_volume,
                    kalshi_volume
                )

                poly_slug = poly_row.get('event_slug', '')
                kalshi_ticker = kalshi_row.get('ticker', '')

                opportunity = {
                    'timestamp': timestamp,
                    'scan_id': scan_id,
                    'polymarket_title': pair['poly_title'],
                    'kalshi_title': pair['kalshi_title'],
                    'polymarket_slug': poly_slug,
                    'kalshi_ticker': kalshi_ticker,
                    'fuzzy_match_score': pair['fuzzy_score'],
                    'ai_equivalence_score': pair.get('ai_equivalence_score', 0),
                    'polymarket_yes_price': poly_price,
                    'kalshi_yes_bid': kalshi_yes_bid,
                    'kalshi_yes_ask': kalshi_yes_ask,
                    'spread_cents': strat['spread_cents'],
                    'spread_direction': strat['spread_direction'],
                    'gross_profit_cents': strat['gross_profit_cents'],
                    'estimated_fees_cents': strat['estimated_fees_cents'],
                    'net_profit_cents': strat['net_profit_cents'],
                    'risk_level': risk,
                    'strategy': strat['strategy'],
                    'polymarket_link': f"https://polymarket.com/event/{poly_slug}",
                    'kalshi_link': f"https://kalshi.com/markets/{kalshi_ticker}",
                }
                all_opportunities.append(opportunity)

                if strat['spread_cents'] > best_spread:
                    best_spread = strat['spread_cents']

        # 5. Display and save results
        if all_opportunities:
            # Sort by net profit descending
            all_opportunities.sort(key=lambda x: x['net_profit_cents'], reverse=True)
            self._save_opportunities(all_opportunities)
            self.display_opportunities(all_opportunities)
        else:
            cprint("\n[!] No opportunities above threshold found", "yellow")

        # 6. Save scan history
        scan_duration = round(time.time() - scan_start, 1)
        self._save_scan_history({
            'timestamp': timestamp,
            'scan_id': scan_id,
            'total_polymarket_markets': len(poly_df),
            'total_kalshi_markets': len(kalshi_df),
            'candidate_pairs_found': len(candidates),
            'ai_validated_pairs': len(validated_pairs),
            'opportunities_above_threshold': len(all_opportunities),
            'best_spread_cents': best_spread,
            'scan_duration_seconds': scan_duration,
        })

        cprint(f"\n[T] Scan completed in {scan_duration}s", "cyan")

        return {
            'status': 'complete',
            'candidates': len(candidates),
            'validated': len(validated_pairs),
            'opportunities': len(all_opportunities),
            'best_spread': best_spread,
        }

    def display_opportunities(self, opportunities: list):
        """Pretty-print arbitrage opportunities."""
        cprint("\n" + "="*80, "white", "on_green", attrs=['bold'])
        cprint(f"[ARB] ARBITRAGE OPPORTUNITIES FOUND: {len(opportunities)}", "white", "on_green", attrs=['bold'])
        cprint("="*80, "white", "on_green", attrs=['bold'])

        for i, opp in enumerate(opportunities):
            spread = opp['spread_cents']
            net = opp['net_profit_cents']

            # Color based on profitability
            if net > 0:
                color = "green"
                tag = "[+]"
            elif net > -1:
                color = "yellow"
                tag = "[~]"
            else:
                color = "red"
                tag = "[-]"

            risk_color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}.get(opp['risk_level'], "white")

            cprint(f"\n{'─'*80}", "cyan")
            cprint(f"{tag} Opportunity #{i+1}", color, attrs=['bold'])
            cprint(f"  Polymarket: {opp['polymarket_title'][:60]}", "white")
            cprint(f"  Kalshi:     {opp['kalshi_title'][:60]}", "white")
            cprint(f"  Match: {opp['fuzzy_match_score']}% fuzzy | {opp['ai_equivalence_score']:.0%} AI confidence", "cyan")
            cprint(f"  Spread: {spread:.1f}c gross → {net:.1f}c net (after {opp['estimated_fees_cents']:.1f}c fees)", color, attrs=['bold'])
            cprint(f"  Direction: {opp['spread_direction']}", "white")
            cprint(f"  Risk: {opp['risk_level']}", risk_color, attrs=['bold'])
            cprint(f"  Strategy: {opp['strategy']}", "white")
            cprint(f"  Polymarket: {opp['polymarket_link']}", "cyan")
            cprint(f"  Kalshi:     {opp['kalshi_link']}", "cyan")

        cprint(f"\n{'='*80}", "green")
        profitable = [o for o in opportunities if o['net_profit_cents'] > 0]
        cprint(f"[+] {len(profitable)}/{len(opportunities)} opportunities are net profitable after fees", "green", attrs=['bold'])
        cprint("="*80 + "\n", "green")

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def scan_loop(self):
        """Continuous arbitrage scanning loop."""
        cprint("\n[ARB] ARBITRAGE SCAN LOOP STARTED", "magenta", attrs=['bold'])
        cprint(f"[T] Scanning every {SCAN_INTERVAL_SECONDS}s\n", "magenta")

        while True:
            try:
                self.run_scan()

                next_scan = datetime.now() + timedelta(seconds=SCAN_INTERVAL_SECONDS)
                cprint(f"[T] Next arbitrage scan at: {next_scan.strftime('%H:%M:%S')}\n", "magenta")

                time.sleep(SCAN_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                break
            except Exception as e:
                cprint(f"[X] Error in scan loop: {e}", "red")
                import traceback
                traceback.print_exc()
                time.sleep(SCAN_INTERVAL_SECONDS)


def main():
    """Arbitrage Agent Main"""
    cprint("\n" + "="*80, "cyan")
    cprint("[*] Arbitrage Agent - Polymarket vs Kalshi", "cyan", attrs=['bold'])
    cprint("="*80, "cyan")
    cprint(f"[~] Fuzzy match threshold: {FUZZY_MATCH_THRESHOLD}%", "yellow")
    cprint(f"[~] Spread threshold: {SPREAD_THRESHOLD_CENTS}c", "yellow")
    cprint(f"[~] AI validation: {'ENABLED' if USE_AI_VALIDATION else 'DISABLED'}", "green" if USE_AI_VALIDATION else "yellow")
    cprint(f"[$] Fees: Polymarket {POLYMARKET_FEE_PERCENT}% + Kalshi {KALSHI_FEE_PERCENT}%", "yellow")
    cprint(f"[T] Scan interval: {SCAN_INTERVAL_SECONDS}s", "yellow")
    cprint("", "yellow")
    cprint("[F] Data Sources:", "cyan", attrs=['bold'])
    cprint(f"   Polymarket: {os.path.join(POLYMARKET_DATA, 'markets.csv')}", "white")
    cprint(f"   Kalshi:     {os.path.join(KALSHI_DATA, 'markets.csv')}", "white")
    cprint("[F] Output:", "cyan", attrs=['bold'])
    cprint(f"   Opportunities: {OPPORTUNITIES_CSV}", "white")
    cprint(f"   Matched pairs: {MATCHED_MARKETS_CSV}", "white")
    cprint(f"   Scan history:  {HISTORY_CSV}", "white")
    cprint("="*80 + "\n", "cyan")

    agent = ArbitrageAgent()

    # Run initial scan
    cprint("[>] Running initial arbitrage scan...\n", "green", attrs=['bold'])
    agent.run_scan()

    # Continuous scanning
    try:
        agent.scan_loop()
    except KeyboardInterrupt:
        cprint("\n\n" + "="*80, "yellow")
        cprint("[!] Arbitrage Agent stopped by user", "yellow", attrs=['bold'])
        cprint("="*80 + "\n", "yellow")
        sys.exit(0)


if __name__ == "__main__":
    main()
