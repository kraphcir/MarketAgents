# Prediction Market Agent Suite

AI-powered prediction market analyzer using multi-model swarm intelligence. Supports Polymarket, Kalshi, and cross-platform arbitrage detection.

## Setup

### 1. Create Environment

```bash
# Create conda environment
conda create -n polymarket python=3.11 -y
conda activate polymarket

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys (at least one required)
```

**Required API Keys (at least one):**
- `ANTHROPIC_KEY` - Claude API
- `OPENAI_KEY` - OpenAI GPT
- `DEEPSEEK_KEY` - DeepSeek
- `GROQ_API_KEY` - Groq
- `GEMINI_KEY` - Google Gemini
- `GROK_API_KEY` - xAI Grok
- `OPENROUTER_API_KEY` - OpenRouter (200+ models)

### 3. Run the Agents

**Run everything (recommended):**
```bash
python run_all.py
```

**Run with fresh data (clears old CSVs):**
```bash
python run_all.py --fresh
```

**Run individual components:**
```bash
python run_all.py --no-kalshi --no-arb --no-crypto    # Polymarket only
python run_all.py --no-poly --no-arb --no-crypto      # Kalshi only
python run_all.py --no-poly --no-kalshi --no-crypto   # Arbitrage only
python run_all.py --no-poly --no-kalshi --no-arb      # Crypto Hedge only
```

**Run agents directly:**
```bash
python src/agents/polymarket_agent.py     # Polymarket agent
python src/agents/kalshi_agent.py         # Kalshi agent
python src/agents/arbitrage_agent.py      # Arbitrage detector
python src/agents/crypto_hedge_agent.py   # Crypto Hedge agent
python dashboard/run.py                   # Dashboard only
```

## Components

### Polymarket Agent

Connects to Polymarket via WebSocket for real-time trade monitoring.

**How it works:**
1. Connects to Polymarket WebSocket for real-time trade data
2. Filters trades (>$500, excludes crypto/sports, excludes near-resolved markets)
3. Runs AI swarm analysis when new markets are collected
4. Generates consensus picks across multiple AI models
5. Saves results to CSV files

**Output:** `src/data/polymarket/`

### Kalshi Agent

Polls Kalshi's REST API for market data (no authentication required for reads).

**How it works:**
1. Polls Kalshi API every 60 seconds for active markets
2. Filters by volume ($10k+ 24h volume), open interest ($5k+), excludes crypto/sports
3. Runs AI swarm analysis every 5 minutes when 3+ new markets collected
4. Generates consensus picks with bid/ask spread analysis
5. Saves results to CSV files

**Output:** `src/data/kalshi/`

### Arbitrage Agent

Detects cross-platform arbitrage opportunities between Polymarket and Kalshi.

**How it works:**
1. Loads market data from both platforms' CSV files
2. Fuzzy-matches market titles using string similarity (threshold: 65%)
3. AI validates matched pairs to confirm identical resolution criteria
4. Calculates price spreads and identifies profitable opportunities
5. Accounts for platform fees (Polymarket ~2%, Kalshi ~1%)

**4 Arbitrage Strategies:**

| Strategy | Condition | Action |
|----------|-----------|--------|
| Direct YES Spread | Kalshi YES ask < Polymarket price | Buy YES on Kalshi |
| Direct NO Spread | Kalshi NO ask < (1 - Polymarket price) | Buy NO on Kalshi |
| Cross-Platform Hedge | Kalshi YES + Poly NO < $1.00 | Buy both for guaranteed profit |
| Reverse Hedge | Poly YES + Kalshi NO < $1.00 | Buy both for guaranteed profit |

**Risk Levels:**
- **LOW**: Fuzzy match >85%, AI confidence >90%, high volume on both platforms
- **MEDIUM**: Fuzzy match >75%, AI confidence >80%
- **HIGH**: Everything else above threshold

**Output:** `src/data/arbitrage/`

### Crypto Hedge Agent

Live ETH/BTC price tracking with Kalshi crypto market hedge detection.

**How it works:**
1. Fetches live ETH/BTC prices from CoinGecko every 30 seconds (free, no API key)
2. Polls Kalshi for all crypto-related prediction markets every 60 seconds
3. Extracts price targets from market titles (e.g., "Bitcoin above $100k")
4. Compares current spot prices against market targets
5. Identifies hedge opportunities where you can profit from price predictions

**Hedge Types:**
- **Bullish Hedge**: Price already above target - buy YES to lock in profit
- **Bearish Hedge**: Price below target - buy NO if you expect it to stay below

**Risk Assessment:**
- **LOW**: Price >20% from target, >7 days to expiry
- **MEDIUM**: Price >10% from target, >3 days to expiry
- **HIGH**: Everything else (close to target or near expiry)

**Output:** `src/data/crypto_hedge/`

### Dashboard

Web-based dashboard to view all agent data.

**Access:** http://localhost:8000

**Features:**
- Real-time stats for all platforms
- Polymarket consensus picks and markets
- Kalshi consensus picks and markets
- Arbitrage opportunities with spread calculations
- Auto-refresh every 30 seconds

## Output Files

### Polymarket (`src/data/polymarket/`)

| File | Description |
|------|-------------|
| `markets.csv` | All tracked markets with trade data |
| `predictions.csv` | AI predictions per market |
| `consensus_picks.csv` | Top consensus picks |

### Kalshi (`src/data/kalshi/`)

| File | Description |
|------|-------------|
| `markets.csv` | All tracked markets with bid/ask prices |
| `predictions.csv` | AI predictions per market |
| `consensus_picks.csv` | Top consensus picks |

### Arbitrage (`src/data/arbitrage/`)

| File | Description |
|------|-------------|
| `opportunities.csv` | Detected arbitrage opportunities |
| `matched_markets.csv` | AI-validated market pairs cache |
| `history.csv` | Scan history and statistics |

### Crypto Hedge (`src/data/crypto_hedge/`)

| File | Description |
|------|-------------|
| `prices.csv` | Current ETH/BTC prices |
| `kalshi_crypto_markets.csv` | Kalshi crypto prediction markets |
| `opportunities.csv` | Hedge opportunities with profit estimates |
| `history.csv` | Scan history and statistics |

## Configuration

### Polymarket Agent

Edit `src/agents/polymarket_agent.py`:

```python
MIN_TRADE_SIZE_USD = 500      # Minimum trade size to track
LOOKBACK_HOURS = 24           # Historical data on startup
ANALYSIS_CHECK_INTERVAL_SECONDS = 300  # Check every 5 min
NEW_MARKETS_FOR_ANALYSIS = 3  # Trigger AI when 3 new markets
USE_SWARM_MODE = True         # Use multiple AI models
```

### Kalshi Agent

Edit `src/agents/kalshi_agent.py`:

```python
POLL_INTERVAL_SECONDS = 60    # Fetch markets every 60s
MIN_VOLUME_24H = 10000        # Minimum 24h volume
MIN_OPEN_INTEREST = 5000      # Minimum open interest
ANALYSIS_CHECK_INTERVAL_SECONDS = 300  # Check every 5 min
NEW_MARKETS_FOR_ANALYSIS = 3  # Trigger AI when 3 new markets
```

### Arbitrage Agent

Edit `src/agents/arbitrage_agent.py`:

```python
SCAN_INTERVAL_SECONDS = 600   # Scan every 10 minutes
FUZZY_MATCH_THRESHOLD = 65    # Minimum string similarity %
MIN_SPREAD_THRESHOLD = 0.05   # Minimum 5 cent spread
POLYMARKET_FEE = 0.02         # 2% fee estimate
KALSHI_FEE = 0.01             # 1% fee estimate
```

### Crypto Hedge Agent

Edit `src/agents/crypto_hedge_agent.py`:

```python
PRICE_UPDATE_INTERVAL = 30    # Fetch prices every 30 seconds (CoinGecko limit)
KALSHI_POLL_INTERVAL = 60     # Fetch Kalshi markets every 60 seconds
HEDGE_SCAN_INTERVAL = 30      # Scan for opportunities every 30 seconds
MIN_HEDGE_PROFIT_PCT = 1.0    # Minimum profit % to flag
KALSHI_FEE = 0.01             # 1% Kalshi fee
```

## API Cost Estimates

The agents use AI models for market analysis. Costs depend on which models are enabled.

### Per Analysis Run

| Model | Input Tokens | Output Tokens | Est. Cost |
|-------|-------------|---------------|-----------|
| Claude Sonnet | ~4,000 | ~2,000 | ~$0.04 |
| GPT-4o | ~4,000 | ~2,000 | ~$0.03 |
| DeepSeek | ~4,000 | ~2,000 | ~$0.002 |
| Groq (Llama) | ~4,000 | ~2,000 | ~$0.003 |

### Estimated Total Costs (all agents running)

| Timeframe | Analysis Runs | Est. Cost (all models) |
|-----------|--------------|------------------------|
| Per hour | 2-6 | $0.10-0.30 |
| Per day (8 hrs) | 16-48 | $0.80-2.40 |
| Per day (24 hrs) | 48-144 | $2.40-7.00 |

*Costs vary based on market activity and enabled models.*

### Reduce Costs

1. **Use fewer models** - Edit `src/agents/swarm_agent.py`:
   ```python
   SWARM_MODELS = {
       "claude": (False, ...),  # Disable expensive models
       "deepseek": (True, ...),  # Keep cheap models
   }
   ```

2. **Run analysis less often** - Increase `ANALYSIS_CHECK_INTERVAL_SECONDS`

3. **Run fewer agents** - Use `--no-kalshi` or `--no-arb` flags

**Note:** The Crypto Hedge Agent uses free APIs (CoinGecko, Kalshi public) and doesn't consume AI tokens.

## Project Structure

```
PolymarketAgent/
├── src/
│   ├── agents/
│   │   ├── polymarket_agent.py    # Polymarket WebSocket agent
│   │   ├── kalshi_agent.py        # Kalshi REST polling agent
│   │   ├── arbitrage_agent.py     # Cross-platform arbitrage detector
│   │   ├── crypto_hedge_agent.py  # ETH/BTC price + Kalshi hedge detector
│   │   └── swarm_agent.py         # Multi-model AI swarm
│   ├── models/                    # LLM provider implementations
│   │   ├── model_factory.py       # Unified model interface
│   │   ├── claude_model.py
│   │   ├── openai_model.py
│   │   ├── deepseek_model.py
│   │   ├── gemini_model.py
│   │   ├── groq_model.py
│   │   └── xai_model.py
│   ├── data/
│   │   ├── polymarket/            # Polymarket output CSVs
│   │   ├── kalshi/                # Kalshi output CSVs
│   │   ├── arbitrage/             # Arbitrage output CSVs
│   │   └── crypto_hedge/          # Crypto hedge output CSVs
│   └── config.py                  # Global configuration
├── dashboard/
│   ├── backend/
│   │   ├── main.py                # FastAPI app
│   │   ├── routes.py              # API endpoints
│   │   ├── csv_reader.py          # Polymarket data reader
│   │   └── kalshi_csv_reader.py   # Kalshi + arbitrage + crypto data reader
│   ├── frontend/
│   │   ├── index.html
│   │   ├── app.js
│   │   └── styles.css
│   └── run.py                     # Dashboard launcher
├── run_all.py                     # Run all components
├── requirements.txt
├── .env.example
└── README.md
```

## Stop the Agents

Press `Ctrl+C` to stop all running components gracefully.
