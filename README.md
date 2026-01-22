# Polymarket Agent

AI-powered Polymarket prediction market analyzer using multi-model swarm intelligence.

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

### 3. Run the Agent

```bash
conda activate polymarket
python src/agents/polymarket_agent.py
```

## What It Does

1. **Connects to Polymarket WebSocket** for real-time trade data
2. **Filters trades** (>$500, excludes crypto/sports, excludes near-resolved markets)
3. **Runs AI swarm analysis** when new markets are collected
4. **Generates consensus picks** across multiple AI models
5. **Saves results** to CSV files

## Output Files

Located in `src/data/polymarket/`:

| File | Description |
|------|-------------|
| `markets.csv` | All tracked markets with trade data |
| `predictions.csv` | AI predictions per market |
| `consensus_picks.csv` | Top consensus picks (best opportunities) |

## Configuration

Edit settings at the top of `src/agents/polymarket_agent.py`:

```python
MIN_TRADE_SIZE_USD = 500      # Minimum trade size to track
LOOKBACK_HOURS = 24           # Historical data on startup
ANALYSIS_CHECK_INTERVAL_SECONDS = 300  # Check every 5 min
NEW_MARKETS_FOR_ANALYSIS = 3  # Trigger AI when 3 new markets
USE_SWARM_MODE = True         # Use multiple AI models
```

## Project Structure

```
PolymarketAgent/
├── src/
│   ├── agents/
│   │   ├── polymarket_agent.py  # Main agent
│   │   └── swarm_agent.py       # Multi-model AI swarm
│   ├── models/                  # LLM provider implementations
│   │   ├── model_factory.py     # Unified model interface
│   │   ├── claude_model.py
│   │   ├── openai_model.py
│   │   ├── deepseek_model.py
│   │   └── ...
│   ├── data/
│   │   └── polymarket/          # Output CSV files
│   └── config.py                # Global configuration
├── requirements.txt
├── .env.example
└── README.md
```

## Stop the Agent

Press `Ctrl+C` to stop gracefully.
