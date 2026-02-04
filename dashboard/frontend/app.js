const { useState, useEffect } = React;

// API fetch helper
const fetchAPI = async (endpoint) => {
    const response = await fetch(`/api${endpoint}`);
    if (!response.ok) throw new Error('API error');
    return response.json();
};

// Stats Bar Component
function StatsBar({ stats }) {
    return (
        <div className="stats-bar">
            <div className="stat">
                <div className="stat-value">{stats.total_markets}</div>
                <div className="stat-label">Poly Markets</div>
            </div>
            <div className="stat">
                <div className="stat-value">{stats.kalshi_markets || 0}</div>
                <div className="stat-label">Kalshi Markets</div>
            </div>
            <div className="stat">
                <div className="stat-value">{stats.total_consensus_picks}</div>
                <div className="stat-label">Poly Picks</div>
            </div>
            <div className="stat">
                <div className="stat-value">{stats.kalshi_consensus_picks || 0}</div>
                <div className="stat-label">Kalshi Picks</div>
            </div>
            <div className="stat">
                <div className="stat-value">{stats.arbitrage_profitable || 0}</div>
                <div className="stat-label">Arb Opps</div>
            </div>
        </div>
    );
}

// Prediction badge helper (shared)
const getBadgeClass = (pred) => {
    if (!pred || pred === 'N/A') return '';
    const p = String(pred).toUpperCase();
    if (p.includes('YES')) return 'badge badge-yes';
    if (p.includes('NO_TRADE') || p.includes('NO TRADE')) return 'badge badge-notrade';
    if (p.includes('NO')) return 'badge badge-no';
    return '';
};

const getSideClass = (side) => {
    const s = String(side).toUpperCase();
    if (s === 'YES') return 'consensus-yes';
    if (s === 'NO') return 'consensus-no';
    return 'consensus-notrade';
};

// ── Polymarket Components ────────────────────────────────────────────────────

function MarketsTable({ markets }) {
    if (!markets.length) {
        return (
            <div className="empty-state">
                <h3>No Markets Yet</h3>
                <p>Run the Polymarket agent to start tracking markets</p>
            </div>
        );
    }

    return (
        <table>
            <thead>
                <tr>
                    <th>Title</th>
                    <th>Outcome</th>
                    <th>Price</th>
                    <th>Size (USD)</th>
                    <th>First Seen</th>
                </tr>
            </thead>
            <tbody>
                {markets.map((m, i) => (
                    <tr key={i}>
                        <td>
                            <a
                                href={`https://polymarket.com/event/${m.event_slug}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="link"
                            >
                                {m.title?.substring(0, 60)}{m.title?.length > 60 ? '...' : ''}
                            </a>
                        </td>
                        <td>{m.outcome}</td>
                        <td className="price">${parseFloat(m.price || 0).toFixed(2)}</td>
                        <td>${parseFloat(m.size_usd || 0).toLocaleString()}</td>
                        <td>{m.first_seen?.split('T')[0]}</td>
                    </tr>
                ))}
            </tbody>
        </table>
    );
}

function PredictionsTable({ predictions }) {
    if (!predictions.length) {
        return (
            <div className="empty-state">
                <h3>No Predictions Yet</h3>
                <p>AI analysis will appear here after the agent runs</p>
            </div>
        );
    }

    return (
        <table>
            <thead>
                <tr>
                    <th>Market</th>
                    <th>Consensus</th>
                    <th>Claude</th>
                    <th>DeepSeek</th>
                    <th>XAI</th>
                    <th>Models</th>
                    <th>Time</th>
                </tr>
            </thead>
            <tbody>
                {predictions.map((p, i) => (
                    <tr key={i}>
                        <td>
                            <a href={p.market_link} target="_blank" rel="noopener noreferrer" className="link">
                                {p.market_title?.substring(0, 40)}{p.market_title?.length > 40 ? '...' : ''}
                            </a>
                        </td>
                        <td><span className={getBadgeClass(p.consensus_prediction)}>{p.consensus_prediction}</span></td>
                        <td><span className={getBadgeClass(p.claude_prediction)}>{p.claude_prediction || '-'}</span></td>
                        <td><span className={getBadgeClass(p.deepseek_prediction)}>{p.deepseek_prediction || '-'}</span></td>
                        <td><span className={getBadgeClass(p.xai_prediction)}>{p.xai_prediction || '-'}</span></td>
                        <td>{p.num_models_responded}</td>
                        <td>{p.analysis_timestamp?.split('T')[1]?.split('.')[0]}</td>
                    </tr>
                ))}
            </tbody>
        </table>
    );
}

function ConsensusPicks({ picks }) {
    if (!picks.length) {
        return (
            <div className="empty-state">
                <h3>No Consensus Picks Yet</h3>
                <p>Top AI recommendations will appear here after analysis runs</p>
            </div>
        );
    }

    return (
        <div>
            {picks.map((pick, i) => (
                <div key={i} className="consensus-card">
                    <h3>
                        #{pick.rank} -{' '}
                        <a href={pick.link} target="_blank" rel="noopener noreferrer" className="link">
                            {pick.market_title}
                        </a>
                    </h3>
                    <div className="consensus-meta">
                        <span className={getSideClass(pick.side)}>
                            Side: <strong>{pick.side}</strong>
                        </span>
                        {' | '}
                        <span>{pick.consensus}</span>
                        {' | '}
                        <span>Run: {pick.run_id}</span>
                    </div>
                    {pick.reasoning && <p className="reasoning">{pick.reasoning}</p>}
                </div>
            ))}
        </div>
    );
}

// ── Kalshi Components ────────────────────────────────────────────────────────

function KalshiMarketsTable({ markets }) {
    if (!markets.length) {
        return (
            <div className="empty-state">
                <h3>No Kalshi Markets Yet</h3>
                <p>Run the Kalshi agent to start tracking markets</p>
            </div>
        );
    }

    return (
        <table>
            <thead>
                <tr>
                    <th>Title</th>
                    <th>YES Bid</th>
                    <th>YES Ask</th>
                    <th>Last Price</th>
                    <th>Volume 24h</th>
                    <th>Open Interest</th>
                    <th>First Seen</th>
                </tr>
            </thead>
            <tbody>
                {markets.map((m, i) => (
                    <tr key={i}>
                        <td>
                            <a
                                href={`https://kalshi.com/markets/${m.ticker}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="link"
                            >
                                {String(m.title || m.ticker).substring(0, 55)}
                                {String(m.title || '').length > 55 ? '...' : ''}
                            </a>
                        </td>
                        <td className="bid-ask">
                            <span className="bid">${parseFloat(m.yes_bid || 0).toFixed(2)}</span>
                        </td>
                        <td className="bid-ask">
                            <span className="ask">${parseFloat(m.yes_ask || 0).toFixed(2)}</span>
                        </td>
                        <td className="price">${parseFloat(m.last_price || 0).toFixed(2)}</td>
                        <td>${parseFloat(m.volume_24h || 0).toLocaleString()}</td>
                        <td>{parseFloat(m.open_interest || 0).toLocaleString()}</td>
                        <td>{m.first_seen?.split('T')[0]}</td>
                    </tr>
                ))}
            </tbody>
        </table>
    );
}

function KalshiConsensusPicks({ picks }) {
    if (!picks.length) {
        return (
            <div className="empty-state">
                <h3>No Kalshi Consensus Picks Yet</h3>
                <p>Run the Kalshi agent and wait for AI analysis</p>
            </div>
        );
    }

    return (
        <div>
            {picks.map((pick, i) => (
                <div key={i} className="consensus-card">
                    <h3>
                        #{pick.rank} -{' '}
                        <a href={pick.link} target="_blank" rel="noopener noreferrer" className="link">
                            {pick.market_title}
                        </a>
                    </h3>
                    <div className="consensus-meta">
                        <span className={getSideClass(pick.side)}>
                            Side: <strong>{pick.side}</strong>
                        </span>
                        {' | '}
                        <span>{pick.consensus}</span>
                        {' | '}
                        <span>Run: {pick.run_id}</span>
                    </div>
                    {pick.reasoning && <p className="reasoning">{pick.reasoning}</p>}
                </div>
            ))}
        </div>
    );
}

// ── Arbitrage Components ─────────────────────────────────────────────────────

function ArbitrageTable({ opportunities }) {
    if (!opportunities.length) {
        return (
            <div className="empty-state">
                <h3>No Arbitrage Opportunities Yet</h3>
                <p>Run both agents and the arbitrage scanner to detect cross-platform spreads</p>
            </div>
        );
    }

    const getRiskClass = (risk) => {
        const r = String(risk).toUpperCase();
        if (r === 'LOW') return 'risk-low';
        if (r === 'MEDIUM') return 'risk-medium';
        return 'risk-high';
    };

    const getProfitClass = (net) => {
        const n = parseFloat(net || 0);
        if (n > 0) return 'arb-positive';
        if (n > -1) return 'arb-marginal';
        return 'arb-negative';
    };

    return (
        <table>
            <thead>
                <tr>
                    <th>Polymarket</th>
                    <th>Kalshi</th>
                    <th>Match</th>
                    <th>Spread</th>
                    <th>Net Profit</th>
                    <th>Risk</th>
                    <th>Direction</th>
                    <th>Links</th>
                </tr>
            </thead>
            <tbody>
                {opportunities.map((o, i) => (
                    <tr key={i}>
                        <td>{String(o.polymarket_title || '').substring(0, 35)}
                            {String(o.polymarket_title || '').length > 35 ? '...' : ''}</td>
                        <td>{String(o.kalshi_title || '').substring(0, 35)}
                            {String(o.kalshi_title || '').length > 35 ? '...' : ''}</td>
                        <td>{o.fuzzy_match_score}%</td>
                        <td>
                            <span className={`spread-badge ${getProfitClass(o.spread_cents)}`}>
                                {parseFloat(o.spread_cents || 0).toFixed(1)}c
                            </span>
                        </td>
                        <td className={getProfitClass(o.net_profit_cents)}>
                            {parseFloat(o.net_profit_cents || 0).toFixed(1)}c
                        </td>
                        <td>
                            <span className={`badge ${getRiskClass(o.risk_level)}`}>
                                {o.risk_level}
                            </span>
                        </td>
                        <td style={{fontSize: '0.8rem'}}>{o.spread_direction}</td>
                        <td>
                            <div className="platform-links">
                                <a href={o.polymarket_link} target="_blank" rel="noopener noreferrer">Poly</a>
                                <a href={o.kalshi_link} target="_blank" rel="noopener noreferrer">Kalshi</a>
                            </div>
                        </td>
                    </tr>
                ))}
            </tbody>
        </table>
    );
}

// ── Main Dashboard ───────────────────────────────────────────────────────────

function Dashboard() {
    const [activeTab, setActiveTab] = useState('consensus');
    const [stats, setStats] = useState({
        total_markets: 0, total_predictions: 0, total_consensus_picks: 0,
        kalshi_markets: 0, kalshi_predictions: 0, kalshi_consensus_picks: 0,
        arbitrage_opportunities: 0, arbitrage_profitable: 0, arbitrage_best_spread_cents: 0,
    });
    const [markets, setMarkets] = useState([]);
    const [predictions, setPredictions] = useState([]);
    const [consensus, setConsensus] = useState([]);
    const [kalshiMarkets, setKalshiMarkets] = useState([]);
    const [kalshiConsensus, setKalshiConsensus] = useState([]);
    const [arbitrageOpps, setArbitrageOpps] = useState([]);
    const [loading, setLoading] = useState(true);
    const [lastRefresh, setLastRefresh] = useState(null);

    const refreshData = async () => {
        try {
            const [statsData, marketsData, predictionsData, consensusData,
                   kalshiMarketsData, kalshiConsensusData, arbData] = await Promise.all([
                fetchAPI('/stats'),
                fetchAPI('/markets?limit=100'),
                fetchAPI('/predictions?limit=50'),
                fetchAPI('/consensus?limit=20'),
                fetchAPI('/kalshi/markets?limit=100'),
                fetchAPI('/kalshi/consensus?limit=20'),
                fetchAPI('/arbitrage/opportunities?limit=50'),
            ]);

            setStats(statsData);
            setMarkets(marketsData);
            setPredictions(predictionsData);
            setConsensus(consensusData);
            setKalshiMarkets(kalshiMarketsData);
            setKalshiConsensus(kalshiConsensusData);
            setArbitrageOpps(arbData);
            setLastRefresh(new Date().toLocaleTimeString());
        } catch (error) {
            console.error('Error fetching data:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        refreshData();
        const interval = setInterval(refreshData, 30000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="dashboard">
            <header>
                <div>
                    <h1>Prediction Market Dashboard</h1>
                    <span className="refresh-indicator">
                        {lastRefresh ? `Last updated: ${lastRefresh}` : 'Loading...'}
                        {' | Auto-refresh: 30s'}
                    </span>
                </div>
                <StatsBar stats={stats} />
            </header>

            <div className="tabs">
                <button
                    className={`tab ${activeTab === 'consensus' ? 'active' : ''}`}
                    onClick={() => setActiveTab('consensus')}
                >
                    Poly Picks
                </button>
                <button
                    className={`tab ${activeTab === 'predictions' ? 'active' : ''}`}
                    onClick={() => setActiveTab('predictions')}
                >
                    Poly Predictions
                </button>
                <button
                    className={`tab ${activeTab === 'markets' ? 'active' : ''}`}
                    onClick={() => setActiveTab('markets')}
                >
                    Poly Markets
                </button>

                <div className="tab-divider"></div>

                <button
                    className={`tab ${activeTab === 'kalshi_picks' ? 'active' : ''}`}
                    onClick={() => setActiveTab('kalshi_picks')}
                >
                    Kalshi Picks
                </button>
                <button
                    className={`tab ${activeTab === 'kalshi_markets' ? 'active' : ''}`}
                    onClick={() => setActiveTab('kalshi_markets')}
                >
                    Kalshi Markets
                </button>

                <div className="tab-divider"></div>

                <button
                    className={`tab ${activeTab === 'arbitrage' ? 'active' : ''}`}
                    onClick={() => setActiveTab('arbitrage')}
                    style={stats.arbitrage_profitable > 0 ? {borderColor: '#00ff88'} : {}}
                >
                    Arbitrage {stats.arbitrage_profitable > 0 ? `(${stats.arbitrage_profitable})` : ''}
                </button>
            </div>

            <div className="panel">
                {loading ? (
                    <div className="loading">Loading data...</div>
                ) : (
                    <>
                        {activeTab === 'consensus' && <ConsensusPicks picks={consensus} />}
                        {activeTab === 'predictions' && <PredictionsTable predictions={predictions} />}
                        {activeTab === 'markets' && <MarketsTable markets={markets} />}
                        {activeTab === 'kalshi_picks' && <KalshiConsensusPicks picks={kalshiConsensus} />}
                        {activeTab === 'kalshi_markets' && <KalshiMarketsTable markets={kalshiMarkets} />}
                        {activeTab === 'arbitrage' && <ArbitrageTable opportunities={arbitrageOpps} />}
                    </>
                )}
            </div>
        </div>
    );
}

// Render the app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<Dashboard />);
