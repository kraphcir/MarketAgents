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
                <div className="stat-label">Markets</div>
            </div>
            <div className="stat">
                <div className="stat-value">{stats.total_predictions}</div>
                <div className="stat-label">Predictions</div>
            </div>
            <div className="stat">
                <div className="stat-value">{stats.total_consensus_picks}</div>
                <div className="stat-label">Consensus Picks</div>
            </div>
        </div>
    );
}

// Markets Table Component
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

// Predictions Table Component
function PredictionsTable({ predictions }) {
    if (!predictions.length) {
        return (
            <div className="empty-state">
                <h3>No Predictions Yet</h3>
                <p>AI analysis will appear here after the agent runs</p>
            </div>
        );
    }

    const getBadgeClass = (pred) => {
        if (!pred || pred === 'N/A') return '';
        const p = String(pred).toUpperCase();
        if (p.includes('YES')) return 'badge badge-yes';
        if (p.includes('NO_TRADE') || p.includes('NO TRADE')) return 'badge badge-notrade';
        if (p.includes('NO')) return 'badge badge-no';
        return '';
    };

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

// Consensus Picks Component
function ConsensusPicks({ picks }) {
    if (!picks.length) {
        return (
            <div className="empty-state">
                <h3>No Consensus Picks Yet</h3>
                <p>Top AI recommendations will appear here after analysis runs</p>
            </div>
        );
    }

    const getSideClass = (side) => {
        const s = String(side).toUpperCase();
        if (s === 'YES') return 'consensus-yes';
        if (s === 'NO') return 'consensus-no';
        return 'consensus-notrade';
    };

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

// Main Dashboard Component
function Dashboard() {
    const [activeTab, setActiveTab] = useState('consensus');
    const [stats, setStats] = useState({ total_markets: 0, total_predictions: 0, total_consensus_picks: 0 });
    const [markets, setMarkets] = useState([]);
    const [predictions, setPredictions] = useState([]);
    const [consensus, setConsensus] = useState([]);
    const [loading, setLoading] = useState(true);
    const [lastRefresh, setLastRefresh] = useState(null);

    const refreshData = async () => {
        try {
            const [statsData, marketsData, predictionsData, consensusData] = await Promise.all([
                fetchAPI('/stats'),
                fetchAPI('/markets?limit=100'),
                fetchAPI('/predictions?limit=50'),
                fetchAPI('/consensus?limit=20')
            ]);

            setStats(statsData);
            setMarkets(marketsData);
            setPredictions(predictionsData);
            setConsensus(consensusData);
            setLastRefresh(new Date().toLocaleTimeString());
        } catch (error) {
            console.error('Error fetching data:', error);
        } finally {
            setLoading(false);
        }
    };

    // Initial load and auto-refresh every 30 seconds
    useEffect(() => {
        refreshData();
        const interval = setInterval(refreshData, 30000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="dashboard">
            <header>
                <div>
                    <h1>Polymarket Agent Dashboard</h1>
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
                    Top Picks
                </button>
                <button
                    className={`tab ${activeTab === 'predictions' ? 'active' : ''}`}
                    onClick={() => setActiveTab('predictions')}
                >
                    Predictions
                </button>
                <button
                    className={`tab ${activeTab === 'markets' ? 'active' : ''}`}
                    onClick={() => setActiveTab('markets')}
                >
                    Markets
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
                    </>
                )}
            </div>
        </div>
    );
}

// Render the app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<Dashboard />);
