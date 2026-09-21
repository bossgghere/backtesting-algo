import { useState } from 'react'
import EquityChart from './components/EquityChart'

// ── Screen 1: Entry ────────────────────────────────────────────────────────
function EntryScreen({ onGenerated }) {
  const [prompt, setPrompt]   = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  const generate = async () => {
    setLoading(true)
    setError(null)
    try {
      const res  = await fetch('/api/generate', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ prompt }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Generation failed')
      onGenerated({ code: data.code, params: data.params, prompt })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="screen entry-screen">
      <div className="entry-inner">
        <div className="brand">
          <span className="brand-name">studio.trade</span>
          <span className="brand-sub">Describe a trading strategy. We'll backtest it on Nifty 50.</span>
        </div>

        <textarea
          className="strategy-input"
          placeholder="e.g. Buy when RSI drops below 30, sell when it crosses above 70"
          value={prompt}
          onChange={e => { setPrompt(e.target.value); setError(null) }}
          disabled={loading}
          rows={5}
          onKeyDown={e => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && prompt.trim().length >= 10) generate()
          }}
        />

        {error && <div className="input-error">{error}</div>}

        <button
          className={`primary-btn ${loading ? 'loading' : ''}`}
          onClick={generate}
          disabled={loading || prompt.trim().length < 10}
        >
          {loading
            ? <><span className="btn-spinner" />Generating strategy…</>
            : 'Generate →'}
        </button>
      </div>
    </div>
  )
}

// ── Screen 2: Parameters ───────────────────────────────────────────────────
function ParamsScreen({ code, params, onResults, onBack }) {
  const [values, setValues] = useState(() => {
    const init = {}
    params.forEach(p => { init[p.name] = p.default })
    return init
  })
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      const res  = await fetch('/api/run', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ code, params: values, symbol: '^NSEI', period: '1y', interval: '1d' }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Backtest failed')
      onResults({ ...data, paramValues: values, code })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const update = (name, val) => setValues(v => ({ ...v, [name]: val }))

  return (
    <div className="screen params-screen">
      <div className="params-inner">
        <button className="back-link" onClick={onBack}>← Back</button>

        <div className="params-title">Tune your strategy</div>
        <div className="params-sub">
          {params.length > 0
            ? 'Adjust the parameters below or run with defaults.'
            : 'No tunable parameters detected. Ready to run.'}
        </div>

        {params.length > 0 && (
          <div className="sliders">
            {params.map(p => (
              <SliderRow
                key={p.name}
                param={p}
                value={values[p.name]}
                onChange={val => update(p.name, val)}
              />
            ))}
          </div>
        )}

        {error && <div className="input-error">{error}</div>}

        <button
          className={`primary-btn ${loading ? 'loading' : ''}`}
          onClick={run}
          disabled={loading}
        >
          {loading
            ? <><span className="btn-spinner" />Running backtest…</>
            : 'Run Backtest →'}
        </button>
      </div>
    </div>
  )
}

function SliderRow({ param, value, onChange }) {
  const isFloat = param.type === 'float'
  const display = isFloat ? Number(value).toFixed(1) : value

  const handleChange = (e) => {
    const raw = isFloat ? parseFloat(e.target.value) : parseInt(e.target.value)
    onChange(raw)
  }

  return (
    <div className="slider-row">
      <div className="slider-top">
        <span className="slider-label">{param.label}</span>
        <span className="slider-value">{display}</span>
      </div>
      <input
        type="range"
        min={param.min}
        max={param.max}
        step={param.step}
        value={value}
        onChange={handleChange}
        className="slider"
      />
      <div className="slider-range">
        <span>{param.min}</span>
        <span>{param.max}</span>
      </div>
    </div>
  )
}

// ── Screen 3: Results ──────────────────────────────────────────────────────
function ResultsScreen({ data, onBack, onTweak }) {
  const [showCode, setShowCode]         = useState(false)
  const [showAllTrades, setShowAllTrades] = useState(false)
  const s      = data.summary
  const trades = showAllTrades ? data.trades : data.trades.slice(-15)
  const pnl    = s.final_capital - s.initial_capital

  return (
    <div className="screen results-screen">
      <div className="results-topbar">
        <button className="back-btn" onClick={onBack}>← New Strategy</button>
        <div className="results-meta">
          <span className="meta-badge">{data.symbol}</span>
          <span className="meta-badge">1y · 1d</span>
          <span className="meta-badge">{s.total_trades} trades</span>
          <span className="meta-badge dim">#{data.fingerprint}</span>
        </div>
        <button className="tweak-btn" onClick={onTweak}>Tweak Parameters</button>
      </div>

      <div className="results-body">
        {/* Hero */}
        <div className="hero-section">
          <div className="hero-left">
            <span className="hero-label">Total Return</span>
            <span className={`hero-value ${pnl >= 0 ? 'positive' : 'negative'}`}>
              {pnl >= 0 ? '+' : ''}{s.total_return_pct}%
            </span>
            <span className="hero-sub">
              ₹{s.initial_capital.toLocaleString('en-IN')} → ₹{s.final_capital.toLocaleString('en-IN')}
              <span className={pnl >= 0 ? 'positive' : 'negative'} style={{ marginLeft: 8 }}>
                ({pnl >= 0 ? '+' : ''}₹{Math.round(pnl).toLocaleString('en-IN')})
              </span>
            </span>
          </div>
          <div className="hero-right">
            <div className="hero-compare">
              <div className="compare-item">
                <span className="compare-label">Strategy</span>
                <span className={`compare-val ${s.total_return_pct >= 0 ? 'positive' : 'negative'}`}>
                  {s.total_return_pct >= 0 ? '+' : ''}{s.total_return_pct}%
                </span>
              </div>
              <div className="compare-sep">vs</div>
              <div className="compare-item">
                <span className="compare-label">Buy &amp; Hold</span>
                <span className={`compare-val ${s.buy_and_hold_pct >= 0 ? 'positive' : 'negative'}`}>
                  {s.buy_and_hold_pct >= 0 ? '+' : ''}{s.buy_and_hold_pct}%
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Stats Row 1 */}
        <div className="stats-grid-4">
          <StatCard label="Win Rate"      value={`${s.win_rate_pct}%`}           sub={`${s.winning_trades}W / ${s.losing_trades}L`}  positive={s.win_rate_pct >= 50} />
          <StatCard label="Sharpe Ratio"  value={s.sharpe_ratio}                  sub="risk-adjusted return"                           positive={s.sharpe_ratio >= 1} />
          <StatCard label="Max Drawdown"  value={`${s.max_drawdown_pct}%`}        sub="peak to trough"                                 positive={s.max_drawdown_pct > -10} />
          <StatCard label="Profit Factor" value={`${s.profit_factor}x`}           sub="gross profit / gross loss"                     positive={s.profit_factor >= 1} />
        </div>

        {/* Stats Row 2 */}
        <div className="stats-grid-4">
          <StatCard label="Avg Win"            value={`+₹${Math.round(s.avg_win_inr).toLocaleString('en-IN')}`}   sub="avg profit per win"     positive={true} />
          <StatCard label="Avg Loss"           value={`−₹${Math.round(s.avg_loss_inr).toLocaleString('en-IN')}`}  sub="avg loss per loss"      positive={false} />
          <StatCard label="Max Losing Streak"  value={s.max_consecutive_losses}                                     sub="consecutive losses"     positive={s.max_consecutive_losses <= 3} />
          <StatCard label="Avg Hold"           value={`${s.avg_holding_days}d`}                                     sub="average trade duration" positive={true} />
        </div>

        {/* Chart */}
        <div className="section">
          <div className="section-title">Equity Curve</div>
          <div className="chart-card"><EquityChart data={data.equity_curve} /></div>
        </div>

        {/* Trades */}
        <div className="section">
          <div className="section-header">
            <div className="section-title">Trade Log</div>
            <span className="section-meta">{data.trades.length} total</span>
          </div>
          <div className="table-card">
            <table className="trade-table">
              <thead>
                <tr>
                  <th>#</th><th>Type</th><th>Entry</th><th>Exit</th>
                  <th>Entry ₹</th><th>Exit ₹</th><th>PnL %</th><th>Net ₹</th><th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {trades.map(t => (
                  <tr key={t.trade_num}>
                    <td className="dim mono">{t.trade_num}</td>
                    <td><span className={`type-badge ${t.type === 'LONG' ? 'long' : 'short'}`}>{t.type}</span></td>
                    <td className="mono">{t.entry_date}</td>
                    <td className="mono">{t.exit_date}</td>
                    <td className="mono">₹{t.entry_price.toLocaleString('en-IN')}</td>
                    <td className="mono">₹{t.exit_price.toLocaleString('en-IN')}</td>
                    <td className={`mono bold ${t.pnl_pct >= 0 ? 'positive' : 'negative'}`}>
                      {t.pnl_pct >= 0 ? '+' : ''}{t.pnl_pct}%
                    </td>
                    <td className={`mono bold ${t.net_pnl >= 0 ? 'positive' : 'negative'}`}>
                      {t.net_pnl >= 0 ? '+' : ''}₹{Math.round(t.net_pnl).toLocaleString('en-IN')}
                    </td>
                    <td className="dim">{t.exit_reason || 'SIGNAL'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {data.trades.length > 15 && (
            <button className="show-more-btn" onClick={() => setShowAllTrades(v => !v)}>
              {showAllTrades ? 'Show last 15' : `Show all ${data.trades.length} trades`}
            </button>
          )}
        </div>

        {/* Code */}
        {data.code && (
          <div className="section">
            <button className="toggle-code-btn" onClick={() => setShowCode(v => !v)}>
              {showCode ? '↑ Hide' : '↓ Show'} generated strategy code
            </button>
            {showCode && <pre className="code-block"><code>{data.code}</code></pre>}
          </div>
        )}
      </div>
    </div>
  )
}

function StatCard({ label, value, sub, positive }) {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <span className={`stat-value ${positive ? 'positive' : 'negative'}`}>{value}</span>
      <span className="stat-sub">{sub}</span>
    </div>
  )
}

// ── Root ──────────────────────────────────────────────────────────────────
export default function App() {
  const [screen, setScreen]   = useState('entry')   // 'entry' | 'params' | 'results'
  const [generated, setGenerated] = useState(null)  // { code, params }
  const [results, setResults] = useState(null)

  const handleGenerated = (data) => {
    setGenerated(data)
    setScreen('params')
  }

  const handleResults = (data) => {
    setResults(data)
    setScreen('results')
  }

  if (screen === 'entry') {
    return <EntryScreen onGenerated={handleGenerated} />
  }

  if (screen === 'params') {
    return (
      <ParamsScreen
        code={generated.code}
        params={generated.params}
        onResults={handleResults}
        onBack={() => setScreen('entry')}
      />
    )
  }

  return (
    <ResultsScreen
      data={results}
      onBack={() => setScreen('entry')}
      onTweak={() => setScreen('params')}
    />
  )
}
