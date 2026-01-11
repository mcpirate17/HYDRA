import { memo } from 'react'

function Header({
  models,
  runs,
  selectedModel,
  selectedRun,
  isLive,
  onModelChange,
  onRunChange,
  onToggleLive,
  currentStep,
  bestLoss,
}) {
  return (
    <header className="dashboard-header">
      <h1>
        <span>HYDRA</span> Training Dashboard
      </h1>

      <div className="header-controls">
        <select
          value={selectedModel}
          onChange={(e) => onModelChange(e.target.value)}
          disabled={models.length === 0}
        >
          <option value="">Select Model</option>
          {models.map((m) => (
            <option key={m.model_id} value={m.model_id}>
              {m.name || m.model_id}
              {m.params_millions ? ` (${m.params_millions}M)` : ''}
            </option>
          ))}
        </select>

        <select
          value={selectedRun}
          onChange={(e) => onRunChange(e.target.value)}
          disabled={runs.length === 0}
        >
          <option value="">Select Run</option>
          {runs.map((r) => (
            <option key={r.run_id} value={r.run_id}>
              {r.run_id}
              {r.start_time ? ` (${new Date(r.start_time).toLocaleDateString()})` : ''}
            </option>
          ))}
        </select>

        <button
          className={`toggle-btn ${isLive ? 'active' : ''}`}
          onClick={onToggleLive}
          disabled={!selectedRun}
        >
          <span className="toggle-indicator" />
          {isLive ? 'Live' : 'Paused'}
        </button>

        {currentStep !== undefined && (
          <div className="badge">
            <span className="badge-label">Step</span>
            <span className="badge-value">{currentStep?.toLocaleString()}</span>
          </div>
        )}

        {bestLoss !== undefined && bestLoss !== null && (
          <div className="badge">
            <span className="badge-label">Best</span>
            <span className="badge-value">{bestLoss?.toFixed(4)}</span>
          </div>
        )}
      </div>
    </header>
  )
}

export default memo(Header)
