import { memo, useMemo } from 'react'
import { theme } from '../styles/theme'

function StatsCards({ currentStep, maxStep, bestLoss, runInfo }) {
  // Calculate ETA and elapsed time
  const { elapsed, eta } = useMemo(() => {
    if (!runInfo?.start_time) {
      return { elapsed: null, eta: null }
    }

    const startTime = new Date(runInfo.start_time)
    const now = new Date()
    const elapsedMs = now - startTime
    const elapsedSec = elapsedMs / 1000

    // Format elapsed time
    const hours = Math.floor(elapsedSec / 3600)
    const minutes = Math.floor((elapsedSec % 3600) / 60)
    const elapsed = hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`

    // Estimate remaining time (rough estimate based on step progress)
    // This is a placeholder - real ETA would need tokens/sec and target steps
    const progress = maxStep > 0 ? currentStep / maxStep : 0
    const eta = progress > 0.1
      ? `~${Math.round((elapsedSec / progress - elapsedSec) / 60)}m remaining`
      : 'Calculating...'

    return { elapsed, eta }
  }, [runInfo, currentStep, maxStep])

  return (
    <div className="grid-row grid-row-4">
      <div className="card">
        <div className="card-title">Current Step</div>
        <div className="card-value">{currentStep?.toLocaleString() || '—'}</div>
        <div className="card-subtitle">of {maxStep?.toLocaleString() || '—'} recorded</div>
      </div>

      <div className="card">
        <div className="card-title">Best Loss</div>
        <div className="card-value" style={{ color: theme.chart.emaLong }}>
          {bestLoss != null ? bestLoss.toFixed(4) : '—'}
        </div>
        <div className="card-subtitle">
          {runInfo?.best_loss_step ? `at step ${runInfo.best_loss_step.toLocaleString()}` : ''}
        </div>
      </div>

      <div className="card">
        <div className="card-title">Elapsed Time</div>
        <div className="card-value">{elapsed || '—'}</div>
        <div className="card-subtitle">{eta || ''}</div>
      </div>

      <div className="card">
        <div className="card-title">Total Tokens</div>
        <div className="card-value">
          {runInfo?.total_tokens
            ? `${(runInfo.total_tokens / 1e9).toFixed(2)}B`
            : '—'}
        </div>
        <div className="card-subtitle">tokens processed</div>
      </div>
    </div>
  )
}

export default memo(StatsCards)
