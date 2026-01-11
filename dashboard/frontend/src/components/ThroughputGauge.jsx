import { memo } from 'react'
import { theme } from '../styles/theme'

function ThroughputGauge({ current, average, target = 12000 }) {
  const percentage = Math.min(100, (current / target) * 100)

  // Color based on performance
  let color = theme.status.success
  if (percentage < 50) {
    color = theme.status.error
  } else if (percentage < 75) {
    color = theme.status.warning
  }

  const formatTps = (v) => {
    if (v >= 1000) {
      return `${(v / 1000).toFixed(1)}K`
    }
    return v.toFixed(0)
  }

  return (
    <div className="gauge-container">
      <div className="gauge-value" style={{ color }}>
        {formatTps(current)}
        <span style={{ fontSize: '14px', color: theme.text.muted }}> tok/s</span>
      </div>

      <div className="gauge-bar">
        <div
          className="gauge-fill"
          style={{
            width: `${percentage}%`,
            background: color,
          }}
        />
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', fontSize: 11 }}>
        <span style={{ color: theme.text.muted }}>
          Avg: {formatTps(average)} tok/s
        </span>
        <span style={{ color: theme.text.muted }}>
          Target: {formatTps(target)} tok/s
        </span>
      </div>
    </div>
  )
}

export default memo(ThroughputGauge)
