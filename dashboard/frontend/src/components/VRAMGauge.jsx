import { memo } from 'react'
import { theme } from '../styles/theme'

function VRAMGauge({ current, max = 32 }) {
  const percentage = Math.min(100, (current / max) * 100)

  // Color zones: green (<24), yellow (24-28), red (>28)
  let color = theme.vram.safe
  if (current >= 28) {
    color = theme.vram.danger
  } else if (current >= 24) {
    color = theme.vram.warning
  }

  return (
    <div className="gauge-container">
      <div className="gauge-value" style={{ color }}>
        {current.toFixed(1)}
        <span style={{ fontSize: '14px', color: theme.text.muted }}> GB</span>
      </div>

      <div className="gauge-bar" style={{ position: 'relative' }}>
        <div
          className="gauge-fill"
          style={{
            width: `${percentage}%`,
            background: color,
          }}
        />
        {/* Zone markers */}
        <div
          style={{
            position: 'absolute',
            left: `${(24 / max) * 100}%`,
            top: 0,
            bottom: 0,
            width: 1,
            background: theme.vram.warning,
            opacity: 0.5,
          }}
        />
        <div
          style={{
            position: 'absolute',
            left: `${(28 / max) * 100}%`,
            top: 0,
            bottom: 0,
            width: 1,
            background: theme.vram.danger,
            opacity: 0.5,
          }}
        />
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', fontSize: 11 }}>
        <span style={{ color: theme.text.muted }}>0 GB</span>
        <span style={{ color: theme.vram.warning, fontSize: 10 }}>24</span>
        <span style={{ color: theme.vram.danger, fontSize: 10 }}>28</span>
        <span style={{ color: theme.text.muted }}>{max} GB</span>
      </div>
    </div>
  )
}

export default memo(VRAMGauge)
