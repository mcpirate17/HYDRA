import { memo, useMemo } from 'react'
import { theme } from '../styles/theme'

/**
 * MoD Capacity Heatmap
 *
 * Displays per-layer token selection fraction over steps.
 * Color scale: 0.75 = green (target), <0.70 or >0.80 = yellow/red
 */
function MoDHeatmap({ data }) {
  const { gridData, layers, steps } = useMemo(() => {
    if (!data || data.length === 0) {
      return { gridData: [], layers: [], steps: [] }
    }

    // Get unique layers and steps
    const layerSet = new Set()
    const stepSet = new Set()

    data.forEach(d => {
      if (d.layer != null) layerSet.add(d.layer)
      if (d.step != null) stepSet.add(d.step)
    })

    const layers = Array.from(layerSet).sort((a, b) => a - b)
    const steps = Array.from(stepSet).sort((a, b) => a - b)

    // Sample steps if too many
    let sampledSteps = steps
    const maxSteps = 50
    if (steps.length > maxSteps) {
      const interval = Math.ceil(steps.length / maxSteps)
      sampledSteps = steps.filter((_, i) => i % interval === 0)
    }

    // Create lookup map
    const lookup = new Map()
    data.forEach(d => {
      const key = `${d.step}-${d.layer}`
      lookup.set(key, d)
    })

    // Build grid data
    const gridData = []
    sampledSteps.forEach((step, stepIdx) => {
      layers.forEach((layer, layerIdx) => {
        const d = lookup.get(`${step}-${layer}`)
        if (d) {
          gridData.push({
            stepIdx,
            layerIdx,
            step,
            layer,
            value: d.selected_frac,
            computeSavings: d.compute_savings_pct,
          })
        }
      })
    })

    return { gridData, layers, steps: sampledSteps }
  }, [data])

  if (gridData.length === 0) {
    return <div className="empty-state">No MoD routing data available</div>
  }

  // Cell dimensions
  const cellWidth = Math.max(8, Math.min(20, 600 / steps.length))
  const cellHeight = Math.max(12, Math.min(24, 200 / layers.length))
  const marginLeft = 40
  const marginTop = 20
  const marginBottom = 30

  // Get color for value (target is 0.75)
  const getColor = (value) => {
    if (value == null) return theme.bg.tertiary

    const deviation = Math.abs(value - 0.75)

    if (deviation <= 0.05) {
      // On target - green
      return theme.heatmap.target
    } else if (deviation <= 0.1) {
      // Slight deviation - yellow
      return theme.heatmap.high
    } else {
      // Large deviation - red
      return theme.heatmap.low
    }
  }

  return (
    <div style={{ width: '100%', height: '100%', overflow: 'hidden' }}>
      <svg
        width="100%"
        height="100%"
        viewBox={`0 0 ${marginLeft + steps.length * cellWidth + 10} ${marginTop + layers.length * cellHeight + marginBottom}`}
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Y-axis labels (layers) */}
        {layers.map((layer, i) => (
          <text
            key={`layer-${layer}`}
            x={marginLeft - 5}
            y={marginTop + i * cellHeight + cellHeight / 2}
            textAnchor="end"
            dominantBaseline="middle"
            fill={theme.text.muted}
            fontSize={10}
          >
            L{layer}
          </text>
        ))}

        {/* X-axis labels (steps) */}
        {steps.filter((_, i) => i % Math.ceil(steps.length / 8) === 0).map((step, i) => (
          <text
            key={`step-${step}`}
            x={marginLeft + steps.indexOf(step) * cellWidth + cellWidth / 2}
            y={marginTop + layers.length * cellHeight + 15}
            textAnchor="middle"
            fill={theme.text.muted}
            fontSize={9}
          >
            {step >= 1000 ? `${(step / 1000).toFixed(0)}K` : step}
          </text>
        ))}

        {/* Heatmap cells */}
        {gridData.map((d, i) => (
          <g key={i}>
            <rect
              x={marginLeft + d.stepIdx * cellWidth}
              y={marginTop + d.layerIdx * cellHeight}
              width={cellWidth - 1}
              height={cellHeight - 1}
              fill={getColor(d.value)}
              rx={2}
            >
              <title>
                Step {d.step.toLocaleString()}, Layer {d.layer}
                {'\n'}Selected: {(d.value * 100).toFixed(1)}%
                {'\n'}Compute savings: {d.computeSavings?.toFixed(1)}%
              </title>
            </rect>
          </g>
        ))}

        {/* Legend */}
        <g transform={`translate(${marginLeft}, ${marginTop + layers.length * cellHeight + 20})`}>
          <rect x={0} y={0} width={12} height={12} fill={theme.heatmap.target} rx={2} />
          <text x={16} y={10} fill={theme.text.muted} fontSize={9}>~75%</text>

          <rect x={50} y={0} width={12} height={12} fill={theme.heatmap.high} rx={2} />
          <text x={66} y={10} fill={theme.text.muted} fontSize={9}>65-85%</text>

          <rect x={110} y={0} width={12} height={12} fill={theme.heatmap.low} rx={2} />
          <text x={126} y={10} fill={theme.text.muted} fontSize={9}>{"<65% / >85%"}</text>
        </g>
      </svg>
    </div>
  )
}

export default memo(MoDHeatmap)
