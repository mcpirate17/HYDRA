import { memo, useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import { theme, chartConfig } from '../styles/theme'

// Layer colors
const layerColors = [
  '#3b82f6', // blue
  '#22c55e', // green
  '#f97316', // orange
  '#a855f7', // purple
  '#ec4899', // pink
  '#14b8a6', // teal
  '#eab308', // yellow
  '#ef4444', // red
]

function MoRDepthChart({ data }) {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return { data: [], layers: [] }

    // Get unique layers and steps
    const layerSet = new Set()
    const stepMap = new Map()

    data.forEach(d => {
      if (d.layer != null) layerSet.add(d.layer)

      if (!stepMap.has(d.step)) {
        stepMap.set(d.step, { step: d.step })
      }

      const stepData = stepMap.get(d.step)
      stepData[`depth_${d.layer}`] = d.avg_depth
      stepData[`expected_${d.layer}`] = d.expected_depth
    })

    const layers = Array.from(layerSet).sort((a, b) => a - b)
    let chartData = Array.from(stepMap.values()).sort((a, b) => a.step - b.step)

    // Sample if too many points
    const maxPoints = 200
    if (chartData.length > maxPoints) {
      const interval = Math.ceil(chartData.length / maxPoints)
      chartData = chartData.filter((_, i) => i % interval === 0)
    }

    return { data: chartData, layers }
  }, [data])

  if (chartData.data.length === 0) {
    return <div className="empty-state">No MoR routing data available</div>
  }

  // Check for depth collapse (avg_depth near 1.0)
  const hasCollapse = data.some(d => d.avg_depth != null && d.avg_depth < 1.2)

  return (
    <div style={{ width: '100%', height: '100%' }}>
      {hasCollapse && (
        <div style={{
          padding: '4px 8px',
          background: 'rgba(239, 68, 68, 0.1)',
          borderRadius: 4,
          fontSize: 11,
          color: theme.status.error,
          marginBottom: 8,
        }}>
          Warning: MoR depth collapsed (avg_depth ~1.0)
        </div>
      )}

      <ResponsiveContainer width="100%" height={hasCollapse ? '90%' : '100%'}>
        <LineChart data={chartData.data} margin={chartConfig.margin}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke={chartConfig.gridColor}
            vertical={false}
          />
          <XAxis
            dataKey="step"
            stroke={chartConfig.axisColor}
            fontSize={chartConfig.fontSize}
            tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(0)}K` : v}
          />
          <YAxis
            stroke={chartConfig.axisColor}
            fontSize={chartConfig.fontSize}
            domain={[0, 'auto']}
            label={{
              value: 'Depth',
              angle: -90,
              position: 'insideLeft',
              fill: theme.text.muted,
              fontSize: 10,
            }}
          />
          <Tooltip
            contentStyle={{
              background: theme.bg.secondary,
              border: `1px solid ${theme.border.default}`,
              borderRadius: 6,
              fontSize: 12,
            }}
            labelFormatter={(v) => `Step ${v?.toLocaleString()}`}
          />
          <Legend
            wrapperStyle={{ fontSize: 10 }}
            formatter={(value) => {
              if (value.startsWith('depth_')) return `L${value.slice(6)} Actual`
              if (value.startsWith('expected_')) return `L${value.slice(9)} Expected`
              return value
            }}
          />

          {/* Render lines for each layer */}
          {chartData.layers.slice(0, 4).map((layer, i) => (
            <Line
              key={`depth_${layer}`}
              type="monotone"
              dataKey={`depth_${layer}`}
              stroke={layerColors[i % layerColors.length]}
              strokeWidth={2}
              dot={false}
              name={`depth_${layer}`}
            />
          ))}

          {/* Expected depth lines (dashed) */}
          {chartData.layers.slice(0, 4).map((layer, i) => (
            <Line
              key={`expected_${layer}`}
              type="monotone"
              dataKey={`expected_${layer}`}
              stroke={layerColors[i % layerColors.length]}
              strokeWidth={1}
              strokeDasharray="5 5"
              dot={false}
              name={`expected_${layer}`}
              opacity={0.5}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default memo(MoRDepthChart)
