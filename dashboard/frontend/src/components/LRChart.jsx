import { memo, useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { theme, chartConfig } from '../styles/theme'

function LRChart({ data }) {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return []

    // Sample to max 500 points
    const maxPoints = 500
    if (data.length <= maxPoints) return data

    const step = Math.ceil(data.length / maxPoints)
    return data.filter((_, i) => i % step === 0)
  }, [data])

  if (chartData.length === 0) {
    return <div className="empty-state">No learning rate data available</div>
  }

  // Get LR range
  const lrs = chartData.map(d => d.lr).filter(v => v != null && v > 0)
  const maxLr = Math.max(...lrs)
  const minLr = Math.min(...lrs)

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={chartData} margin={chartConfig.margin}>
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
          domain={[0, maxLr * 1.1]}
          tickFormatter={(v) => v.toExponential(1)}
        />
        <Tooltip
          contentStyle={{
            background: theme.bg.secondary,
            border: `1px solid ${theme.border.default}`,
            borderRadius: 6,
            fontSize: 12,
          }}
          labelFormatter={(v) => `Step ${v?.toLocaleString()}`}
          formatter={(value) => [value?.toExponential(4), 'Learning Rate']}
        />

        <Line
          type="monotone"
          dataKey="lr"
          stroke={theme.chart.lr}
          strokeWidth={2}
          dot={false}
          name="lr"
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

export default memo(LRChart)
