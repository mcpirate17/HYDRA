import { memo, useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Dot,
} from 'recharts'
import { theme, chartConfig } from '../styles/theme'

// Custom dot for spike highlighting
function SpikeDot(props) {
  const { cx, cy, payload } = props
  if (!payload) return null

  // Highlight if loss > ema_short * 1.1 (spike detection)
  const isSpike = payload.loss_total > (payload.ema_short || payload.loss_total) * 1.1

  if (!isSpike) return null

  return (
    <circle
      cx={cx}
      cy={cy}
      r={4}
      fill={theme.chart.loss}
      stroke="#fff"
      strokeWidth={1}
    />
  )
}

function LossChart({ data }) {
  // Sample data if too large
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return []

    // Sample to max 500 points for performance
    const maxPoints = 500
    if (data.length <= maxPoints) return data

    const step = Math.ceil(data.length / maxPoints)
    return data.filter((_, i) => i % step === 0)
  }, [data])

  if (chartData.length === 0) {
    return <div className="empty-state">No loss data available</div>
  }

  // Calculate Y domain with padding
  const losses = chartData.map(d => d.loss_total).filter(v => v != null && isFinite(v))
  const minLoss = Math.min(...losses)
  const maxLoss = Math.max(...losses)
  const yPadding = (maxLoss - minLoss) * 0.1
  const yDomain = [
    Math.max(0, minLoss - yPadding),
    maxLoss + yPadding
  ]

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
          domain={yDomain}
          tickFormatter={(v) => v.toFixed(2)}
        />
        <Tooltip
          contentStyle={{
            background: theme.bg.secondary,
            border: `1px solid ${theme.border.default}`,
            borderRadius: 6,
            fontSize: 12,
          }}
          labelFormatter={(v) => `Step ${v.toLocaleString()}`}
          formatter={(value, name) => {
            const labels = {
              loss_total: 'Loss',
              ema_short: 'EMA Short',
              ema_medium: 'EMA Medium',
              ema_long: 'EMA Long',
            }
            return [value?.toFixed(4), labels[name] || name]
          }}
        />

        {/* Target line at 2.8 */}
        <ReferenceLine
          y={2.8}
          stroke={theme.chart.target}
          strokeDasharray="5 5"
          label={{
            value: 'Target',
            fill: theme.chart.target,
            fontSize: 10,
            position: 'right',
          }}
        />

        {/* EMA lines (render first so they're behind) */}
        <Line
          type="monotone"
          dataKey="ema_long"
          stroke={theme.chart.emaLong}
          strokeWidth={1.5}
          dot={false}
          name="ema_long"
        />
        <Line
          type="monotone"
          dataKey="ema_medium"
          stroke={theme.chart.emaMedium}
          strokeWidth={1.5}
          dot={false}
          name="ema_medium"
        />
        <Line
          type="monotone"
          dataKey="ema_short"
          stroke={theme.chart.emaShort}
          strokeWidth={1.5}
          dot={false}
          name="ema_short"
        />

        {/* Main loss line with spike highlighting */}
        <Line
          type="monotone"
          dataKey="loss_total"
          stroke={theme.chart.loss}
          strokeWidth={2}
          dot={<SpikeDot />}
          activeDot={{ r: 4, fill: theme.chart.loss }}
          name="loss_total"
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

export default memo(LossChart)
