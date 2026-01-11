import { memo, useMemo } from 'react'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Line,
} from 'recharts'
import { theme, chartConfig } from '../styles/theme'

function GradNormChart({ data }) {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return []

    // Sample to max 500 points
    const maxPoints = 500
    if (data.length <= maxPoints) return data

    const step = Math.ceil(data.length / maxPoints)
    return data.filter((_, i) => i % step === 0)
  }, [data])

  if (chartData.length === 0) {
    return <div className="empty-state">No gradient data available</div>
  }

  // Calculate Y domain
  const norms = chartData
    .map(d => d.grad_norm_pre_clip || d.grad_norm)
    .filter(v => v != null && isFinite(v) && v < 100)
  const maxNorm = Math.max(...norms, 1.5)

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={chartData} margin={chartConfig.margin}>
        <defs>
          <linearGradient id="gradNormGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={theme.chart.gradNorm} stopOpacity={0.3} />
            <stop offset="95%" stopColor={theme.chart.gradNorm} stopOpacity={0} />
          </linearGradient>
        </defs>
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
          domain={[0, maxNorm]}
          tickFormatter={(v) => v.toFixed(1)}
        />
        <Tooltip
          contentStyle={{
            background: theme.bg.secondary,
            border: `1px solid ${theme.border.default}`,
            borderRadius: 6,
            fontSize: 12,
          }}
          labelFormatter={(v) => `Step ${v?.toLocaleString()}`}
          formatter={(value, name) => {
            const labels = {
              grad_norm: 'Grad Norm (clipped)',
              grad_norm_pre_clip: 'Grad Norm (raw)',
            }
            return [value?.toFixed(4), labels[name] || name]
          }}
        />

        {/* Clip threshold line */}
        <ReferenceLine
          y={1.0}
          stroke={theme.status.warning}
          strokeDasharray="5 5"
          label={{
            value: 'Clip',
            fill: theme.status.warning,
            fontSize: 10,
            position: 'right',
          }}
        />

        {/* Pre-clip gradient norm (raw) */}
        <Area
          type="monotone"
          dataKey="grad_norm_pre_clip"
          stroke={theme.chart.gradNorm}
          strokeWidth={1.5}
          fill="url(#gradNormGradient)"
          dot={false}
          name="grad_norm_pre_clip"
        />

        {/* Post-clip gradient norm */}
        <Line
          type="monotone"
          dataKey="grad_norm"
          stroke={theme.status.success}
          strokeWidth={1}
          dot={false}
          name="grad_norm"
          opacity={0.7}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}

export default memo(GradNormChart)
