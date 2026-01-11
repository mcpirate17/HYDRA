// HYDRA Dashboard Theme - Dark mode colors

export const theme = {
  // Background colors
  bg: {
    primary: '#0d1117',
    secondary: '#161b22',
    tertiary: '#21262d',
    hover: '#30363d',
  },

  // Border colors
  border: {
    default: '#30363d',
    muted: '#21262d',
  },

  // Text colors
  text: {
    primary: '#c9d1d9',
    secondary: '#8b949e',
    muted: '#6e7681',
    link: '#58a6ff',
  },

  // Chart colors
  chart: {
    loss: '#ef4444',      // Red - main loss
    emaShort: '#f97316',  // Orange - short EMA
    emaMedium: '#eab308', // Yellow - medium EMA
    emaLong: '#22c55e',   // Green - long EMA
    gradNorm: '#a855f7',  // Purple - gradient norm
    lr: '#3b82f6',        // Blue - learning rate
    target: '#6b7280',    // Gray - reference lines
  },

  // Status colors
  status: {
    success: '#22c55e',
    warning: '#eab308',
    error: '#ef4444',
    info: '#3b82f6',
  },

  // Heatmap colors (for MoD capacity)
  heatmap: {
    low: '#ef4444',    // Red - too low
    target: '#22c55e', // Green - on target (~0.75)
    high: '#eab308',   // Yellow - too high
  },

  // VRAM gauge colors
  vram: {
    safe: '#22c55e',    // Green - < 24GB
    warning: '#eab308', // Yellow - 24-28GB
    danger: '#ef4444',  // Red - > 28GB
  },
}

// Chart configuration defaults
export const chartConfig = {
  margin: { top: 10, right: 30, left: 60, bottom: 30 },
  gridColor: '#21262d',
  axisColor: '#6e7681',
  fontSize: 11,
  fontFamily: 'JetBrains Mono, monospace',
}

export default theme
