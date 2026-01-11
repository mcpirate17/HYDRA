# HYDRA Training Dashboard

Real-time web dashboard for monitoring HYDRA LLM training.

## Quick Start

### 1. Start Backend
```bash
cd dashboard/backend
source /home/tim/venvs/llm/bin/activate
pip install -r requirements.txt  # First time only
uvicorn server:app --host 0.0.0.0 --port 8765
```

Or use the run script:
```bash
./dashboard/backend/run.sh
```

### 2. Start Frontend
```bash
cd dashboard/frontend
npm install  # First time only
npm run dev
```

### 3. Open Dashboard
Navigate to http://localhost:5173

## Features

### Core Metrics
- **Loss Chart**: Multi-scale EMA (short/medium/long) with spike highlighting
- **Throughput Gauge**: Tokens/sec with target comparison
- **VRAM Gauge**: GPU memory with color-coded zones (green/yellow/red)
- **Stats Cards**: Current step, best loss, elapsed time, total tokens

### Routing Health
- **MoD Heatmap**: Per-layer token selection fraction over time
- **MoR Depth Chart**: Recursion depth per layer with collapse detection

### Training Dynamics
- **Learning Rate Chart**: LR schedule visualization
- **Gradient Norm Chart**: Pre/post-clip gradient norms with threshold

### Sidebar
- Model info (size, dimension, blocks)
- Run config (LR, batch size, dataset)
- Database stats

### Real-time Updates
Toggle "Live" mode to enable WebSocket updates (polls every 5s).

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models` | GET | List all models |
| `/api/runs/{model_id}` | GET | List runs for model |
| `/api/stats/{model_id}` | GET | Model summary stats |
| `/api/metrics/{run_id}` | GET | Step metrics (paginated) |
| `/api/routing/mod/{run_id}` | GET | MoD routing data |
| `/api/routing/mor/{run_id}` | GET | MoR routing data |
| `/api/routing/moe/{run_id}` | GET | MoE metrics |
| `/api/run/{run_id}` | GET | Run details + config |
| `/api/milestones/{model_id}` | GET | Loss milestones |
| `/ws/{run_id}` | WebSocket | Real-time updates |

## Architecture

```
dashboard/
├── backend/
│   ├── server.py           # FastAPI + WebSocket endpoints
│   ├── requirements.txt    # Python dependencies
│   └── run.sh              # Launcher script
└── frontend/
    ├── src/
    │   ├── App.jsx         # Main app with layout
    │   ├── components/     # React components
    │   │   ├── Header.jsx
    │   │   ├── LossChart.jsx
    │   │   ├── ThroughputGauge.jsx
    │   │   ├── VRAMGauge.jsx
    │   │   ├── StatsCards.jsx
    │   │   ├── MoDHeatmap.jsx
    │   │   ├── MoRDepthChart.jsx
    │   │   ├── GradNormChart.jsx
    │   │   ├── LRChart.jsx
    │   │   └── RunSidebar.jsx
    │   ├── hooks/
    │   │   └── useTrainingData.js  # Data fetching hook
    │   └── styles/
    │       ├── theme.js    # Color theme
    │       └── index.css   # Global styles
    ├── package.json
    └── vite.config.js
```

## Data Source

The dashboard queries `checkpoints/training.db`, which is populated by the trainer during/after training runs. The database stores:

- Per-step metrics (loss, EMA, grad_norm, lr, tokens_per_sec, vram_gb)
- MoD routing stats per layer
- MoR routing stats per layer
- MoE metrics (if enabled)
- Run metadata and config

## Development

### Frontend
```bash
cd dashboard/frontend
npm run dev  # Hot reload on http://localhost:5173
```

### Backend
```bash
cd dashboard/backend
uvicorn server:app --reload --port 8765
```

### Proxy Configuration
The Vite dev server proxies `/api` and `/ws` to the backend (see `vite.config.js`).

## Color Theme

| Element | Color |
|---------|-------|
| Background | #0d1117 |
| Card background | #161b22 |
| Loss | #ef4444 (red) |
| EMA Short | #f97316 (orange) |
| EMA Medium | #eab308 (yellow) |
| EMA Long | #22c55e (green) |
| Gradient norm | #a855f7 (purple) |
| Learning rate | #3b82f6 (blue) |
