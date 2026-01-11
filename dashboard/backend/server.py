#!/usr/bin/env python3
"""
HYDRA Training Dashboard - FastAPI Backend

Provides REST API and WebSocket endpoints for the training dashboard.
Queries the training.db SQLite database via TrainingDB.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add HYDRA project to path for TrainingDB import
HYDRA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HYDRA_ROOT))

from hydra.training.db import TrainingDB

# Initialize FastAPI app
app = FastAPI(
    title="HYDRA Training Dashboard API",
    description="Real-time training metrics for HYDRA LLM",
    version="1.0.0",
)

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
DB_PATH = HYDRA_ROOT / "checkpoints" / "training.db"


def get_db() -> TrainingDB:
    """Get TrainingDB instance."""
    return TrainingDB(DB_PATH)


# =============================================================================
# Pydantic Models
# =============================================================================

class ModelInfo(BaseModel):
    model_id: str
    name: Optional[str]
    params_millions: Optional[float]
    created_at: Optional[str]


class RunInfo(BaseModel):
    run_id: str
    model_id: str
    start_step: Optional[int]
    end_step: Optional[int]
    start_time: Optional[str]
    end_time: Optional[str]
    best_loss: Optional[float]
    best_loss_step: Optional[int]
    total_tokens: Optional[int]


class ModelStats(BaseModel):
    model_id: str
    step_count: int
    min_step: Optional[int]
    max_step: Optional[int]
    best_loss: Optional[float]
    avg_loss: Optional[float]
    run_count: int


# =============================================================================
# REST Endpoints
# =============================================================================

@app.get("/api/models", response_model=List[ModelInfo])
def get_models():
    """List all models in the database."""
    db = get_db()
    with db._conn() as conn:
        rows = conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


@app.get("/api/runs/{model_id}", response_model=List[RunInfo])
def get_runs(model_id: str):
    """Get all runs for a model."""
    db = get_db()
    runs = db.get_runs(model_id)
    return runs


@app.get("/api/stats/{model_id}", response_model=ModelStats)
def get_stats(model_id: str):
    """Get summary statistics for a model."""
    db = get_db()
    stats = db.get_model_stats(model_id)
    stats["model_id"] = model_id
    return stats


@app.get("/api/metrics/{run_id}")
def get_metrics(
    run_id: str,
    start_step: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=10000),
):
    """
    Get step metrics for a run (paginated).

    Returns steps data with all metrics including loss, EMA, grad_norm, etc.
    """
    db = get_db()

    # Get model_id from run_id prefix
    model_id = run_id.split("_")[0] if "_" in run_id else run_id

    with db._conn() as conn:
        rows = conn.execute("""
            SELECT * FROM steps
            WHERE run_id = ? AND step >= ?
            ORDER BY step
            LIMIT ?
        """, (run_id, start_step, limit)).fetchall()

        steps = [dict(r) for r in rows]

        # Get max step for this run
        max_row = conn.execute("""
            SELECT MAX(step) as max_step FROM steps WHERE run_id = ?
        """, (run_id,)).fetchone()

        return {
            "run_id": run_id,
            "model_id": model_id,
            "steps": steps,
            "count": len(steps),
            "max_step": max_row["max_step"] if max_row else 0,
        }


@app.get("/api/ema/{model_id}")
def get_ema_series(
    model_id: str,
    start_step: int = Query(0, ge=0),
    end_step: Optional[int] = None,
):
    """Get multi-scale EMA series for plotting."""
    db = get_db()
    ema = db.get_ema_series(model_id, start_step, end_step)
    return ema


@app.get("/api/routing/mod/{run_id}")
def get_routing_mod(
    run_id: str,
    start_step: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
):
    """
    Get MoD routing data for heatmap visualization.

    Returns per-layer selected_frac and compute_savings.
    """
    db = get_db()

    with db._conn() as conn:
        rows = conn.execute("""
            SELECT step, layer, selected_frac, compute_savings_pct, probs_mean, probs_std, status
            FROM routing_mod
            WHERE run_id = ? AND step >= ?
            ORDER BY step, layer
            LIMIT ?
        """, (run_id, start_step, limit * 20)).fetchall()  # Extra for multi-layer

        return {
            "run_id": run_id,
            "data": [dict(r) for r in rows],
        }


@app.get("/api/routing/mor/{run_id}")
def get_routing_mor(
    run_id: str,
    start_step: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
):
    """
    Get MoR routing data for depth visualization.

    Returns per-layer avg_depth and expected_depth.
    """
    db = get_db()

    with db._conn() as conn:
        rows = conn.execute("""
            SELECT step, layer, avg_depth, expected_depth, router_probs_mean, status
            FROM routing_mor
            WHERE run_id = ? AND step >= ?
            ORDER BY step, layer
            LIMIT ?
        """, (run_id, start_step, limit * 20)).fetchall()

        return {
            "run_id": run_id,
            "data": [dict(r) for r in rows],
        }


@app.get("/api/routing/moe/{run_id}")
def get_routing_moe(
    run_id: str,
    start_step: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
):
    """Get MoE routing metrics."""
    db = get_db()

    with db._conn() as conn:
        rows = conn.execute("""
            SELECT step, entropy, divergence,
                   util_expert_0, util_expert_1, util_expert_2, util_expert_3
            FROM routing_moe
            WHERE run_id = ? AND step >= ?
            ORDER BY step
            LIMIT ?
        """, (run_id, start_step, limit)).fetchall()

        return {
            "run_id": run_id,
            "data": [dict(r) for r in rows],
        }


@app.get("/api/adaptive_lr/{run_id}")
def get_adaptive_lr(
    run_id: str,
    start_step: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
):
    """Get adaptive learning rate state history."""
    db = get_db()

    with db._conn() as conn:
        rows = conn.execute("""
            SELECT step, loss_ema_short, loss_ema_long, patience_counter, cooldown_triggered
            FROM adaptive_lr
            WHERE run_id = ? AND step >= ?
            ORDER BY step
            LIMIT ?
        """, (run_id, start_step, limit)).fetchall()

        return {
            "run_id": run_id,
            "data": [dict(r) for r in rows],
        }


@app.get("/api/run/{run_id}")
def get_run_details(run_id: str):
    """Get detailed info for a specific run including config."""
    db = get_db()

    with db._conn() as conn:
        row = conn.execute("""
            SELECT * FROM runs WHERE run_id = ?
        """, (run_id,)).fetchone()

        if not row:
            return {"error": "Run not found"}

        result = dict(row)

        # Parse config_json if present
        if result.get("config_json"):
            try:
                result["config"] = json.loads(result["config_json"])
            except json.JSONDecodeError:
                result["config"] = {}

        return result


@app.get("/api/milestones/{model_id}")
def get_milestones(
    model_id: str,
    interval: int = Query(10000, ge=1000),
):
    """Get loss milestones for a model."""
    db = get_db()
    milestones = db.get_loss_milestones(model_id, interval)
    return {"model_id": model_id, "milestones": milestones}


# =============================================================================
# WebSocket for Real-time Updates
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, run_id: str):
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)

    def disconnect(self, websocket: WebSocket, run_id: str):
        if run_id in self.active_connections:
            self.active_connections[run_id].remove(websocket)
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]

    async def broadcast(self, run_id: str, message: dict):
        if run_id in self.active_connections:
            for connection in self.active_connections[run_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


manager = ConnectionManager()


@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint for real-time training updates.

    Polls the database every 5 seconds and pushes new steps to connected clients.
    """
    await manager.connect(websocket, run_id)

    db = get_db()
    last_step = 0

    # Get initial last step
    with db._conn() as conn:
        row = conn.execute("""
            SELECT MAX(step) as max_step FROM steps WHERE run_id = ?
        """, (run_id,)).fetchone()
        if row and row["max_step"]:
            last_step = row["max_step"]

    try:
        while True:
            # Poll for new data every 5 seconds
            await asyncio.sleep(5)

            with db._conn() as conn:
                # Check for new steps
                rows = conn.execute("""
                    SELECT * FROM steps
                    WHERE run_id = ? AND step > ?
                    ORDER BY step
                    LIMIT 100
                """, (run_id, last_step)).fetchall()

                if rows:
                    steps = [dict(r) for r in rows]
                    last_step = steps[-1]["step"]

                    # Get routing data for new steps
                    mod_rows = conn.execute("""
                        SELECT step, layer, selected_frac, compute_savings_pct
                        FROM routing_mod
                        WHERE run_id = ? AND step > ?
                        ORDER BY step, layer
                    """, (run_id, last_step - len(steps) * 100)).fetchall()

                    mor_rows = conn.execute("""
                        SELECT step, layer, avg_depth, expected_depth
                        FROM routing_mor
                        WHERE run_id = ? AND step > ?
                        ORDER BY step, layer
                    """, (run_id, last_step - len(steps) * 100)).fetchall()

                    await websocket.send_json({
                        "type": "update",
                        "steps": steps,
                        "routing_mod": [dict(r) for r in mod_rows],
                        "routing_mor": [dict(r) for r in mor_rows],
                        "last_step": last_step,
                    })
                else:
                    # Send heartbeat
                    await websocket.send_json({
                        "type": "heartbeat",
                        "last_step": last_step,
                    })

    except WebSocketDisconnect:
        manager.disconnect(websocket, run_id)
    except Exception:
        manager.disconnect(websocket, run_id)


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint."""
    db = get_db()
    try:
        count = db.get_step_count()
        return {"status": "healthy", "total_steps": count}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
