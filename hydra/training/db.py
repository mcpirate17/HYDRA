"""
TrainingDB: SQLite-based persistent storage for training metrics.

Designed for batch training workflows:
- JSON logging during training (fast, append-only)
- Load to DB when training ends (queryable, cross-run analysis)

Supports multiple model types (500m, 626m, etc.) and tracks:
- Per-step metrics with multi-scale EMA
- Routing diagnostics (MoD, MoR, MoE)
- Run-level summaries
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Default DB location
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "checkpoints" / "training.db"


@dataclass
class StepRecord:
    """A single training step record."""
    step: int
    run_id: str
    model_id: str
    timestamp: str
    loss_total: float
    loss_ce: float
    loss_aux: float = 0.0
    loss_ponder: float = 0.0
    loss_advantage: float = 0.0
    lr: float = 0.0
    grad_norm: float = 0.0
    grad_norm_pre_clip: float = 0.0
    ema_short: float = 0.0
    ema_medium: float = 0.0
    ema_long: float = 0.0
    # Performance metrics (for dashboard)
    tokens_per_sec: float = 0.0
    vram_gb: float = 0.0
    batch_size: int = 0
    seq_len: int = 0


class TrainingDB:
    """
    SQLite database for training metrics with multi-model support.
    
    Usage:
        db = TrainingDB()
        
        # Load from existing JSON files
        db.load_diagnostics_json("checkpoints/diagnostics_500m_*.json")
        db.load_training_report("reports/training_report_*.json")
        
        # Query
        steps = db.get_steps(model_id="500m", start_step=100000, end_step=150000)
        ema = db.get_ema_series(model_id="500m")
    """
    
    # EMA alphas for multi-scale tracking
    EMA_ALPHA_SHORT = 0.99    # ~100 step window
    EMA_ALPHA_MEDIUM = 0.999  # ~1K step window
    EMA_ALPHA_LONG = 0.9999   # ~10K step window
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
    
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._conn() as conn:
            conn.executescript("""
                -- Models table (supports 500m, 626m, 1b, etc.)
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT,
                    params_millions REAL,
                    architecture_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Runs table (each training session)
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    start_step INTEGER,
                    end_step INTEGER,
                    start_time TEXT,
                    end_time TEXT,
                    total_tokens INTEGER,
                    best_loss REAL,
                    best_loss_step INTEGER,
                    config_json TEXT,
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                );
                
                -- Per-step metrics (main table for analysis)
                CREATE TABLE IF NOT EXISTS steps (
                    step INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    timestamp TEXT,
                    loss_total REAL,
                    loss_ce REAL,
                    loss_aux REAL,
                    loss_ponder REAL,
                    loss_advantage REAL,
                    lr REAL,
                    grad_norm REAL,
                    grad_norm_pre_clip REAL,
                    ema_short REAL,
                    ema_medium REAL,
                    ema_long REAL,
                    tokens_per_sec REAL,
                    vram_gb REAL,
                    batch_size INTEGER,
                    seq_len INTEGER,
                    PRIMARY KEY (model_id, step),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                
                -- MoD routing per layer
                CREATE TABLE IF NOT EXISTS routing_mod (
                    step INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    layer INTEGER NOT NULL,
                    selected_frac REAL,
                    compute_savings_pct REAL,
                    probs_mean REAL,
                    probs_std REAL,
                    status TEXT,
                    PRIMARY KEY (model_id, step, layer)
                );
                
                -- MoR routing per layer
                CREATE TABLE IF NOT EXISTS routing_mor (
                    step INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    layer INTEGER NOT NULL,
                    avg_depth REAL,
                    expected_depth REAL,
                    router_probs_mean REAL,
                    status TEXT,
                    PRIMARY KEY (model_id, step, layer)
                );
                
                -- MoE routing (aggregated per step)
                CREATE TABLE IF NOT EXISTS routing_moe (
                    step INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    entropy REAL,
                    divergence REAL,
                    util_expert_0 REAL,
                    util_expert_1 REAL,
                    util_expert_2 REAL,
                    util_expert_3 REAL,
                    PRIMARY KEY (model_id, step)
                );
                
                -- Adaptive LR state
                CREATE TABLE IF NOT EXISTS adaptive_lr (
                    step INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    loss_ema_short REAL,
                    loss_ema_long REAL,
                    patience_counter INTEGER,
                    cooldown_triggered INTEGER,
                    PRIMARY KEY (model_id, step)
                );
                
                -- Indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_steps_model_step ON steps(model_id, step);
                CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id);
                CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_id);
            """)
        # Run migrations for existing databases
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Add new columns to existing tables (safe for fresh DBs)."""
        new_columns = [
            ("steps", "tokens_per_sec", "REAL DEFAULT 0"),
            ("steps", "vram_gb", "REAL DEFAULT 0"),
            ("steps", "batch_size", "INTEGER DEFAULT 0"),
            ("steps", "seq_len", "INTEGER DEFAULT 0"),
        ]
        with self._conn() as conn:
            for table, column, col_type in new_columns:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                except sqlite3.OperationalError:
                    # Column already exists, ignore
                    pass

    def ensure_model(self, model_id: str, name: str = None, params_millions: float = None) -> None:
        """Ensure a model exists in the database."""
        with self._conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO models (model_id, name, params_millions)
                VALUES (?, ?, ?)
            """, (model_id, name or model_id, params_millions))
    
    def insert_run(
        self,
        run_id: str,
        model_id: str,
        start_step: int = 0,
        end_step: int = 0,
        start_time: str = None,
        end_time: str = None,
        total_tokens: int = 0,
        best_loss: float = None,
        best_loss_step: int = None,
        config: dict = None,
    ) -> None:
        """Insert or update a training run."""
        self.ensure_model(model_id)
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO runs 
                (run_id, model_id, start_step, end_step, start_time, end_time, 
                 total_tokens, best_loss, best_loss_step, config_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, model_id, start_step, end_step, start_time, end_time,
                total_tokens, best_loss, best_loss_step,
                json.dumps(config) if config else None
            ))
    
    def insert_step(self, record: StepRecord) -> None:
        """Insert a single step record."""
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO steps
                (step, run_id, model_id, timestamp, loss_total, loss_ce, loss_aux,
                 loss_ponder, loss_advantage, lr, grad_norm, grad_norm_pre_clip,
                 ema_short, ema_medium, ema_long, tokens_per_sec, vram_gb,
                 batch_size, seq_len)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.step, record.run_id, record.model_id, record.timestamp,
                record.loss_total, record.loss_ce, record.loss_aux,
                record.loss_ponder, record.loss_advantage,
                record.lr, record.grad_norm, record.grad_norm_pre_clip,
                record.ema_short, record.ema_medium, record.ema_long,
                record.tokens_per_sec, record.vram_gb,
                record.batch_size, record.seq_len
            ))
    
    def insert_steps_batch(self, records: List[StepRecord]) -> None:
        """Batch insert step records (much faster for backfill)."""
        if not records:
            return
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO steps
                (step, run_id, model_id, timestamp, loss_total, loss_ce, loss_aux,
                 loss_ponder, loss_advantage, lr, grad_norm, grad_norm_pre_clip,
                 ema_short, ema_medium, ema_long, tokens_per_sec, vram_gb,
                 batch_size, seq_len)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (r.step, r.run_id, r.model_id, r.timestamp,
                 r.loss_total, r.loss_ce, r.loss_aux,
                 r.loss_ponder, r.loss_advantage,
                 r.lr, r.grad_norm, r.grad_norm_pre_clip,
                 r.ema_short, r.ema_medium, r.ema_long,
                 r.tokens_per_sec, r.vram_gb,
                 r.batch_size, r.seq_len)
                for r in records
            ])
    
    def insert_routing_mod_batch(self, records: List[Dict]) -> None:
        """Batch insert MoD routing records."""
        if not records:
            return
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO routing_mod
                (step, run_id, model_id, layer, selected_frac, compute_savings_pct,
                 probs_mean, probs_std, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
    
    def insert_routing_mor_batch(self, records: List[Dict]) -> None:
        """Batch insert MoR routing records."""
        if not records:
            return
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO routing_mor
                (step, run_id, model_id, layer, avg_depth, expected_depth,
                 router_probs_mean, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
    
    def insert_routing_moe_batch(self, records: List[Dict]) -> None:
        """Batch insert MoE routing records."""
        if not records:
            return
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO routing_moe
                (step, run_id, model_id, entropy, divergence,
                 util_expert_0, util_expert_1, util_expert_2, util_expert_3)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
    
    def insert_adaptive_lr_batch(self, records: List[Dict]) -> None:
        """Batch insert adaptive LR records."""
        if not records:
            return
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO adaptive_lr
                (step, run_id, model_id, loss_ema_short, loss_ema_long,
                 patience_counter, cooldown_triggered)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, records)
    
    # =========================================================================
    # QUERY METHODS
    # =========================================================================
    
    def get_steps(
        self,
        model_id: str,
        start_step: int = 0,
        end_step: int = None,
        columns: List[str] = None,
    ) -> List[Dict]:
        """Get step records for a model within a step range."""
        cols = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols} FROM steps WHERE model_id = ? AND step >= ?"
        params = [model_id, start_step]
        
        if end_step is not None:
            query += " AND step <= ?"
            params.append(end_step)
        
        query += " ORDER BY step"
        
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
    
    def get_ema_series(
        self,
        model_id: str,
        start_step: int = 0,
        end_step: int = None,
    ) -> Dict[str, List[Tuple[int, float]]]:
        """Get multi-scale EMA series for plotting."""
        steps = self.get_steps(
            model_id, start_step, end_step,
            columns=["step", "loss_total", "ema_short", "ema_medium", "ema_long"]
        )
        return {
            "raw": [(s["step"], s["loss_total"]) for s in steps],
            "short": [(s["step"], s["ema_short"]) for s in steps],
            "medium": [(s["step"], s["ema_medium"]) for s in steps],
            "long": [(s["step"], s["ema_long"]) for s in steps],
        }
    
    def get_loss_milestones(self, model_id: str, milestone_interval: int = 10000) -> Dict[int, float]:
        """Get best loss at each milestone (e.g., every 10K steps)."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT (step / ?) * ? as milestone, MIN(loss_total) as best_loss
                FROM steps
                WHERE model_id = ?
                GROUP BY milestone
                ORDER BY milestone
            """, (milestone_interval, milestone_interval, model_id)).fetchall()
            return {r["milestone"]: r["best_loss"] for r in rows}
    
    def get_runs(self, model_id: str = None) -> List[Dict]:
        """Get all runs, optionally filtered by model."""
        query = "SELECT * FROM runs"
        params = []
        if model_id:
            query += " WHERE model_id = ?"
            params.append(model_id)
        query += " ORDER BY start_time"
        
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
    
    def get_latest_step(self, model_id: str) -> Optional[int]:
        """Get the latest step for a model (for resume continuity)."""
        with self._conn() as conn:
            row = conn.execute("""
                SELECT MAX(step) as max_step FROM steps WHERE model_id = ?
            """, (model_id,)).fetchone()
            return row["max_step"] if row else None
    
    def get_latest_ema(self, model_id: str) -> Tuple[float, float, float]:
        """Get latest EMA values for resume (short, medium, long)."""
        with self._conn() as conn:
            row = conn.execute("""
                SELECT ema_short, ema_medium, ema_long 
                FROM steps 
                WHERE model_id = ? 
                ORDER BY step DESC LIMIT 1
            """, (model_id,)).fetchone()
            if row:
                return (row["ema_short"], row["ema_medium"], row["ema_long"])
            return (0.0, 0.0, 0.0)
    
    def get_step_count(self, model_id: str = None) -> int:
        """Get total number of step records."""
        query = "SELECT COUNT(*) as cnt FROM steps"
        params = []
        if model_id:
            query += " WHERE model_id = ?"
            params.append(model_id)
        
        with self._conn() as conn:
            row = conn.execute(query, params).fetchone()
            return row["cnt"]
    
    def get_model_stats(self, model_id: str) -> Dict:
        """Get summary stats for a model."""
        with self._conn() as conn:
            row = conn.execute("""
                SELECT 
                    COUNT(*) as step_count,
                    MIN(step) as min_step,
                    MAX(step) as max_step,
                    MIN(loss_total) as best_loss,
                    AVG(loss_total) as avg_loss,
                    (SELECT COUNT(*) FROM runs WHERE model_id = ?) as run_count
                FROM steps
                WHERE model_id = ?
            """, (model_id, model_id)).fetchone()
            return dict(row) if row else {}
    
    # =========================================================================
    # BACKFILL FROM JSON
    # =========================================================================
    
    def load_diagnostics_json(
        self,
        json_path: Path,
        model_id: str,
        run_id: str = None,
    ) -> int:
        """
        Load a diagnostics JSON file into the database.
        
        Returns number of records loaded.
        """
        if not json_path.exists():
            return 0
        
        with open(json_path) as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return 0
        
        # Extract run_id from filename if not provided
        if run_id is None:
            # diagnostics_500m_20260109_144733.json -> 500m_20260109_144733
            name = json_path.stem
            if name.startswith("diagnostics_"):
                run_id = name.replace("diagnostics_", "")
            else:
                run_id = name
        
        self.ensure_model(model_id)
        
        # Compute EMAs from scratch for this file
        ema_short, ema_medium, ema_long = self.get_latest_ema(model_id)
        
        step_records = []
        mod_records = []
        mor_records = []
        moe_records = []
        alr_records = []
        
        for record in data:
            if not isinstance(record, dict):
                continue
            
            step = record.get("step", 0)
            timestamp = record.get("timestamp", "")
            losses = record.get("losses", {})
            
            loss_total = losses.get("total", 0.0)
            loss_ce = losses.get("ce", 0.0)
            loss_aux = losses.get("aux", 0.0)
            loss_ponder = losses.get("ponder", 0.0)
            loss_advantage = losses.get("advantage", 0.0)
            
            # Update EMAs
            if ema_short == 0:
                ema_short = loss_total
            else:
                ema_short = self.EMA_ALPHA_SHORT * ema_short + (1 - self.EMA_ALPHA_SHORT) * loss_total
            
            if ema_medium == 0:
                ema_medium = loss_total
            else:
                ema_medium = self.EMA_ALPHA_MEDIUM * ema_medium + (1 - self.EMA_ALPHA_MEDIUM) * loss_total
            
            if ema_long == 0:
                ema_long = loss_total
            else:
                ema_long = self.EMA_ALPHA_LONG * ema_long + (1 - self.EMA_ALPHA_LONG) * loss_total
            
            step_records.append(StepRecord(
                step=step,
                run_id=run_id,
                model_id=model_id,
                timestamp=timestamp,
                loss_total=loss_total,
                loss_ce=loss_ce,
                loss_aux=loss_aux,
                loss_ponder=loss_ponder,
                loss_advantage=loss_advantage,
                lr=record.get("lr", 0.0),
                grad_norm=record.get("grad_norm", 0.0),
                grad_norm_pre_clip=record.get("grad_norm_pre_clip", 0.0),
                ema_short=ema_short,
                ema_medium=ema_medium,
                ema_long=ema_long,
                tokens_per_sec=record.get("tokens_per_sec", 0.0),
                vram_gb=record.get("vram_gb", 0.0),
                batch_size=record.get("batch_size", 0),
                seq_len=record.get("seq_len", 0),
            ))
            
            # MoD routing
            for layer_data in record.get("mod_layers", []):
                mod_records.append((
                    step, run_id, model_id,
                    layer_data.get("layer", 0),
                    layer_data.get("selected_frac", 0.0),
                    layer_data.get("compute_savings_pct", 0.0),
                    layer_data.get("probs_mean", 0.0),
                    layer_data.get("probs_std", 0.0),
                    layer_data.get("status", ""),
                ))
            
            # MoR routing
            for layer_data in record.get("mor_layers", []):
                mor_records.append((
                    step, run_id, model_id,
                    layer_data.get("layer", 0),
                    layer_data.get("avg_depth", 0.0),
                    layer_data.get("expected_depth", 0.0),
                    layer_data.get("router_probs_mean", 0.0),
                    layer_data.get("status", ""),
                ))
            
            # MoE routing
            moe = record.get("moe", {})
            if moe:
                util = moe.get("utilization_pct", [0, 0, 0, 0])
                while len(util) < 4:
                    util.append(0.0)
                moe_records.append((
                    step, run_id, model_id,
                    moe.get("entropy", 0.0),
                    moe.get("divergence", 0.0),
                    util[0], util[1], util[2], util[3],
                ))
            
            # Adaptive LR
            alr = record.get("adaptive_lr", {})
            if alr:
                alr_records.append((
                    step, run_id, model_id,
                    alr.get("loss_ema_short", 0.0),
                    alr.get("loss_ema_long", 0.0),
                    alr.get("patience_counter", 0),
                    1 if alr.get("cooldown_triggered") else 0,
                ))
        
        # Batch insert all records
        self.insert_steps_batch(step_records)
        self.insert_routing_mod_batch(mod_records)
        self.insert_routing_mor_batch(mor_records)
        self.insert_routing_moe_batch(moe_records)
        self.insert_adaptive_lr_batch(alr_records)
        
        return len(step_records)
    
    def load_training_report(self, json_path: Path, model_id: str = None) -> bool:
        """
        Load a training report JSON into the runs table.
        
        Returns True if loaded successfully.
        """
        if not json_path.exists():
            return False
        
        with open(json_path) as f:
            data = json.load(f)
        
        config = data.get("configuration", {})
        summary = data.get("training_summary", {})
        loss = data.get("loss_analysis", {})
        meta = data.get("metadata", {})
        
        run_id = config.get("run_id", json_path.stem)
        
        # Infer model_id from run_id if not provided
        if model_id is None:
            if run_id.startswith("500m_"):
                model_id = "500m"
            elif run_id.startswith("626m_"):
                model_id = "626m"
            elif run_id.startswith("100m_"):
                model_id = "100m"
            else:
                model_id = run_id.split("_")[0]
        
        self.insert_run(
            run_id=run_id,
            model_id=model_id,
            start_step=config.get("start_step", 0),
            end_step=summary.get("total_steps", 0),
            start_time=meta.get("timestamp"),
            end_time=None,  # Not stored in reports currently
            total_tokens=summary.get("total_tokens", 0),
            best_loss=loss.get("best_loss"),
            best_loss_step=loss.get("best_loss_step"),
            config=config,
        )
        
        return True
