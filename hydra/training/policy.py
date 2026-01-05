"""
SeqLenPolicy: owns HYDRA_SEQ_LEN_* env vars, JSON parsing, and policy matching.

The trainer uses this module to get the appropriate config patch for a given sequence length.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Any


def _load_from_env() -> Optional[Dict[str, Any]]:
    """Load seq-len policy JSON from HYDRA_SEQ_LEN_POLICY_JSON environment variable."""
    path = os.environ.get("HYDRA_SEQ_LEN_POLICY_JSON", "").strip()
    if not path:
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        warnings.warn(f"Failed to load HYDRA_SEQ_LEN_POLICY_JSON='{path}': {e}")
        return None


def _match_by_seq_len(policy: Dict[str, Any], seq_len: int) -> Dict[str, Any]:
    """
    Return the config patch for a specific sequence length.
    
    Looks for exact match in by_seq_len, then falls back to default.
    """
    by_seq = policy.get("by_seq_len", {}) if isinstance(policy, dict) else {}
    if isinstance(by_seq, dict):
        exact = by_seq.get(str(seq_len))
        if isinstance(exact, dict):
            return exact
    default = policy.get("default", {}) if isinstance(policy, dict) else {}
    return default if isinstance(default, dict) else {}


def _auto_heuristic(seq_len: int) -> Dict[str, Any]:
    """
    Built-in heuristic policy keyed on current sequence length.

    This is intentionally conservative (safe) and only touches runtime knobs.
    It does NOT attempt to change architecture (e.g. LA3 vs CCQA) mid-run.
    """
    L = int(seq_len)

    # Short context: prioritize throughput (checkpointing is overhead; chunked CE can be slower).
    if L <= 512:
        return {
            "use_chunked_ce": False,
            "gradient_checkpointing": True,
            "checkpoint_every_n": 4,
        }

    # Mid context: balanced
    if L <= 1024:
        return {
            "use_chunked_ce": True,
            "gradient_checkpointing": True,
            "checkpoint_every_n": 2,
        }

    # Long context: prioritize memory
    return {
        "use_chunked_ce": True,
        "gradient_checkpointing": True,
        "checkpoint_every_n": 1,
    }


@dataclass
class SeqLenPolicy:
    """
    Manages sequence-length-dependent configuration patches.

    Loads policy from HYDRA_SEQ_LEN_POLICY_JSON or uses built-in auto-tune
    heuristics when HYDRA_SEQ_LEN_AUTO_TUNE=1.

    Usage:
        policy = SeqLenPolicy.from_env()
        patch = policy.get_patch(seq_len=2048)
        if patch:
            apply_patch_to_config(patch)
    """

    raw_policy: Optional[Dict[str, Any]] = field(default=None, repr=False)
    auto_tune: bool = False

    @classmethod
    def from_env(cls) -> "SeqLenPolicy":
        """Create SeqLenPolicy from environment variables."""
        raw = _load_from_env()
        auto_tune = False

        if raw is None:
            # Check for auto-tune mode
            env_auto = os.environ.get("HYDRA_SEQ_LEN_AUTO_TUNE", "").strip().lower()
            if env_auto in ("1", "true", "yes"):
                raw = {"version": 1, "default": {}, "by_seq_len": {}}
                auto_tune = True

        return cls(raw_policy=raw, auto_tune=auto_tune)

    def get_patch(self, seq_len: int) -> Dict[str, Any]:
        """
        Get the configuration patch for a given sequence length.

        Returns empty dict if no policy is configured or no match found.
        """
        if self.raw_policy is None:
            return {}

        patch = _match_by_seq_len(self.raw_policy, int(seq_len))
        if not patch and self.auto_tune:
            patch = _auto_heuristic(int(seq_len))
        return patch

    @property
    def is_active(self) -> bool:
        """Return True if a policy is configured."""
        return self.raw_policy is not None
