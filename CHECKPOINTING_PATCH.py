# HYDRA Logging Architecture Fix
# 
# This patch modifies checkpointing.py to save diagnostics to diagnostics/output/
# instead of mixing them with checkpoint .pt files.
#
# Apply manually by editing hydra/training/checkpointing.py

# STEP 1: Add this new function BEFORE load_diagnostics():

def _get_diagnostics_dir(checkpoint_dir: str) -> Path:
    """Get the diagnostics output directory.
    
    Diagnostics are stored in diagnostics/output/ (sibling to checkpoint_dir parent).
    This keeps them separate from large .pt checkpoint files.
    """
    # checkpoint_dir is typically "checkpoints", so diagnostics go to "diagnostics/output"
    ckpt_path = Path(checkpoint_dir)
    project_root = ckpt_path.parent if ckpt_path.name == "checkpoints" else ckpt_path.parent
    return project_root / "diagnostics" / "output"


# STEP 2: Replace load_diagnostics() with:

def load_diagnostics(
    *,
    checkpoint_dir: str,
    run_id: Optional[str] = None,
    logger=None,
) -> List[dict]:
    """Load existing diagnostics from a run-specific JSON file for resuming.
    
    Returns empty list if file doesn't exist or can't be loaded.
    Checks both new location (diagnostics/output/) and legacy location (checkpoints/).
    """
    diag_dir = _get_diagnostics_dir(checkpoint_dir)
    legacy_dir = Path(checkpoint_dir)
    
    if run_id:
        filename = f"diagnostics_{run_id}.json"
    else:
        filename = "training_diagnostics.json"
    
    # Try new location first, then legacy
    diag_path = diag_dir / filename
    if not diag_path.exists():
        diag_path = legacy_dir / filename
    
    if not diag_path.exists():
        return []
    
    try:
        with open(diag_path, "r") as f:
            data = json.load(f)
        if logger:
            logger.info(f"📊 Loaded {len(data)} existing diagnostic records from {diag_path}")
        return data if isinstance(data, list) else []
    except Exception as e:
        if logger:
            logger.warning(f"⚠️  Failed to load existing diagnostics: {e}")
        return []


# STEP 3: Replace save_diagnostics() with:

def save_diagnostics(
    *,
    checkpoint_dir: str,
    diagnostics_data: list[dict],
    logger,
    run_id: Optional[str] = None,
) -> None:
    """Save diagnostics to a run-specific JSON file in diagnostics/output/.
    
    Files are named: diagnostics_{run_id}.json
    If run_id is None, falls back to generic training_diagnostics.json
    """
    if not diagnostics_data:
        return
    
    diag_dir = _get_diagnostics_dir(checkpoint_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    # Use run-specific filename if run_id provided
    if run_id:
        diag_path = diag_dir / f"diagnostics_{run_id}.json"
    else:
        diag_path = diag_dir / "training_diagnostics.json"
    
    try:
        with open(diag_path, "w") as f:
            json.dump(diagnostics_data, f, indent=2)
        logger.debug(f"📊 Diagnostics saved to {diag_path}")
    except Exception as e:
        logger.warning(f"⚠️  Failed to save diagnostics: {e}")
