"""
CurriculumController: Owns MoR/MoD curriculum gating state and decision logic.

The trainer queries this controller each step to determine if MoR/MoD should be
triggered or modified. The controller does not mutate the model directly - it
returns action commands for the trainer to apply.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any


@dataclass
class CurriculumConfig:
    """Configuration for MoR/MoD curriculum, typically extracted from TrainingConfig."""
    # MoR settings
    mor_adaptive: bool = False
    mor_enable_pct: float = 0.0
    mor_enable_min_steps: int = 3000
    mor_enable_loss_threshold: float = 0.0
    mor_rampup_steps: int = 1000
    mor_already_enabled: bool = False
    
    # MoD settings
    mod_capacity: float = 1.0  # >= 1.0 means MoD is OFF
    mod_enable_min_step: int = 3000
    mod_enable_mor_early_exit_threshold: float = 0.38
    mod_enable_loss_threshold: float = 4.5
    
    max_steps: int = 10000


@dataclass
class CurriculumAction:
    """An action the trainer should apply to the model."""
    action_type: str  # "trigger_mor_early", "trigger_mod_from_mor", "update_mod_loss_ema"
    step: int
    params: dict = field(default_factory=dict)
    log_message: Optional[str] = None


class CurriculumController:
    """
    Manages MoR/MoD curriculum gating during training.
    
    Key responsibilities:
    - Track when MoR routing should be enabled (step threshold OR loss threshold)
    - Track when MoD should be enabled (based on MoR early-exit ratio)
    - Return actions for trainer to apply to model
    
    Usage:
        controller = CurriculumController.from_training_config(config, start_step)
        
        # In training loop:
        actions = controller.step(
            step=step,
            ce_ema=ce_ema,
            routing_stats=model.get_routing_stats() if step % 100 == 0 else None
        )
        for action in actions:
            apply_action_to_model(model, action)
    """
    
    def __init__(
        self,
        config: CurriculumConfig,
        start_step: int = 0,
    ):
        self.config = config
        self.start_step = start_step
        
        # Computed values
        self.mor_enable_step: int = 0
        self.mor_rampup_steps: int = 0
        
        # Curriculum state
        self._mor_triggered_by_loss: bool = False
        self._mod_triggered: bool = False
        
        # Compute MoR enable step
        if config.mor_adaptive:
            mor_enable_min = config.mor_enable_min_steps
            self.mor_enable_step = max(
                mor_enable_min,
                int(config.max_steps * config.mor_enable_pct)
            )
            if config.mor_already_enabled:
                self.mor_enable_step = 0
            
            # Compute rampup steps
            remaining_steps = config.max_steps - self.mor_enable_step
            default_rampup = min(config.mor_rampup_steps, 2 * self.mor_enable_step)
            self.mor_rampup_steps = min(default_rampup, remaining_steps)
            self.mor_rampup_steps = max(self.mor_rampup_steps, min(100, int(config.max_steps * 0.1)))

    @classmethod
    def from_training_config(cls, config: Any, start_step: int = 0) -> "CurriculumController":
        """Create CurriculumController from a TrainingConfig object."""
        curriculum_config = CurriculumConfig(
            mor_adaptive=getattr(config, "mor_adaptive", False),
            mor_enable_pct=getattr(config, "mor_enable_pct", 0.0),
            mor_enable_min_steps=getattr(config, "mor_enable_min_steps", 3000),
            mor_enable_loss_threshold=getattr(config, "mor_enable_loss_threshold", 0.0) or 0.0,
            mor_rampup_steps=getattr(config, "mor_rampup_steps", 1000),
            mor_already_enabled=getattr(config, "mor_already_enabled", False),
            mod_capacity=getattr(config, "mod_capacity", 1.0),
            mod_enable_min_step=getattr(config, "mod_enable_min_step", 3000),
            mod_enable_mor_early_exit_threshold=getattr(config, "mod_enable_mor_early_exit_threshold", 0.38),
            mod_enable_loss_threshold=getattr(config, "mod_enable_loss_threshold", 4.5),
            max_steps=getattr(config, "max_steps", 10000),
        )
        return cls(curriculum_config, start_step)

    def step(
        self,
        step: int,
        ce_ema: float,
        routing_stats: Optional[dict] = None,
    ) -> List[CurriculumAction]:
        """
        Check curriculum conditions and return any actions to apply.
        
        Args:
            step: Current training step
            ce_ema: Current CE EMA loss value
            routing_stats: Optional model routing stats (pass every ~100 steps for MoD check)
        
        Returns:
            List of CurriculumAction objects to apply to the model
        """
        actions: List[CurriculumAction] = []
        config = self.config
        
        # Always request MoD loss EMA update (model uses it for its own decisions)
        actions.append(CurriculumAction(
            action_type="update_mod_loss_ema",
            step=step,
            params={"ce_ema": ce_ema},
        ))
        
        # Check for early MoR trigger by loss threshold
        if not self._mor_triggered_by_loss and config.mor_adaptive:
            mor_loss_thr = config.mor_enable_loss_threshold
            mor_min_steps = config.mor_enable_min_steps
            
            # Loss threshold can only trigger AFTER mor_enable_min_steps (hard floor)
            if (mor_loss_thr > 0 and ce_ema > 0 and ce_ema < mor_loss_thr 
                and step >= mor_min_steps and step < self.mor_enable_step):
                
                rampup = min(1000, max(100, int(config.max_steps * 0.05)))
                actions.append(CurriculumAction(
                    action_type="trigger_mor_early",
                    step=step,
                    params={"rampup_steps": rampup},
                    log_message=(
                        f"🚀 MoR EARLY TRIGGER: CE_EMA={ce_ema:.3f} < threshold={mor_loss_thr:.1f} "
                        f"at step {step}. Starting {rampup}-step rampup now."
                    )
                ))
                self._mor_triggered_by_loss = True
                self.mor_enable_step = step  # Update for tracking
        
        # NOTE: MoD enablement is step-based warmup (see TrainingConfig.mod_mlp_warmup_steps).
        # The previous MoR-informed trigger was removed because it was too brittle.
        
        return actions

    def _compute_early_exit_ratio(self, routing_stats: dict) -> Optional[float]:
        """Compute MoR early exit ratio from routing stats."""
        mor_stats = routing_stats.get("mor_layers", [])
        if not mor_stats:
            return None
        
        # Aggregate depth histograms across all MoR layers
        all_hists = [s.get("depth_histogram", []) for s in mor_stats if s.get("depth_histogram")]
        if not all_hists:
            return None
        
        max_len = max(len(h) for h in all_hists)
        if max_len == 0:
            return None
        
        agg_hist = [0.0] * max_len
        for h in all_hists:
            for i, v in enumerate(h):
                agg_hist[i] += v
        
        total = sum(agg_hist) or 1
        # Early exit ratio = tokens that exit before max depth
        # Last bucket is max depth, so early_exit = 1 - (last_bucket / total)
        early_exit_ratio = 1.0 - (agg_hist[-1] / total)
        return early_exit_ratio

    def get_mor_curriculum_params(self) -> Tuple[int, int]:
        """Return (mor_enable_step, mor_rampup_steps) for model initialization."""
        return self.mor_enable_step, self.mor_rampup_steps

    @property
    def mor_already_triggered(self) -> bool:
        """Return True if MoR was triggered early by loss threshold."""
        return self._mor_triggered_by_loss

    @property
    def mod_triggered(self) -> bool:
        """Return True if MoD has been triggered."""
        return self._mod_triggered
