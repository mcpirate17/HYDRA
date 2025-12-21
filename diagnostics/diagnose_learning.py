from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True, slots=True)
class StepRecord:
    step: int
    max_steps: int
    loss: float
    ema: float
    lr: float
    grad: float
    tok_s: float
    avg_tok_s: float
    steps_s: float


@dataclass(frozen=True, slots=True)
class EvalRecord:
    step: int
    eval_loss: float
    train_loss: float


@dataclass(frozen=True, slots=True)
class LogDiagnosis:
    tokens_per_step: Optional[int]
    steps: Tuple[StepRecord, ...]
    evals: Tuple[EvalRecord, ...]
    mor_issue_lines: Tuple[str, ...]
    mod_issue_lines: Tuple[str, ...]
    synthetic_fallback_lines: Tuple[str, ...]


class PlateauDetector:
    __slots__ = ("window", "min_step", "slope_abs_threshold")

    def __init__(self, *, window: int, min_step: int, slope_abs_threshold: float) -> None:
        if window <= 2:
            raise ValueError("window must be > 2")
        self.window = window
        self.min_step = min_step
        self.slope_abs_threshold = slope_abs_threshold

    def find_plateau_start(self, steps: Sequence[StepRecord]) -> Optional[int]:
        if len(steps) < self.window:
            return None
        xs = [s.step for s in steps]
        ys = [s.loss for s in steps]
        for i in range(self.window - 1, len(steps)):
            if steps[i].step < self.min_step:
                continue
            w_x = xs[i - self.window + 1 : i + 1]
            w_y = ys[i - self.window + 1 : i + 1]
            slope = _ols_slope(w_x, w_y)
            if slope is not None and abs(slope) <= self.slope_abs_threshold:
                return steps[i - self.window + 1].step
        return None


_STEP_RE = re.compile(
    r"Step\s+(?P<step>\d+)/(?P<max_steps>\d+)\s+\|\s+"
    r"Loss:\s+(?P<loss>[0-9.]+)\s+\(EMA:\s+(?P<ema>[0-9.]+)\)\s+\|\s+"
    r"LR:\s+(?P<lr>[0-9.eE+-]+)\s+\|\s+"
    r"Grad:\s+(?P<grad>[0-9.]+)\s+\|\s+"
    r"(?P<tok_s>[0-9.]+)K\s+tok/s\s+\|\s+"
    r"Avg:\s+(?P<avg_tok_s>[0-9.]+)K\s+tok/s\s+\((?P<steps_s>[0-9.]+)\s+steps/s\)"
)

_EVAL_RE = re.compile(
    r"\[EVAL\]\s+step=(?P<step>\d+)\s+"
    r"eval_loss=(?P<eval_loss>[0-9.]+)\s+"
    r"train_loss=(?P<train_loss>[0-9.]+)"
)

_TOKENS_PER_STEP_RE = re.compile(r"Tokens per step:\s+(?P<tps>[0-9,]+)")


def parse_training_log(lines: Iterable[str]) -> LogDiagnosis:
    tokens_per_step: Optional[int] = None
    steps: List[StepRecord] = []
    evals: List[EvalRecord] = []
    mor_issue_lines: List[str] = []
    mod_issue_lines: List[str] = []
    synthetic_fallback_lines: List[str] = []

    for raw in lines:
        line = raw.rstrip("\n")

        m_tps = _TOKENS_PER_STEP_RE.search(line)
        if m_tps is not None:
            tokens_per_step = int(m_tps.group("tps").replace(",", ""))
            continue

        m_step = _STEP_RE.search(line)
        if m_step is not None:
            steps.append(
                StepRecord(
                    step=int(m_step.group("step")),
                    max_steps=int(m_step.group("max_steps")),
                    loss=float(m_step.group("loss")),
                    ema=float(m_step.group("ema")),
                    lr=float(m_step.group("lr")),
                    grad=float(m_step.group("grad")),
                    tok_s=float(m_step.group("tok_s")) * 1000.0,
                    avg_tok_s=float(m_step.group("avg_tok_s")) * 1000.0,
                    steps_s=float(m_step.group("steps_s")),
                )
            )
            continue

        m_eval = _EVAL_RE.search(line)
        if m_eval is not None:
            evals.append(
                EvalRecord(
                    step=int(m_eval.group("step")),
                    eval_loss=float(m_eval.group("eval_loss")),
                    train_loss=float(m_eval.group("train_loss")),
                )
            )
            continue

        if "MoR Issues:" in line:
            mor_issue_lines.append(line)
            continue
        if "MoD Issues:" in line:
            mod_issue_lines.append(line)
            continue

        # Data poisoning risk: streaming loader falls back to synthetic tokens
        if ("adding synthetic tokens" in line) or ("Too many failures" in line and "synthetic" in line):
            synthetic_fallback_lines.append(line)
            continue

    return LogDiagnosis(
        tokens_per_step=tokens_per_step,
        steps=tuple(steps),
        evals=tuple(evals),
        mor_issue_lines=tuple(mor_issue_lines),
        mod_issue_lines=tuple(mod_issue_lines),
        synthetic_fallback_lines=tuple(synthetic_fallback_lines),
    )


def summarize(d: LogDiagnosis, *, plateau_window: int, plateau_min_step: int, plateau_slope: float) -> str:
    if not d.steps:
        raise ValueError("No training steps parsed from log")

    first = d.steps[0]
    last = d.steps[-1]

    detector = PlateauDetector(window=plateau_window, min_step=plateau_min_step, slope_abs_threshold=plateau_slope)
    plateau_start = detector.find_plateau_start(d.steps)

    token_line = ""
    if d.tokens_per_step is not None:
        total_tokens = last.step * d.tokens_per_step
        token_line = f"tokens_per_step={d.tokens_per_step:,}  tokens_seen≈{total_tokens/1e6:.2f}M"
    else:
        token_line = "tokens_per_step=(not found in log)"

    eval_line = ""
    if d.evals:
        ev = d.evals[-1]
        gap = ev.train_loss - ev.eval_loss
        eval_line = f"last_eval: step={ev.step} eval_loss={ev.eval_loss:.4f} train_loss={ev.train_loss:.4f} gap={gap:+.4f}"
    else:
        eval_line = "last_eval: (none found)"

    plateau_line = ""
    if plateau_start is None:
        plateau_line = f"plateau: none detected (window={plateau_window}, |slope|≤{plateau_slope})"
    else:
        plateau_line = f"plateau: detected starting around step {plateau_start} (window={plateau_window}, |slope|≤{plateau_slope})"

    issues = []
    if d.mor_issue_lines:
        issues.append(f"MoR warnings: {len(d.mor_issue_lines)} (example: {d.mor_issue_lines[0]})")
    else:
        issues.append("MoR warnings: none")
    if d.mod_issue_lines:
        issues.append(f"MoD warnings: {len(d.mod_issue_lines)} (example: {d.mod_issue_lines[0]})")
    else:
        issues.append("MoD warnings: none")
    if d.synthetic_fallback_lines:
        issues.append(f"data fallback: DETECTED ({len(d.synthetic_fallback_lines)} lines)")
    else:
        issues.append("data fallback: not detected")

    ppl_first = math.exp(first.loss) if first.loss < 50 else float("inf")
    ppl_last = math.exp(last.loss) if last.loss < 50 else float("inf")
    ppl_line = f"perplexity: start≈{ppl_first:.1f}  last≈{ppl_last:.1f}"

    out = [
        "HYDRA learning diagnosis",
        "---",
        f"steps_parsed={len(d.steps)}  step_range=[{first.step}..{last.step}]  {token_line}",
        f"loss: start={first.loss:.4f}  last={last.loss:.4f}  delta={last.loss-first.loss:+.4f}",
        ppl_line,
        plateau_line,
        eval_line,
        *issues,
        "---",
        "What this usually means:",
        "- If plateau happens very early (<~1-2e8 tokens), it can be normal underfitting; but with TinyStories you typically still see a steady downward slope.",
        "- Repeated MoR 'all tokens at same depth' suggests router collapse; this can reduce effective model capacity and may stall improvement.",
        "- Any 'synthetic tokens' fallback indicates the data loader silently substituted random tokens, which will severely harm learning.",
        "---",
        "Next actions to validate (fast, no code changes):",
        "- Re-run with MoR delayed: set --mor_enable_pct 0.35 (default) instead of 0.0, and compare the first 500 steps.",
        "- Re-run with MoR off: set --mor_adaptive false; if loss improves smoothly, routing is the culprit.",
        "- Ensure logs contain no 'adding synthetic tokens' lines; if they do, fix data loading before tuning anything else.",
    ]
    return "\n".join(out)


def _ols_slope(xs: Sequence[int], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = float(len(xs))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    s_xx = 0.0
    s_xy = 0.0
    for x, y in zip(xs, ys, strict=True):
        dx = float(x) - mean_x
        s_xx += dx * dx
        s_xy += dx * (float(y) - mean_y)
    if s_xx == 0.0:
        return None
    return s_xy / s_xx


def _read_text(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines(True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose HYDRA learning behavior from a training log")
    parser.add_argument("log", type=Path, help="Path to a HYDRA training .log file")
    parser.add_argument("--plateau-window", type=int, default=8, help="Window size (in log points) for plateau detection")
    parser.add_argument("--plateau-min-step", type=int, default=100, help="Ignore steps before this when detecting plateaus")
    parser.add_argument(
        "--plateau-slope",
        type=float,
        default=1e-4,
        help="Detect plateau when |d(loss)/d(step)| is below this threshold",
    )
    args = parser.parse_args(argv)

    if not args.log.exists():
        raise FileNotFoundError(args.log)

    diagnosis = parse_training_log(_read_text(args.log))
    print(
        summarize(
            diagnosis,
            plateau_window=args.plateau_window,
            plateau_min_step=args.plateau_min_step,
            plateau_slope=args.plateau_slope,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
