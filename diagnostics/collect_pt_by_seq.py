"""Collect converted .pt shards into per-sequence-length folders and report totals.

Goal:
- Create a unified folder per seq_len regardless of source dataset.
- Rename shard filenames to avoid collisions.
- Compute approximate totals (examples + tokens) per seq_len.

Expected input layouts:
1) Standardized layout (preferred):
   <root>/<seq_len>/<dataset>/chunk_*.pt
2) Legacy layout (fallback):
   <root>/<dataset>_<seq_len>/chunk_*.pt

Output layout:
- Flat (preferred for "one source of truth"):
    <out>/<seq_len>/chunk_000000.pt  (symlink/hardlink/copy; sequentially numbered)
- Verbose (debugging):
    <out>/<seq_len>/chunk_<dataset>_<root-tag>_<orig>.pt

Token counting:
- If shard is list[dict] with input_ids: counts sum(max(0, len(ids)-1)).
- If shard is torch.Tensor [N, seq_len+1]: counts N * seq_len.

Example:
  source /home/tim/venvs/llm/bin/activate && \
  python diagnostics/collect_pt_by_seq.py \
    --roots /mnt/nvme0/hydra_small_chat_pt /mnt/nvme0/hydra_nemotron_pt \
    --out /mnt/nvme0/hydra_all_chat_by_seq \
    --seq-lens 512,1024,2048,4096,8192 --mode symlink
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Iterable

import torch


def _parse_seq_lens(arg: str) -> list[int]:
    parts = [p.strip() for p in str(arg).split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        try:
            v = int(p)
        except Exception:
            continue
        if v < 128 or v > 16384:
            continue
        if v % 128 != 0:
            continue
        out.append(v)
    return sorted(set(out))


def _safe_tag(path: Path) -> str:
    s = str(path).strip().rstrip("/")
    if not s:
        return "root"
    # keep last 2 path parts for disambiguation
    parts = [p for p in s.split("/") if p]
    tag = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tag)
    return tag or "root"


_CHUNK_NUM_RE = re.compile(r"^chunk_(\d+)\.pt$")


def _next_flat_index(seq_dir: Path) -> int:
    """Return next available numeric chunk index in seq_dir."""
    if not seq_dir.exists():
        return 0
    mx = -1
    for p in seq_dir.iterdir():
        if not p.is_file() and not p.is_symlink():
            continue
        m = _CHUNK_NUM_RE.match(p.name)
        if not m:
            continue
        try:
            v = int(m.group(1))
        except Exception:
            continue
        mx = max(mx, v)
    return mx + 1


def _clear_flat_chunks(seq_dir: Path) -> int:
    """Remove chunk_*.pt directly under seq_dir (does not touch subfolders)."""
    if not seq_dir.exists():
        return 0
    n = 0
    for p in list(seq_dir.iterdir()):
        if not (p.is_file() or p.is_symlink()):
            continue
        if not _CHUNK_NUM_RE.match(p.name):
            continue
        try:
            p.unlink()
            n += 1
        except Exception:
            continue
    return n


@dataclass(frozen=True)
class ShardInfo:
    src: Path
    dataset: str
    seq_len: int


def _iter_standardized(root: Path, seq_lens: set[int]) -> Iterable[ShardInfo]:
    for seq_len in sorted(seq_lens):
        seq_dir = root / str(seq_len)
        if not seq_dir.is_dir():
            continue
        for ds_dir in seq_dir.iterdir():
            if not ds_dir.is_dir():
                continue
            dataset = ds_dir.name
            for pt in sorted(ds_dir.glob("chunk_*.pt")):
                yield ShardInfo(src=pt, dataset=dataset, seq_len=seq_len)


def _iter_legacy(root: Path, seq_lens: set[int]) -> Iterable[ShardInfo]:
    # root/<dataset>_<seq_len>/chunk_*.pt
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if "_" not in name:
            continue
        try:
            tail = int(name.rsplit("_", 1)[-1])
        except Exception:
            continue
        if tail not in seq_lens:
            continue
        dataset = name.rsplit("_", 1)[0]
        for pt in sorted(child.glob("chunk_*.pt")):
            yield ShardInfo(src=pt, dataset=dataset, seq_len=tail)


def _count_shard(path: Path, seq_len: int) -> tuple[int, int]:
    obj = torch.load(path, weights_only=False)

    if isinstance(obj, torch.Tensor):
        # Assume [N, seq_len+1]
        n = int(obj.shape[0])
        return n, n * int(seq_len)

    if isinstance(obj, list):
        examples = 0
        tokens = 0
        for item in obj:
            if isinstance(item, dict):
                ids = item.get("input_ids", item.get("tokens"))
                if isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                if isinstance(ids, list):
                    examples += 1
                    tokens += max(0, len(ids) - 1)
            elif isinstance(item, torch.Tensor):
                ids = item.tolist()
                examples += 1
                tokens += max(0, len(ids) - 1)
            elif isinstance(item, list):
                examples += 1
                tokens += max(0, len(item) - 1)
        return examples, tokens

    return 0, 0


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return

    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        import shutil

        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--seq-lens",
        type=str,
        default="512,1024,2048",
        help="Comma-separated seq lens to collect (e.g. 512,1024,2048,4096,8192)",
    )
    ap.add_argument(
        "--mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to materialize collected shards into --out",
    )
    ap.add_argument(
        "--flat",
        action="store_true",
        help="Write a single flat directory per seq_len (no dataset/root in filenames)",
    )
    ap.add_argument(
        "--clear-flat",
        action="store_true",
        help="When used with --flat, delete existing chunk_*.pt files in each <out>/<seq_len>/ before writing",
    )
    ap.add_argument(
        "--write-json",
        action="store_true",
        help="Write per-bucket totals JSON to <out>/seq_bucket_totals.json",
    )
    ap.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write mapping manifest JSONL to <out>/manifest.jsonl (dst -> src)",
    )
    ap.add_argument(
        "--max-shards-per-dataset",
        type=int,
        default=None,
        help="Optional cap for debugging; limits collected shards per dataset per root per seq",
    )

    args = ap.parse_args()

    seq_lens = set(_parse_seq_lens(args.seq_lens))
    if not seq_lens:
        raise SystemExit("No valid --seq-lens provided")

    out = Path(os.path.abspath(args.out))
    out.mkdir(parents=True, exist_ok=True)

    # Flat-mode counters: keep chunk_XXXXXX.pt unique per seq bucket.
    flat_next: dict[int, int] = {}
    if args.flat:
        for s in sorted(seq_lens):
            seq_dir = out / str(int(s))
            seq_dir.mkdir(parents=True, exist_ok=True)
            if args.clear_flat:
                _clear_flat_chunks(seq_dir)
            flat_next[int(s)] = _next_flat_index(seq_dir)

    manifest_fp = None
    if args.write_manifest:
        manifest_fp = (out / "manifest.jsonl").open("w", encoding="utf-8")

    totals: dict[str, dict[str, Any]] = {}
    for s in sorted(seq_lens):
        totals[str(s)] = {
            "seq_len": int(s),
            "examples": 0,
            "tokens": 0,
            "shards": 0,
        }

    for root in args.roots:
        root = Path(root)
        if not root.exists():
            continue
        root_tag = _safe_tag(root)

        # Gather shards (prefer standardized, but include legacy too).
        shards = list(_iter_standardized(root, seq_lens))
        shards += list(_iter_legacy(root, seq_lens))

        # Apply optional cap per dataset/seq within each root.
        if args.max_shards_per_dataset is not None:
            limited: list[ShardInfo] = []
            seen: dict[tuple[int, str], int] = {}
            cap = int(args.max_shards_per_dataset)
            for sh in shards:
                key = (int(sh.seq_len), str(sh.dataset))
                seen[key] = seen.get(key, 0) + 1
                if seen[key] <= cap:
                    limited.append(sh)
            shards = limited

        for sh in shards:
            seq_dir = out / str(int(sh.seq_len))
            seq_dir.mkdir(parents=True, exist_ok=True)

            if args.flat:
                idx = flat_next.get(int(sh.seq_len), 0)
                flat_next[int(sh.seq_len)] = idx + 1
                dst = seq_dir / f"chunk_{idx:06d}.pt"
            else:
                # Verbose rename to avoid collisions, keep LocalDataLoader-compatible prefix.
                dst = seq_dir / f"chunk_{sh.dataset}_{root_tag}_{sh.src.name}"

            _link_or_copy(sh.src, dst, args.mode)

            if manifest_fp is not None:
                manifest_fp.write(
                    json.dumps(
                        {
                            "seq_len": int(sh.seq_len),
                            "dataset": sh.dataset,
                            "src": str(sh.src),
                            "dst": str(dst),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

            ex, tok = _count_shard(sh.src, sh.seq_len)
            bucket = totals[str(int(sh.seq_len))]
            bucket["examples"] += int(ex)
            bucket["tokens"] += int(tok)
            bucket["shards"] += 1

    # Print report
    for s in sorted(seq_lens):
        b = totals[str(s)]
        gb = (b["tokens"] * 4) / (1024**3)  # rough int32 token footprint
        print(
            f"seq={b['seq_len']:>5}  shards={b['shards']:>6}  examples={b['examples']:>10}  tokens={b['tokens']:>12}  (~{gb:.2f} GiB @ int32/tok)"
        )

    if args.write_json:
        p = out / "seq_bucket_totals.json"
        p.write_text(json.dumps(totals, indent=2, sort_keys=True))
        print(f"Wrote {p}")

    if manifest_fp is not None:
        manifest_fp.close()
        print(f"Wrote {out / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
