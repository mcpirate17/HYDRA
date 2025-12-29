"""Convert small instruction/chat datasets into HYDRA local `.pt` shards.

Inputs (downloaded under /mnt/nvme0/small_chat):
- Alpaca Cleaned: JSON array of {instruction, input, output}
- CodeAlpaca-20k: JSON array of {instruction, input, output}
- Dolly 15k: JSONL of {instruction, context, response, category}
- oo-labeled_correct.gpt4.sharegpt.jsonl: ShareGPT JSONL with conversations[{from,value}, ...]

Output:
- Writes chunk_XXXX.pt shards as a list[dict] with keys:
    - input_ids: list[int]
    - length: int

This format is directly supported by hydra.data.universal_data_loader.LocalDataLoader
and preserves example boundaries (to avoid "garbage" window splits).

Layout (standardized):
- <out-base>/<seq_len>/<dataset>/chunk_XXXX.pt

Example (single run creates 512/1024/2048 directories, no truncation):
    source /home/tim/venvs/llm/bin/activate && \
    python diagnostics/convert_small_chat_to_pt.py \
        --input-dir /mnt/nvme0/small_chat \
        --out-base /mnt/nvme0/hydra_small_chat_pt \
        --seq-lens 512,1024,2048 --tokenizer gpt2 --shard-samples 4096
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

try:
    from transformers import AutoTokenizer
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "transformers is required for this converter. Install it in your env. "
        f"Original error: {type(e).__name__}: {e}"
    )


_SYSTEM = "You are a helpful assistant."


@dataclass(frozen=True)
class Example:
    user: str
    assistant: str


@dataclass
class _BucketWriter:
    out_dir: Path
    shard_samples: int
    shard_idx: int = 0
    shard: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard = []

    def add(self, sample: dict[str, Any]) -> None:
        self.shard.append(sample)
        if len(self.shard) >= self.shard_samples:
            self.flush()

    def flush(self) -> None:
        if not self.shard:
            return
        torch.save(self.shard, self.out_dir / f"chunk_{self.shard_idx:04d}.pt")
        self.shard_idx += 1
        self.shard.clear()


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _format_instruction_pair(*, instruction: str, context: str, output: str) -> Optional[Example]:
    inst = instruction.strip()
    out = output.strip()
    ctx = context.strip()
    if not inst or not out:
        return None

    # Minimal guardrails: avoid tiny or placeholder outputs.
    if len(out) < 16:
        return None

    if ctx:
        user = f"{inst}\n\n### Input\n{ctx}"
    else:
        user = inst

    return Example(user=user, assistant=out)


def _format_chat_text(ex: Example) -> str:
    return "\n".join(
        [
            f"<|system|>\n{_SYSTEM}",
            f"<|user|>\n{ex.user}",
            f"<|assistant|>\n{ex.assistant}",
        ]
    )


def _iter_alpaca_json(path: Path) -> Iterable[Example]:
    arr = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(arr, list):
        return
    for obj in arr:
        if not isinstance(obj, dict):
            continue
        inst = _as_str(obj.get("instruction")).strip()
        inp = _as_str(obj.get("input")).strip()
        out = _as_str(obj.get("output")).strip()
        ex = _format_instruction_pair(instruction=inst, context=inp, output=out)
        if ex is not None:
            yield ex


def _iter_dolly_jsonl(path: Path) -> Iterable[Example]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            inst = _as_str(obj.get("instruction")).strip()
            ctx = _as_str(obj.get("context")).strip()
            resp = _as_str(obj.get("response")).strip()
            ex = _format_instruction_pair(instruction=inst, context=ctx, output=resp)
            if ex is not None:
                yield ex


def _iter_sharegpt_jsonl(path: Path) -> Iterable[Example]:
    """Parse ShareGPT-style conversations into single-turn examples.

    Expected: {"conversations": [{"from": "human"|"gpt"|..., "value": str}, ...]}
    We extract the LAST (human -> assistant) pair to keep prompts short and avoid
    dumping long multi-turn history into 512/1024 phases.
    """

    def norm_role(r: str) -> str:
        r = r.strip().lower()
        if r in {"human", "user"}:
            return "user"
        if r in {"gpt", "assistant", "model"}:
            return "assistant"
        if r in {"system"}:
            return "system"
        return r

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv = obj.get("conversations")
            if not isinstance(conv, list) or not conv:
                continue

            # Find last assistant message and its preceding user message.
            last_asst = None
            last_user = None
            for i in range(len(conv) - 1, -1, -1):
                m = conv[i]
                if not isinstance(m, dict):
                    continue
                role = norm_role(_as_str(m.get("from", "")))
                text = _as_str(m.get("value", "")).strip()
                if not text:
                    continue
                if last_asst is None and role == "assistant":
                    last_asst = text
                    continue
                if last_asst is not None and role == "user":
                    last_user = text
                    break

            if not last_user or not last_asst:
                continue

            ex = _format_instruction_pair(instruction=last_user, context="", output=last_asst)
            if ex is not None:
                yield ex


def _convert_dataset_multi_bucket(
    *,
    dataset_name: str,
    examples: Iterable[Example],
    out_base: Path,
    seq_lens: list[int],
    tok,
    eos_token_id: int,
    shard_samples: int,
    max_examples: Optional[int],
    min_seq_len: int = 0,
) -> None:
    writers: dict[int, _BucketWriter] = {}
    for seq_len in seq_lens:
        if int(seq_len) < int(min_seq_len):
            continue
        writers[int(seq_len)] = _BucketWriter(
            out_dir=out_base / str(int(seq_len)) / dataset_name,
            shard_samples=shard_samples,
        )

    n_seen = 0
    for ex in examples:
        text = _format_chat_text(ex)
        if len(text) < 80:
            continue

        ids = tok.encode(text, add_special_tokens=False)
        ids.append(int(eos_token_id))
        if len(ids) < 65:
            continue

        for seq_len, w in writers.items():
            if len(ids) <= (int(seq_len) + 1):
                w.add({"input_ids": ids, "length": len(ids)})

        n_seen += 1
        if max_examples is not None and n_seen >= max_examples:
            break

    for w in writers.values():
        w.flush()


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
    # Ensure ascending unique.
    out = sorted(set(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("/mnt/nvme0/small_chat"))
    ap.add_argument("--out-base", type=Path, default=Path("/mnt/nvme0/hydra_small_chat_pt"))
    ap.add_argument(
        "--seq-lens",
        type=str,
        default="512,1024,2048",
        help="Comma-separated seq lens to generate in one run (e.g. 512,1024,2048)",
    )
    ap.add_argument("--tokenizer", type=str, default="gpt2")
    ap.add_argument("--shard-samples", type=int, default=4096, help="Examples per chunk_XXXX.pt")
    ap.add_argument("--max-examples", type=int, default=None, help="Optional cap for smoke runs")

    args = ap.parse_args()

    input_dir: Path = args.input_dir
    out_base = Path(os.path.abspath(args.out_base))
    seq_lens = _parse_seq_lens(args.seq_lens)
    if not seq_lens:
        raise SystemExit("No valid --seq-lens provided")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tok.model_max_length = 10**9
    eos = tok.eos_token_id
    if eos is None:
        raise ValueError(f"Tokenizer {args.tokenizer} has no eos_token_id")

    alpaca = input_dir / "alpaca_data_cleaned.json"
    code_alpaca = input_dir / "code_alpaca_20k.json"
    dolly = input_dir / "databricks-dolly-15k.jsonl"
    sharegpt = input_dir / "oo-labeled_correct.gpt4.sharegpt.jsonl"

    if not alpaca.exists() or not code_alpaca.exists() or not dolly.exists():
        raise SystemExit(
            "Expected files not found under input-dir. Need: alpaca_data_cleaned.json, code_alpaca_20k.json, databricks-dolly-15k.jsonl"
        )

    shard_samples = max(128, int(args.shard_samples))

    _convert_dataset_multi_bucket(
        dataset_name="alpaca_cleaned",
        examples=_iter_alpaca_json(alpaca),
        out_base=out_base,
        seq_lens=seq_lens,
        tok=tok,
        eos_token_id=int(eos),
        shard_samples=shard_samples,
        max_examples=args.max_examples,
    )

    _convert_dataset_multi_bucket(
        dataset_name="code_alpaca",
        examples=_iter_alpaca_json(code_alpaca),
        out_base=out_base,
        seq_lens=seq_lens,
        tok=tok,
        eos_token_id=int(eos),
        shard_samples=shard_samples,
        max_examples=args.max_examples,
    )

    _convert_dataset_multi_bucket(
        dataset_name="dolly",
        examples=_iter_dolly_jsonl(dolly),
        out_base=out_base,
        seq_lens=seq_lens,
        tok=tok,
        eos_token_id=int(eos),
        shard_samples=shard_samples,
        max_examples=args.max_examples,
    )

    # ShareGPT-style file is intended for >=1024 contexts.
    if sharegpt.exists():
        _convert_dataset_multi_bucket(
            dataset_name="oo_labeled_correct",
            examples=_iter_sharegpt_jsonl(sharegpt),
            out_base=out_base,
            seq_lens=seq_lens,
            tok=tok,
            eos_token_id=int(eos),
            shard_samples=shard_samples,
            max_examples=args.max_examples,
            min_seq_len=1024,
        )


if __name__ == "__main__":
    main()
