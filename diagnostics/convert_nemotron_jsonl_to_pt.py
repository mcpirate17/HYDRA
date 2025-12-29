"""Convert Nemotron instruction-following chat JSONL into HYDRA local .pt shards.

Input format (observed): JSONL records with:
  - messages: [{role: str, content: str}, ...]
  - optional other fields (reasoning/tools/uuid/etc)

Output format:
    - Standardized layout: <out-base>/<seq_len>/<dataset>/chunk_XXXX.pt
    - Writes chunk_XXXX.pt shards as a list[dict] with keys:
            - input_ids: list[int]
            - length: int
    - Compatible with hydra.data.universal_data_loader.LocalDataLoader

Notes:
- We intentionally ignore top-level 'reasoning' fields. Training on hidden chain-of-thought
  style fields is usually undesirable.
- Empty system messages are dropped by default.

Example:
  source /home/tim/venvs/llm/bin/activate && \
  python diagnostics/convert_nemotron_jsonl_to_pt.py \
        --input /mnt/nvme0/nvidia_nemotron_instruction_following/chat_if.jsonl \
        --out-base /mnt/nvme0/hydra_nemotron_pt \
        --dataset chat_if \
        --seq-lens 512,1024,2048,4096,8192 --tokenizer gpt2 --shard-samples 4096
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from transformers import AutoTokenizer
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "transformers is required for this converter. Install it in your env. "
        f"Original error: {type(e).__name__}: {e}"
    )


def _format_messages(messages: list[dict[str, Any]], drop_empty_system: bool) -> str:
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "user"))
        content = m.get("content", "")
        if content is None:
            content = ""
        content = str(content)
        if drop_empty_system and role == "system" and content.strip() == "":
            continue
        parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


def _extract_last_user_assistant(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract a single-turn (user -> assistant) pair from a multi-turn chat.

    Keeps the first non-empty system message (if present), then the last user message
    that precedes the last assistant message.
    """

    if not messages:
        return []

    def norm_role(r: Any) -> str:
        r = str(r or "").strip().lower()
        if r in {"human", "user"}:
            return "user"
        if r in {"assistant", "gpt", "model"}:
            return "assistant"
        if r in {"system"}:
            return "system"
        return r

    system_msg: dict[str, Any] | None = None
    for m in messages:
        if not isinstance(m, dict):
            continue
        if norm_role(m.get("role")) != "system":
            continue
        content = m.get("content", "")
        if content is None:
            content = ""
        if str(content).strip() == "":
            continue
        system_msg = {"role": "system", "content": str(content)}
        break

    last_asst_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if not isinstance(m, dict):
            continue
        if norm_role(m.get("role")) != "assistant":
            continue
        content = m.get("content", "")
        if content is None:
            content = ""
        if str(content).strip() == "":
            continue
        last_asst_idx = i
        break

    if last_asst_idx is None:
        return []

    last_user: dict[str, Any] | None = None
    for i in range(last_asst_idx - 1, -1, -1):
        m = messages[i]
        if not isinstance(m, dict):
            continue
        if norm_role(m.get("role")) != "user":
            continue
        content = m.get("content", "")
        if content is None:
            content = ""
        if str(content).strip() == "":
            continue
        last_user = {"role": "user", "content": str(content)}
        break

    if last_user is None:
        return []

    last_asst = messages[last_asst_idx]
    last_asst_content = last_asst.get("content", "")
    if last_asst_content is None:
        last_asst_content = ""
    last_asst_norm = {"role": "assistant", "content": str(last_asst_content)}

    out: list[dict[str, Any]] = []
    if system_msg is not None:
        out.append(system_msg)
    out.append(last_user)
    out.append(last_asst_norm)
    return out


def _iter_texts(
    fp,
    *,
    drop_empty_system: bool,
    require_messages: bool = True,
    pair_mode: str = "full",
) -> Iterable[str]:
    for line in fp:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        messages = obj.get("messages")
        if require_messages:
            if not isinstance(messages, list):
                continue
            if pair_mode == "last_pair":
                selected = _extract_last_user_assistant(messages)
                if not selected:
                    continue
                text = _format_messages(selected, drop_empty_system=drop_empty_system)
            else:
                text = _format_messages(messages, drop_empty_system=drop_empty_system)
        else:
            text = str(obj.get("text", ""))

        if len(text) < 50:
            continue
        yield text


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


def convert_jsonl_to_pt_multi_bucket(
    *,
    input_path: Path,
    out_base: Path,
    dataset_name: str,
    seq_lens: list[int],
    tokenizer_name: str,
    batch_size: int,
    shard_samples: int,
    max_records: int | None,
    drop_empty_system: bool,
    pair_mode: str,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.model_max_length = 10**9
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError(f"Tokenizer {tokenizer_name} has no eos_token_id")

    writers: dict[int, _BucketWriter] = {}
    for seq_len in seq_lens:
        writers[int(seq_len)] = _BucketWriter(
            out_dir=out_base / str(int(seq_len)) / str(dataset_name),
            shard_samples=shard_samples,
        )

    n_records = 0
    with input_path.open("r", encoding="utf-8") as f:
        texts_batch: list[str] = []
        for text in _iter_texts(
            f, drop_empty_system=drop_empty_system, pair_mode=str(pair_mode)
        ):
            texts_batch.append(text)
            if len(texts_batch) < batch_size:
                continue

            enc = tokenizer(
                texts_batch,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            for ids in enc["input_ids"]:
                ids = list(ids)
                ids.append(int(eos))
                if len(ids) < 65:
                    continue

                for seq_len, w in writers.items():
                    if len(ids) <= (int(seq_len) + 1):
                        w.add({"input_ids": ids, "length": len(ids)})

                n_records += 1
                if max_records is not None and n_records >= max_records:
                    break

            texts_batch.clear()
            if max_records is not None and n_records >= max_records:
                break

        if texts_batch and (max_records is None or n_records < max_records):
            enc = tokenizer(
                texts_batch,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            for ids in enc["input_ids"]:
                ids = list(ids)
                ids.append(int(eos))
                if len(ids) < 65:
                    continue

                for seq_len, w in writers.items():
                    if len(ids) <= (int(seq_len) + 1):
                        w.add({"input_ids": ids, "length": len(ids)})

                n_records += 1
                if max_records is not None and n_records >= max_records:
                    break

    for w in writers.values():
        w.flush()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--out-base", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=str, help="Dataset name used in output folder (e.g. chat_if)")
    p.add_argument(
        "--seq-lens",
        type=str,
        default="512,1024,2048",
        help="Comma-separated seq lens to generate in one run (e.g. 512,1024,2048,4096,8192)",
    )
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--batch", type=int, default=256, help="Records tokenized per call")
    p.add_argument("--shard-samples", type=int, default=4096, help="Examples per .pt shard")
    p.add_argument("--max-records", type=int, default=None, help="Optional limit for quick smoke conversion")
    p.add_argument("--keep-empty-system", action="store_true", help="Keep empty system messages")
    p.add_argument(
        "--pair-mode",
        choices=["full", "last_pair"],
        default="full",
        help="'full' keeps full multi-turn; 'last_pair' keeps only last user->assistant (+first system if any)",
    )

    args = p.parse_args()

    input_path: Path = args.input
    out_base: Path = args.out_base
    dataset_name: str = str(args.dataset)
    seq_lens = _parse_seq_lens(args.seq_lens)
    if not seq_lens:
        raise SystemExit("No valid --seq-lens provided")

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    # Avoid accidentally writing into a relative path from elsewhere.
    out_base = Path(os.path.abspath(out_base))

    convert_jsonl_to_pt_multi_bucket(
        input_path=input_path,
        out_base=out_base,
        dataset_name=dataset_name,
        seq_lens=seq_lens,
        tokenizer_name=args.tokenizer,
        batch_size=max(1, int(args.batch)),
        shard_samples=max(64, int(args.shard_samples)),
        max_records=args.max_records,
        drop_empty_system=not args.keep_empty_system,
        pair_mode=str(args.pair_mode),
    )


if __name__ == "__main__":
    main()
