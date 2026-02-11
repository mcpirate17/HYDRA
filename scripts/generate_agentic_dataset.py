#!/usr/bin/env python3
"""
Generate agentic thinking training data for HYDRA.

Creates examples where the model thinks/plans before responding.
Uses Claude API to generate high-quality agentic traces.

Usage:
    source /home/tim/venvs/llm/bin/activate && \
    ANTHROPIC_API_KEY=sk-... python scripts/generate_agentic_dataset.py \
        --num_samples 1000 --output data/agentic_thinking.jsonl

    # Estimate cost first
    python scripts/generate_agentic_dataset.py --num_samples 1000 --estimate_cost
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_log = logging.getLogger(__name__)

# Cost estimates (Claude Sonnet)
INPUT_COST_PER_M = 3.0
OUTPUT_COST_PER_M = 15.0
AVG_INPUT_TOKENS = 300
AVG_OUTPUT_TOKENS = 250

SYSTEM_PROMPT = """You are generating training data for a small AI assistant that will have agentic capabilities (file operations, search, code execution, etc.).

Your task: Generate realistic user requests and model responses that demonstrate THINKING BEFORE ACTING.

Format your response EXACTLY as:
<|user|>
[realistic user request - varied complexity, natural language]
<|assistant|>
<think>
[1-4 sentences of internal reasoning:
- What is the user actually asking for?
- What approach should I take?
- What tools/actions might I need?
- Any edge cases or considerations?]
</think>
[The actual response - helpful, concise, may include action placeholders like [ACTION: tool_name args]]

IMPORTANT GUIDELINES:
1. User requests should be DIVERSE - not just coding. Include:
   - File/folder operations ("find files with...", "organize my downloads")
   - Information lookup ("what's the weather", "explain concept X")
   - Task planning ("help me plan a trip", "create a schedule")
   - Code assistance ("fix this bug", "write a function")
   - System operations ("check disk space", "list processes")
   - Creative tasks ("write an email", "summarize this")
   - Conversational ("how are you", "what can you do")

2. The <think> section should be PRACTICAL, not performative:
   - Short (1-4 sentences)
   - Focus on task decomposition
   - Identify what tools/approaches to use
   - Note any ambiguities to clarify

3. Responses should be NATURAL:
   - Varied length (some short, some detailed)
   - Sometimes ask clarifying questions
   - Use [ACTION: ...] for tool use
   - Be helpful but not overly verbose

4. Include some MULTI-TURN awareness:
   - "Based on what you mentioned earlier..."
   - "Following up on the previous request..."

Generate exactly ONE example per response. Make each unique and realistic."""

# Seed prompts to guide variety
SEED_CATEGORIES = [
    "file operations",
    "code debugging",
    "code writing",
    "explanation request",
    "creative writing",
    "task planning",
    "system administration",
    "data analysis",
    "conversational/greeting",
    "search/lookup",
    "summarization",
    "email/communication",
    "learning/tutorial",
    "troubleshooting",
    "configuration",
    "refactoring",
    "testing",
    "documentation",
    "git operations",
    "general help",
]


def generate_example(client, model: str, category: str) -> dict[str, Any] | None:
    """Generate a single agentic thinking example."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Generate a training example in the category: {category}\n\nRemember to use the exact format with <|user|>, <|assistant|>, and <think> tags.",
                }
            ],
        )

        content = response.content[0].text

        # Parse the response
        if "<|user|>" not in content or "<|assistant|>" not in content:
            _log.warning(f"Invalid format, missing tags")
            return None

        if "<think>" not in content or "</think>" not in content:
            _log.warning(f"Invalid format, missing think tags")
            return None

        # Extract parts
        parts = content.split("<|assistant|>")
        if len(parts) != 2:
            return None

        user_part = parts[0].replace("<|user|>", "").strip()
        assistant_part = parts[1].strip()

        # Extract thinking
        think_start = assistant_part.find("<think>")
        think_end = assistant_part.find("</think>")
        if think_start == -1 or think_end == -1:
            return None

        thinking = assistant_part[think_start + 7:think_end].strip()
        response_text = assistant_part[think_end + 8:].strip()

        return {
            "prompt": user_part,
            "thinking": thinking,
            "response": response_text,
            "full_completion": assistant_part,
            "category": category,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model": model,
            "generated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        _log.error(f"Generation error: {e}")
        return None


def estimate_cost(num_samples: int) -> float:
    """Estimate API cost."""
    input_cost = (num_samples * AVG_INPUT_TOKENS / 1_000_000) * INPUT_COST_PER_M
    output_cost = (num_samples * AVG_OUTPUT_TOKENS / 1_000_000) * OUTPUT_COST_PER_M
    return input_cost + output_cost


def main():
    parser = argparse.ArgumentParser(description="Generate agentic thinking dataset")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="data/agentic_thinking.jsonl", help="Output file path")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="Model to use")
    parser.add_argument("--estimate_cost", action="store_true", help="Only estimate cost, don't run")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    args = parser.parse_args()

    if args.estimate_cost:
        cost = estimate_cost(args.num_samples)
        print(f"Estimated cost for {args.num_samples} samples: ${cost:.2f}")
        print(f"  Input:  ~{args.num_samples * AVG_INPUT_TOKENS:,} tokens (${(args.num_samples * AVG_INPUT_TOKENS / 1_000_000) * INPUT_COST_PER_M:.2f})")
        print(f"  Output: ~{args.num_samples * AVG_OUTPUT_TOKENS:,} tokens (${(args.num_samples * AVG_OUTPUT_TOKENS / 1_000_000) * OUTPUT_COST_PER_M:.2f})")
        return

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _log.error("ANTHROPIC_API_KEY environment variable not set")
        return

    # Import anthropic
    try:
        import anthropic
    except ImportError:
        _log.error("anthropic package not installed. Run: pip install anthropic")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Setup output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for resume
    existing_count = 0
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing_count = sum(1 for _ in f)
        _log.info(f"Resuming from {existing_count} existing examples")

    samples_needed = args.num_samples - existing_count
    if samples_needed <= 0:
        _log.info(f"Already have {existing_count} samples, nothing to do")
        return

    _log.info(f"Generating {samples_needed} agentic thinking examples...")
    _log.info(f"Output: {output_path}")
    _log.info(f"Model: {args.model}")
    _log.info(f"Estimated cost: ${estimate_cost(samples_needed):.2f}")

    # Generate examples
    mode = "a" if args.resume else "w"
    success_count = 0
    fail_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    with open(output_path, mode) as f:
        for i in range(samples_needed):
            # Rotate through categories
            category = SEED_CATEGORIES[i % len(SEED_CATEGORIES)]

            example = generate_example(client, args.model, category)

            if example:
                f.write(json.dumps(example) + "\n")
                f.flush()
                success_count += 1
                total_input_tokens += example["input_tokens"]
                total_output_tokens += example["output_tokens"]

                if success_count % 10 == 0:
                    cost_so_far = (total_input_tokens / 1_000_000) * INPUT_COST_PER_M + \
                                  (total_output_tokens / 1_000_000) * OUTPUT_COST_PER_M
                    _log.info(f"Progress: {success_count}/{samples_needed} ({fail_count} failed) - ${cost_so_far:.2f}")
            else:
                fail_count += 1

            # Rate limiting
            time.sleep(0.1)

    # Final stats
    total_cost = (total_input_tokens / 1_000_000) * INPUT_COST_PER_M + \
                 (total_output_tokens / 1_000_000) * OUTPUT_COST_PER_M

    _log.info("=" * 60)
    _log.info(f"Generation complete!")
    _log.info(f"  Successful: {success_count}")
    _log.info(f"  Failed: {fail_count}")
    _log.info(f"  Total tokens: {total_input_tokens + total_output_tokens:,}")
    _log.info(f"  Total cost: ${total_cost:.2f}")
    _log.info(f"  Output: {output_path}")
    _log.info("=" * 60)


if __name__ == "__main__":
    main()
