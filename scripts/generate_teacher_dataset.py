#!/usr/bin/env python3
"""
Generate teacher completions for offline distillation.

Uses Claude API (or other providers) to generate high-quality reasoning
completions that can be used for distillation training.

Usage:
    # Generate dataset with Claude
    source /home/tim/venvs/llm/bin/activate && python scripts/generate_teacher_dataset.py \
        --provider claude --model claude-sonnet-4-20250514 \
        --num_samples 1000 --output data/teacher_completions.jsonl

    # Estimate cost before running
    source /home/tim/venvs/llm/bin/activate && python scripts/generate_teacher_dataset.py \
        --provider claude --model claude-sonnet-4-20250514 \
        --num_samples 1000 --estimate_cost

    # Use OpenAI instead
    source /home/tim/venvs/llm/bin/activate && python scripts/generate_teacher_dataset.py \
        --provider openai --model gpt-4o \
        --num_samples 1000 --output data/teacher_completions.jsonl

Environment variables:
    ANTHROPIC_API_KEY - Required for Claude
    OPENAI_API_KEY - Required for OpenAI
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_log = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    name: str
    default_model: str
    input_cost_per_m: float  # $ per million tokens
    output_cost_per_m: float
    max_tokens: int = 1024


PROVIDERS = {
    "claude": ProviderConfig(
        name="claude",
        default_model="claude-sonnet-4-20250514",
        input_cost_per_m=3.0,
        output_cost_per_m=15.0,
        max_tokens=1024,
    ),
    "claude-opus": ProviderConfig(
        name="claude",
        default_model="claude-opus-4-20250514",
        input_cost_per_m=15.0,
        output_cost_per_m=75.0,
        max_tokens=1024,
    ),
    "openai": ProviderConfig(
        name="openai",
        default_model="gpt-4o",
        input_cost_per_m=2.5,
        output_cost_per_m=10.0,
        max_tokens=1024,
    ),
    "openai-mini": ProviderConfig(
        name="openai",
        default_model="gpt-4o-mini",
        input_cost_per_m=0.15,
        output_cost_per_m=0.60,
        max_tokens=1024,
    ),
}


REASONING_SYSTEM_PROMPT = """You are a helpful math tutor. When solving problems:

1. Think step-by-step, showing your reasoning clearly
2. Use clear mathematical notation
3. Verify your answer makes sense
4. Present your final answer in \\boxed{answer} format

Be thorough but concise. Focus on the mathematical reasoning."""


def load_math_prompts(max_prompts: int = 1000) -> List[Dict[str, Any]]:
    """Load math problems from OpenMathInstruct-2."""
    try:
        from datasets import load_dataset

        _log.info("Loading OpenMathInstruct-2 dataset...")
        ds = load_dataset(
            "nvidia/OpenMathInstruct-2",
            split="train",
            streaming=True,
        )

        prompts = []
        for i, item in enumerate(ds):
            if i >= max_prompts:
                break
            prompts.append({
                "prompt": item["problem"],
                "expected_answer": item.get("expected_answer", item.get("answer", "")),
                "source": "OpenMathInstruct-2",
            })

        _log.info(f"Loaded {len(prompts)} prompts")
        return prompts

    except Exception as e:
        _log.warning(f"Could not load OpenMathInstruct-2: {e}")
        _log.info("Using fallback prompts")
        return _get_fallback_prompts(max_prompts)


def _get_fallback_prompts(max_prompts: int) -> List[Dict[str, Any]]:
    """Fallback math prompts if dataset unavailable."""
    base_prompts = [
        {"prompt": "What is 15% of 240?", "expected_answer": "36"},
        {"prompt": "Solve for x: 2x + 5 = 13", "expected_answer": "4"},
        {"prompt": "A rectangle has length 8cm and width 5cm. What is its area?", "expected_answer": "40"},
        {"prompt": "What is the sum of the first 10 positive integers?", "expected_answer": "55"},
        {"prompt": "If a train travels 120 miles in 2 hours, what is its average speed?", "expected_answer": "60"},
        {"prompt": "What is 3^4?", "expected_answer": "81"},
        {"prompt": "Simplify: (x^2 * x^3)", "expected_answer": "x^5"},
        {"prompt": "What is the area of a circle with radius 7? (Use pi = 22/7)", "expected_answer": "154"},
        {"prompt": "If 5 workers can complete a job in 12 days, how many days will 10 workers take?", "expected_answer": "6"},
        {"prompt": "What is the LCM of 12 and 18?", "expected_answer": "36"},
    ]

    # Repeat to fill
    prompts = []
    for i in range(max_prompts):
        p = base_prompts[i % len(base_prompts)].copy()
        p["source"] = "fallback"
        prompts.append(p)

    return prompts


class TeacherGenerator:
    """Generate teacher completions using LLM APIs."""

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ):
        self.provider_config = PROVIDERS.get(provider, PROVIDERS["claude"])
        self.model = model or self.provider_config.default_model
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.total_input_tokens = 0
        self.total_output_tokens = 0

        self._init_client()

    def _init_client(self):
        """Initialize the API client."""
        provider = self.provider_config.name

        if provider == "claude":
            try:
                from anthropic import Anthropic
                self.client = Anthropic()
                _log.info(f"Initialized Claude client with model: {self.model}")
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")
            except Exception as e:
                raise RuntimeError(f"Failed to init Claude client: {e}")

        elif provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI()
                _log.info(f"Initialized OpenAI client with model: {self.model}")
            except ImportError:
                raise ImportError("Install openai: pip install openai")
            except Exception as e:
                raise RuntimeError(f"Failed to init OpenAI client: {e}")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate(self, prompt: str) -> Dict[str, Any]:
        """Generate a teacher completion for a prompt."""
        provider = self.provider_config.name

        user_message = f"Solve this problem step by step:\n\n{prompt}"

        try:
            if provider == "claude":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=REASONING_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )

                completion = response.content[0].text
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

            elif provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                )

                completion = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            return {
                "completion": completion,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model": self.model,
            }

        except Exception as e:
            _log.error(f"Generation failed: {e}")
            return {
                "completion": "",
                "error": str(e),
                "input_tokens": 0,
                "output_tokens": 0,
            }

    def get_cost(self) -> float:
        """Get total cost so far."""
        input_cost = (self.total_input_tokens / 1_000_000) * self.provider_config.input_cost_per_m
        output_cost = (self.total_output_tokens / 1_000_000) * self.provider_config.output_cost_per_m
        return input_cost + output_cost

    def estimate_cost(self, num_samples: int, avg_input_tokens: int = 200, avg_output_tokens: int = 500) -> float:
        """Estimate cost for generating N samples."""
        input_cost = (num_samples * avg_input_tokens / 1_000_000) * self.provider_config.input_cost_per_m
        output_cost = (num_samples * avg_output_tokens / 1_000_000) * self.provider_config.output_cost_per_m
        return input_cost + output_cost


def generate_dataset(
    prompts: List[Dict[str, Any]],
    generator: TeacherGenerator,
    output_path: str,
    resume: bool = True,
) -> List[Dict[str, Any]]:
    """Generate teacher completions for all prompts."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing if resuming
    existing = {}
    if resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                item = json.loads(line)
                existing[item["prompt"]] = item
        _log.info(f"Resuming from {len(existing)} existing completions")

    results = []
    start_time = time.time()

    with open(output_path, "a") as f:
        for i, prompt_data in enumerate(prompts):
            prompt = prompt_data["prompt"]

            # Skip if already generated
            if prompt in existing:
                results.append(existing[prompt])
                continue

            # Generate
            gen_result = generator.generate(prompt)

            if gen_result.get("error"):
                _log.warning(f"Skipping prompt {i} due to error: {gen_result['error']}")
                continue

            # Create record
            record = {
                "prompt": prompt,
                "expected_answer": prompt_data.get("expected_answer", ""),
                "teacher_completion": gen_result["completion"],
                "teacher_model": gen_result["model"],
                "input_tokens": gen_result["input_tokens"],
                "output_tokens": gen_result["output_tokens"],
                "source": prompt_data.get("source", "unknown"),
                "generated_at": datetime.now().isoformat(),
            }

            # Write immediately (for resume support)
            f.write(json.dumps(record) + "\n")
            f.flush()

            results.append(record)

            # Progress
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                cost = generator.get_cost()
                rate = (i + 1) / elapsed * 60
                _log.info(
                    f"Progress: {i + 1}/{len(prompts)} | "
                    f"Cost: ${cost:.2f} | "
                    f"Rate: {rate:.1f}/min | "
                    f"Elapsed: {elapsed/60:.1f}min"
                )

            # Rate limiting (be nice to APIs)
            time.sleep(0.1)

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate teacher dataset for distillation")

    parser.add_argument("--provider", type=str, default="claude",
                        choices=list(PROVIDERS.keys()),
                        help="LLM provider")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (uses provider default if not specified)")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data/teacher_completions.jsonl",
                        help="Output file path")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Max tokens per completion")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Generation temperature")
    parser.add_argument("--estimate_cost", action="store_true",
                        help="Only estimate cost, don't generate")
    parser.add_argument("--no_resume", action="store_true",
                        help="Don't resume from existing file")

    args = parser.parse_args()

    # Initialize generator
    generator = TeacherGenerator(
        provider=args.provider,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Cost estimation
    if args.estimate_cost:
        cost = generator.estimate_cost(args.num_samples)
        _log.info(f"\n{'='*60}")
        _log.info(f"COST ESTIMATE")
        _log.info(f"{'='*60}")
        _log.info(f"Provider: {args.provider}")
        _log.info(f"Model: {generator.model}")
        _log.info(f"Samples: {args.num_samples}")
        _log.info(f"Estimated cost: ${cost:.2f}")
        _log.info(f"{'='*60}")
        return

    # Load prompts
    prompts = load_math_prompts(args.num_samples)

    # Generate
    _log.info(f"\n{'='*60}")
    _log.info("GENERATING TEACHER DATASET")
    _log.info(f"{'='*60}")
    _log.info(f"Provider: {args.provider}")
    _log.info(f"Model: {generator.model}")
    _log.info(f"Samples: {len(prompts)}")
    _log.info(f"Output: {args.output}")
    _log.info(f"{'='*60}\n")

    results = generate_dataset(
        prompts=prompts,
        generator=generator,
        output_path=args.output,
        resume=not args.no_resume,
    )

    # Summary
    total_cost = generator.get_cost()
    _log.info(f"\n{'='*60}")
    _log.info("GENERATION COMPLETE")
    _log.info(f"{'='*60}")
    _log.info(f"Samples generated: {len(results)}")
    _log.info(f"Total input tokens: {generator.total_input_tokens:,}")
    _log.info(f"Total output tokens: {generator.total_output_tokens:,}")
    _log.info(f"Total cost: ${total_cost:.2f}")
    _log.info(f"Output file: {args.output}")
    _log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
