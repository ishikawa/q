#!/usr/bin/env python
"""
Benchmark script for measuring language model performance metrics:
- TPS (Tokens Per Second)
- Memory Usage
"""

import gc
import json
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import click
import psutil
from tqdm import tqdm

from q.bench.base import BenchTextGeneration

# Sample prompts for evaluation
SAMPLE_PROMPTS = [
    "Explain the concept of machine learning to a 5-year-old.",
    "Write a short poem about artificial intelligence.",
    "What are the key differences between Python and JavaScript?",
    "Describe the process of photosynthesis in plants.",
    "Explain how transformers work in natural language processing.",
    "Write a function in Python to find the nth Fibonacci number.",
    "What are the ethical concerns associated with AI development?",
    "Explain the theory of relativity in simple terms.",
    "Write a short story about a robot discovering emotions.",
    "How does quantum computing differ from classical computing?",
]


@dataclass
class GenerationMetrics:
    """Class to hold metrics for a single text generation."""

    generated_text: str
    tps: float  # Tokens Per Second
    peak_memory: float  # Peak memory usage in MB
    token_count: int  # Number of tokens generated
    total_time: float  # Total generation time in seconds


class PerformanceMetrics:
    """Class to track and calculate model performance metrics."""

    def __init__(self):
        self.tps_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.token_counts: List[int] = []
        self.generation_times: List[float] = []

    def add_sample(
        self,
        tps: float,
        memory_usage: float,
        token_count: int,
        generation_time: float,
    ):
        """Add a performance sample to the metrics."""
        self.tps_samples.append(tps)
        self.memory_samples.append(memory_usage)
        self.token_counts.append(token_count)
        self.generation_times.append(generation_time)


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def generate_with_metrics(
    model: BenchTextGeneration,
    prompt: str,
    max_length: int,
) -> GenerationMetrics:
    """
    Generate text from a prompt and measure performance metrics.

    Returns:
        GenerationMetrics containing metrics for the generation
    """
    # Clear CUDA cache if device is CUDA
    # if device == "cuda" and torch.cuda.is_available():
    #    torch.cuda.empty_cache()

    # Garbage collect to start with a clean slate
    gc.collect()

    # Start generation in a separate thread
    start_time = time.time()
    output = model.generate(
        prompt=prompt,
        max_length=max_length,
    )

    end_time = time.time()

    generated_text = output.generated_text
    token_count = output.token_count
    peak_memory = output.peak_memory

    # Calculate metrics
    total_time = end_time - start_time
    tps = (token_count / total_time) if token_count > 0 and total_time > 0 else 0

    return GenerationMetrics(
        generated_text=generated_text,
        tps=tps,
        peak_memory=peak_memory,
        token_count=token_count,
        total_time=total_time,
    )


def run_benchmark(
    model_name: str,
    max_length: int,
    output_dir: Optional[str] = None,
    num_runs: int = 3,
) -> Dict[str, Any]:
    """
    Run performance benchmark on a model.

    Args:
        model_name: HuggingFace model identifier
        prompts: List of prompts to test (uses defaults if None)
        output_dir: Directory to save results
        max_length: Maximum sequence length for generation
        device: Device to run on ('cuda', 'mps', 'cpu')
        num_runs: Number of times to run each prompt

    Returns:
        Dictionary with benchmark results
    """
    prompts = SAMPLE_PROMPTS

    print(f"Loading model: {model_name}, max_length: {max_length} ===")

    if model_name == "q":
        from q.bench.q import QBenchTextGeneration

        generation = QBenchTextGeneration()
    else:
        from q.bench.hf import HFBenchTextGeneration

        generation = HFBenchTextGeneration(model_name=model_name)

    # Ensure output directory exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Setup metrics tracker
    metrics = PerformanceMetrics()
    total_peak_memory = 0
    max_peak_memory = 0

    # Storage for generated outputs
    generations = []

    # Run benchmark
    for prompt_idx, prompt in enumerate(prompts):
        prompt_metrics = PerformanceMetrics()

        for run in tqdm(range(num_runs), desc=f"Prompt {prompt_idx+1}/{len(prompts)}"):
            try:
                generation_metrics = generate_with_metrics(
                    model=generation,
                    prompt=prompt,
                    max_length=max_length,
                )

                # Track peak memory
                total_peak_memory += generation_metrics.peak_memory
                max_peak_memory = max(max_peak_memory, generation_metrics.peak_memory)

                # Add to metrics
                metrics.add_sample(
                    generation_metrics.tps,
                    generation_metrics.peak_memory,
                    generation_metrics.token_count,
                    generation_metrics.total_time,
                )
                prompt_metrics.add_sample(
                    generation_metrics.tps,
                    generation_metrics.peak_memory,
                    generation_metrics.token_count,
                    generation_metrics.total_time,
                )

                if output_dir:
                    generations.append(
                        {
                            "prompt": prompt,
                            "generated_text": generation_metrics.generated_text,
                            "prompt_idx": prompt_idx,
                            "run": run,
                            "metrics": {
                                "tps": generation_metrics.tps,
                                "peak_memory_mb": generation_metrics.peak_memory,
                                "token_count": generation_metrics.token_count,
                                "total_time": generation_metrics.total_time,
                            },
                        }
                    )

            except Exception as e:
                print(f"Error during generation: {e}")
                raise

        # Print prompt summary
        tps_mean = (
            statistics.mean(prompt_metrics.tps_samples)
            if prompt_metrics.tps_samples
            else 0
        )
        memory_mean = (
            statistics.mean(prompt_metrics.memory_samples)
            if prompt_metrics.memory_samples
            else 0
        )
        memory_max = (
            max(prompt_metrics.memory_samples) if prompt_metrics.memory_samples else 0
        )

        print(f"\nPrompt {prompt_idx+1} Summary:")
        print(
            f"  TPS: {tps_mean:.2f} tokens/sec (median: {statistics.median(prompt_metrics.tps_samples):.2f})"
        )
        print(f"  Memory Usage: {memory_mean:.2f}MB (peak: {memory_max:.2f}MB)")

    # Calculate statistics
    final_stats = {
        "tps": {
            "mean": statistics.mean(metrics.tps_samples) if metrics.tps_samples else 0,
            "median": (
                statistics.median(metrics.tps_samples) if metrics.tps_samples else 0
            ),
        },
        "memory_usage_mb": {
            "mean": (
                statistics.mean(metrics.memory_samples) if metrics.memory_samples else 0
            ),
            "max": max(metrics.memory_samples) if metrics.memory_samples else 0,
        },
        "token_counts": {
            "total": sum(metrics.token_counts),
            "mean": (
                statistics.mean(metrics.token_counts) if metrics.token_counts else 0
            ),
        },
        "generation_times": {
            "total": sum(metrics.generation_times),
            "mean": (
                statistics.mean(metrics.generation_times)
                if metrics.generation_times
                else 0
            ),
        },
    }

    avg_peak_memory = (
        total_peak_memory / (len(prompts) * num_runs)
        if len(prompts) * num_runs > 0
        else 0
    )

    # Create benchmark results
    results = {
        "model": model_name,
        "device": "mps",
        "parameters": {
            "max_length": max_length,
            "temperature": 1.0,
            "num_runs": num_runs,
            "num_prompts": len(prompts),
        },
        "metrics": final_stats,
        "peak_memory": {
            "average_mb": avg_peak_memory,
            "max_mb": max_peak_memory,
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_shortname = model_name.split("/")[-1] if model_name else "q"

        # Save summary results
        results_path = os.path.join(
            output_dir, f"bench_{model_shortname}_{max_length}_{timestamp}.json"
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save generations if requested
        if generations:
            generations_path = os.path.join(
                output_dir,
                f"bench_{model_shortname}_{max_length}_generations_{timestamp}.json",
            )
            with open(generations_path, "w") as f:
                json.dump(generations, f, indent=2)

        print(f"Results saved to {results_path}")

    # Print summary
    print("\nSummary:")
    print(f"  Average TPS: {final_stats['tps']['mean']:.2f} tokens/sec")
    print(f"  Average Total Memory: {avg_peak_memory:.2f}MB")
    print(f"  Maximum Total Memory: {max_peak_memory:.2f}MB")
    print(f"  Total Tokens Generated: {final_stats['token_counts']['total']}")
    print(f"  Total Generation Time: {final_stats['generation_times']['total']:.2f}s")

    return results


@click.command()
@click.option(
    "--model",
    "model_name",
    required=True,
    default="q",
    help='"q" or HuggingFace model name',
)
@click.option(
    "--output-path", "output_path", required=False, help="Directory to save results"
)
@click.option(
    "--max-length",
    "max_length",
    default=64,
    type=int,
    help="Maximum sequence length for generation",
)
@click.option(
    "--num-runs", "num_runs", default=3, type=int, help="Number of runs per prompt"
)
def main(
    model_name: str,
    max_length: int,
    num_runs: int,
    output_path: Optional[str] = None,
):
    """Benchmark a transformer model's performance metrics."""

    # Run benchmark
    run_benchmark(
        model_name=model_name,
        output_dir=output_path,
        max_length=max_length,
        num_runs=num_runs,
    )


if __name__ == "__main__":
    main()
