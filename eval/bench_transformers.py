#!/usr/bin/env python
"""
Benchmark script for measuring language model performance metrics:
- TTFT (Time to First Token)
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
from threading import Thread
from typing import Any, Dict, List

import click
import psutil
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

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
    ttft: float  # Time to First Token in seconds
    tps: float  # Tokens Per Second
    memory_increase: float  # Memory increase in MB
    peak_memory: float  # Peak memory usage in MB
    token_count: int  # Number of tokens generated
    total_time: float  # Total generation time in seconds


class PerformanceMetrics:
    """Class to track and calculate model performance metrics."""

    def __init__(self):
        self.ttft_samples: List[float] = []
        self.tps_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.token_counts: List[int] = []
        self.generation_times: List[float] = []

    def add_sample(
        self,
        ttft: float,
        tps: float,
        memory_usage: float,
        token_count: int,
        generation_time: float,
    ):
        """Add a performance sample to the metrics."""
        self.ttft_samples.append(ttft)
        self.tps_samples.append(tps)
        self.memory_samples.append(memory_usage)
        self.token_counts.append(token_count)
        self.generation_times.append(generation_time)

    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate statistics for all metrics."""
        return {
            "ttft": {
                "mean": statistics.mean(self.ttft_samples),
                "median": statistics.median(self.ttft_samples),
                "min": min(self.ttft_samples),
                "max": max(self.ttft_samples),
                "samples": self.ttft_samples,
            },
            "tps": {
                "mean": statistics.mean(self.tps_samples),
                "median": statistics.median(self.tps_samples),
                "min": min(self.tps_samples),
                "max": max(self.tps_samples),
                "samples": self.tps_samples,
            },
            "memory_usage_mb": {
                "mean": statistics.mean(self.memory_samples),
                "median": statistics.median(self.memory_samples),
                "min": min(self.memory_samples),
                "max": max(self.memory_samples),
                "samples": self.memory_samples,
            },
            "token_counts": {
                "mean": statistics.mean(self.token_counts),
                "total": sum(self.token_counts),
                "samples": self.token_counts,
            },
            "generation_times": {
                "mean": statistics.mean(self.generation_times),
                "total": sum(self.generation_times),
                "samples": self.generation_times,
            },
        }


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    print(f"memory_info: {memory_info}")
    return process.memory_info().rss / (1024 * 1024)


def generate_with_metrics(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    device: str = "mps",
) -> GenerationMetrics:
    """
    Generate text from a prompt and measure performance metrics.

    Returns:
        GenerationMetrics containing metrics for the generation
    """
    # Clear CUDA cache if device is CUDA
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Garbage collect to start with a clean slate
    gc.collect()

    # Record initial memory
    initial_memory = get_memory_usage()
    peak_memory = initial_memory

    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Set up streamer for monitoring token generation
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    # Prepare generation parameters
    generation_kwargs = {
        "input_ids": input_ids,
        "max_length": max_length,
        "temperature": 1.0,
        "streamer": streamer,
    }

    # Start generation in a separate thread
    start_time = time.time()
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Monitor token generation
    first_token_time = None
    generated_text = ""
    token_count = 0

    for new_text in streamer:
        if first_token_time is None:
            first_token_time = time.time()

        # Check current memory after each token for peak tracking
        current_memory = get_memory_usage()
        peak_memory = max(peak_memory, current_memory)

        generated_text += new_text
        token_count += 1

    # Wait for the thread to complete
    thread.join()
    end_time = time.time()

    # One final memory check after thread completes
    current_memory = get_memory_usage()
    peak_memory = max(peak_memory, current_memory)

    # Calculate metrics
    ttft = first_token_time - start_time if first_token_time else 0
    total_time = end_time - start_time
    tps = (
        token_count / (total_time - ttft)
        if token_count > 0 and (total_time - ttft) > 0
        else 0
    )

    # Calculate memory increase from baseline to peak
    memory_increase = peak_memory - initial_memory

    return GenerationMetrics(
        generated_text=generated_text,
        ttft=ttft,
        tps=tps,
        memory_increase=memory_increase,
        peak_memory=peak_memory,
        token_count=token_count,
        total_time=total_time,
    )


def run_benchmark(
    model_name: str,
    output_dir: str = "eval/outputs",
    max_length: int = 100,
    device: str = "mps",
    num_runs: int = 1,
    save_generations: bool = True,
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
        save_generations: Whether to save generated text

    Returns:
        Dictionary with benchmark results
    """
    prompts = SAMPLE_PROMPTS

    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
        device_map=device,
    )

    # Ensure output directory exists
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
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_length=max_length,
                    device=device,
                )
                print("Peak memory usage:", generation_metrics.peak_memory)

                # Track peak memory
                total_peak_memory += generation_metrics.peak_memory
                max_peak_memory = max(max_peak_memory, generation_metrics.peak_memory)

                # Add to metrics
                metrics.add_sample(
                    generation_metrics.ttft,
                    generation_metrics.tps,
                    generation_metrics.memory_increase,
                    generation_metrics.token_count,
                    generation_metrics.total_time,
                )
                prompt_metrics.add_sample(
                    generation_metrics.ttft,
                    generation_metrics.tps,
                    generation_metrics.memory_increase,
                    generation_metrics.token_count,
                    generation_metrics.total_time,
                )

                if save_generations:
                    generations.append(
                        {
                            "prompt": prompt,
                            "generated_text": generation_metrics.generated_text,
                            "prompt_idx": prompt_idx,
                            "run": run,
                            "metrics": {
                                "ttft": generation_metrics.ttft,
                                "tps": generation_metrics.tps,
                                "memory_usage_mb": generation_metrics.memory_increase,
                                "peak_memory_mb": generation_metrics.peak_memory,
                                "token_count": generation_metrics.token_count,
                                "total_time": generation_metrics.total_time,
                            },
                        }
                    )

                # Give GPU some time to cool down between runs if using CUDA
                if device == "cuda":
                    time.sleep(0.5)

            except Exception as e:
                print(f"Error during generation: {e}")
                continue

        # Print prompt summary
        prompt_stats = prompt_metrics.calculate_statistics()
        print(f"\nPrompt {prompt_idx+1} Summary:")
        print(
            f"  TTFT: {prompt_stats['ttft']['mean']:.4f}s (median: {prompt_stats['ttft']['median']:.4f}s)"
        )
        print(
            f"  TPS: {prompt_stats['tps']['mean']:.2f} tokens/sec (median: {prompt_stats['tps']['median']:.2f})"
        )
        print(
            f"  Memory Increase: {prompt_stats['memory_usage_mb']['mean']:.2f}MB (peak increase: {prompt_stats['memory_usage_mb']['max']:.2f}MB)"
        )

    # Calculate statistics
    final_stats = metrics.calculate_statistics()
    avg_peak_memory = (
        total_peak_memory / (len(prompts) * num_runs)
        if len(prompts) * num_runs > 0
        else 0
    )

    # Create benchmark results
    results = {
        "model": model_name,
        "device": device,
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_shortname = model_name.split("/")[-1]

    # Save summary results
    results_path = os.path.join(output_dir, f"bench_{model_shortname}_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save generations if requested
    if save_generations and generations:
        generations_path = os.path.join(
            output_dir, f"bench_{model_shortname}_generations_{timestamp}.json"
        )
        with open(generations_path, "w") as f:
            json.dump(generations, f, indent=2)

    print(f"\nBenchmark complete! Results saved to {results_path}")

    # Print summary
    print("\nSummary:")
    print(f"  Average TTFT: {final_stats['ttft']['mean']:.4f}s")
    print(f"  Average TPS: {final_stats['tps']['mean']:.2f} tokens/sec")
    print(f"  Average Memory Increase: {final_stats['memory_usage_mb']['mean']:.2f}MB")
    print(f"  Peak Memory Increase: {final_stats['memory_usage_mb']['max']:.2f}MB")
    print(f"  Average Total Memory: {avg_peak_memory:.2f}MB")
    print(f"  Maximum Total Memory: {max_peak_memory:.2f}MB")
    print(f"  Total Tokens Generated: {final_stats['token_counts']['total']}")
    print(f"  Total Generation Time: {final_stats['generation_times']['total']:.2f}s")

    return results


@click.command()
@click.option("--pretrained", required=True, help="HuggingFace model name or path")
@click.option("--output-path", default="eval/outputs", help="Directory to save results")
@click.option(
    "--max-length",
    default=100,
    type=int,
    help="Maximum sequence length for generation",
)
@click.option(
    "--device", default="mps", type=str, help="Device to run on (cuda, mps, cpu)"
)
@click.option("--num-runs", default=3, type=int, help="Number of runs per prompt")
@click.option("--no-save-generations", is_flag=True, help="Don't save generated text")
def main(
    pretrained: str,
    output_path: str,
    max_length: int,
    device: str,
    num_runs: int,
    no_save_generations: bool,
):
    """Benchmark a transformer model's performance metrics."""
    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"

    print(f"Running on device: {device}")

    # Run benchmark
    run_benchmark(
        model_name=pretrained,
        output_dir=output_path,
        max_length=max_length,
        device=device,
        num_runs=num_runs,
        save_generations=not no_save_generations,
    )


if __name__ == "__main__":
    main()
