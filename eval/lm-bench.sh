#!/bin/bash
set -x

OUTPUT_PATH="eval/outputs/$(date '+%Y%m%d_%H%M%S')"

for max_length in 64 128 256
do
    echo "Running with max length $max_length"
    poetry run python -m q.bench.cli --output-path="$OUTPUT_PATH" --max-length="$max_length"
    poetry run python -m q.bench.cli --output-path="$OUTPUT_PATH" --model gpt2 --max-length="$max_length"
done
