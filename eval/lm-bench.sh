#!/bin/bash
set -x

poetry run python -m q.bench.cli
poetry run python -m q.bench.cli --pretrained gpt2
