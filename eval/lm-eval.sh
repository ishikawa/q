#!/bin/bash
set -x

OUTPUT_PATH="eval/outputs/$(date '+%Y%m%d_%H%M%S')"
TASKS="hellaswag,mmlu"
DEVICE="mps"
BATCH_SIZE=2

# q (124M)
poetry run eval --model q \
    --model_args model_size=124M \
    --output_path $OUTPUT_PATH \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE

# q (355M)
# poetry run eval --model q \
#     --model_args model_size=355M \
#     --output_path $OUTPUT_PATH \
#     --tasks $TASKS \
#     --batch_size $BATCH_SIZE

# GPT-2 (124M)
poetry run lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --output_path $OUTPUT_PATH \
    --tasks $TASKS \
    --device $DEVICE \
    --batch_size $BATCH_SIZE

# GPT-2 (355M)
# poetry run lm_eval --model hf \
#     --model_args pretrained=gpt2-medium \
#     --output_path $OUTPUT_PATH \
#     --tasks $TASKS \
#     --device $DEVICE \
#     --batch_size $BATCH_SIZE

# Qwen/Qwen2.5-0.5B
# https://huggingface.co/Qwen/Qwen2.5-0.5B
poetry run lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-0.5B \
    --output_path $OUTPUT_PATH \
    --tasks $TASKS \
    --device $DEVICE \
    --batch_size $BATCH_SIZE
