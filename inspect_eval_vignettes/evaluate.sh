#!/usr/bin/env bash


# These 509 requests use around 2.25m input tokens and 250k output tokens (OpenAI). This is increased by a factor of around 1.2 in the Google models and Grok.
MODELS=(
    "openai/gpt-4o-mini-2024-07-18" # $1.04 for this and gpt-4.1-mini
    "openai/gpt-4.1-mini-2025-04-14"
    "google/gemini-2.5-flash-lite" # bill was £0.42 before
    "google/gemini-1.5-flash"
    "anthropic/claude-3-haiku-20240307" # cost $0.86
    "grok/grok-3-mini" # bill $0.93
)
MAX_TOKENS=4000
LOG_FORMAT=json

for model in "${MODELS[@]}"; do
    inspect eval 1__evaluate.py --model "$model" --max-tokens $MAX_TOKENS --log-format $LOG_FORMAT
done

# Calculate cosine similarity
python 2__cosine_similarity.py

# Run GLM models
Rscript 3__run_glm_models.R