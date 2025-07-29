#!/bin/sh
set -e

# Run training with provided arguments
python train.py "$@"

# Start the OpenAI-compatible API server to serve the trained model
export MODEL_PATH=/app/model_output
uvicorn server.api_server:app --host 0.0.0.0 --port 8000
