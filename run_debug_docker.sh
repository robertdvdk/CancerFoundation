#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="cancerfoundation"

# Build the image
docker build -t "$IMAGE_NAME" -f Dockerfile .

# Run debug.sh inside the container
docker run --rm \
    --gpus all \
    --shm-size 32g \
    --env-file .devcontainer/devcontainer.env \
    -v "$(pwd)":/workspace \
    -w /workspace \
    "$IMAGE_NAME" \
    bash -c "uv pip install --system --no-deps -e . && bash debug.sh"
