FROM nvcr.io/nvidia/clara/bionemo-framework:2.7.1

RUN apt-get update && apt-get install -y --no-install-recommends tmux \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install project dependencies (cached unless pyproject.toml changes).
# Uses a stub package so uv can resolve without the full source tree.
COPY pyproject.toml /tmp/cancerfoundation/
RUN mkdir -p /tmp/cancerfoundation/cancerfoundation \
    && touch /tmp/cancerfoundation/cancerfoundation/__init__.py \
    && cd /tmp/cancerfoundation \
    && uv pip install --system ".[dev]" \
    && rm -rf /tmp/cancerfoundation
