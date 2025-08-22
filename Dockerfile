FROM node:24

# Install only essential tools used by agents and runner
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    jq \
    bash \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create workspace and ensure the non-root user owns it
RUN mkdir -p /workspace && chown -R node:node /workspace

# Install both agent CLIs globally so they are available regardless of volume mounts
RUN npm install -g @anthropic-ai/claude-code @openai/codex \
    && npm cache clean --force

# Switch to the non-root user expected by the runner and set defaults
USER node
WORKDIR /workspace

# Keep the container alive if run interactively; the runner execs specific CLIs
CMD ["bash"]
