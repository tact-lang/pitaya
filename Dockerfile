FROM node:24

# Install basic development tools
RUN apt update && apt install -y \
    git \
    curl \
    build-essential \
    python3 \
    python3-pip \
    procps \
    sudo \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory with proper permissions for node user
RUN mkdir -p /workspace && \
    chown -R node:node /workspace

# Set up npm global directory for node user (already exists but ensure permissions)
RUN mkdir -p /home/node/.npm-global && \
    chown -R node:node /home/node/.npm-global

# Switch to node user
USER node
WORKDIR /workspace

# Configure npm to use the global directory
ENV NPM_CONFIG_PREFIX=/home/node/.npm-global
ENV PATH=$PATH:/home/node/.npm-global/bin

# Install Claude Code CLI as claude user
RUN npm install -g @anthropic-ai/claude-code

# Default command
CMD ["claude", "--help"]