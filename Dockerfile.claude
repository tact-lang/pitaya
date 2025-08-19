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
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory with proper permissions for node user
RUN mkdir -p /workspace && \
    chown -R node:node /workspace

# Install Claude Code CLI globally as root so it survives /home/node volume mounts
RUN npm install -g @anthropic-ai/claude-code

# Set up npm global directory for node user (runtime installs, if any)
RUN mkdir -p /home/node/.npm-global && \
    chown -R node:node /home/node/.npm-global

# Switch to node user
USER node
WORKDIR /workspace

# Configure npm to use the user global directory (not used for preinstalled CLI)
ENV NPM_CONFIG_PREFIX=/home/node/.npm-global
ENV PATH=$PATH:/home/node/.npm-global/bin

# Configure AppImage to extract and run
ENV APPIMAGE_EXTRACT_AND_RUN=1
ENV TMPDIR=/workspace

# Default command
CMD ["claude", "--help"]
