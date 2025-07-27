# 🎯 Orchestrator

<div align="center">

**Scale AI coding horizontally**

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## ⚠️ WARNING: ACTIVE DEVELOPMENT

> **This project is under heavy active development and is NOT recommended for production use or as an external dependency!**
>
> -   APIs and interfaces are subject to breaking changes without notice
> -   Features may be incomplete, unstable, or experimental
> -   Documentation may be outdated or missing
> -   Use at your own risk for experimentation only

---

## 🚀 The Problem

When using AI for software development, **outcomes vary significantly between runs** - even with identical prompts. Sometimes you get elegant solutions, sometimes overcomplicated ones, sometimes the AI misunderstands entirely.

**Orchestrator turns this variability into a strength.**

## 💡 The Solution

Instead of running one AI coding instance and hoping for the best, Orchestrator runs **multiple instances in parallel**, each exploring different solution paths. Then it implements intelligent strategies to **automatically identify and select the best results**.

```bash
# Instead of this (hoping for a good result):
claude "implement authentication"

# Do this (explore 5 paths, pick the best):
uv run orchestrator "implement authentication" --strategy best-of-n -S n=5
```

## ✨ Features

### 🔄 **Parallel Exploration**

-   Run N instances of Claude Code simultaneously
-   Each instance works in complete isolation with its own Docker container
-   Every solution gets its own git branch for easy comparison

### 🧠 **Intelligent Selection**

-   **Best-of-N**: Generate multiple solutions, score each one, select the highest rated
-   **Iterative Refinement**: Generate, review, improve in cycles
-   **Custom Strategies**: Build your own multi-stage workflows

### 📊 **Real-Time Monitoring**

-   Beautiful TUI dashboard showing all instances
-   Live progress tracking with cost accumulation
-   Adaptive display that scales to dozens of instances

### 🔧 **Development Features**

-   **Resumable**: Interrupt anytime with Ctrl+C, resume where you left off
-   **Fault Tolerant**: Automatic retries for transient failures
-   **Cost Aware**: Real-time token usage and cost tracking

### 🏗️ **Clean Architecture**

-   Three-layer design with clear separation of concerns
-   Event-driven communication between components
-   Extensible plugin system for different AI tools

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.
