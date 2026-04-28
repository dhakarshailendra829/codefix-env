# CodeFix-Env

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)](https://fastapi.tiangolo.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Compatible-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-197%2F197%20Passing-brightgreen)](tests/)

A production-ready reinforcement learning environment for training and evaluating large language models on automated code debugging tasks.

## 📖 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [What It Does](#what-it-does)
- [Task Categories](#task-categories)
- [Usage Examples](#usage-examples)
- [LLM Training](#llm-training)
- [API Reference](#api-reference)
- [CI/CD Pipeline](#cicd-pipeline)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

---

## 📌 Overview

CodeFix-Env is a **Gymnasium-compatible RL environment** that simulates real-world Python code debugging workflows. It enables researchers and engineers to:

- Train large language models (LLMs) on code repair tasks
- Evaluate AI agents on real debugging scenarios
- Benchmark code generation and reasoning capabilities
- Build automated debugging systems

Perfect for anyone working with LLMs like Qwen, Llama, CodeLlama, or DeepSeek models.

---

## ✨ Key Features

**🧪 Task Library**
- 21 carefully designed debugging tasks
- 3 difficulty levels: Easy (8), Medium (8), Hard (5)
- Each task includes buggy code, solution, test cases, and hints
- Covers real Python patterns: loops, recursion, algorithms, decorators

**🤖 LLM Integration**
- Works with HuggingFace Transformers ecosystem
- Compatible with training frameworks: SFT, DPO, GRPO, PPO
- Pre-built support for Qwen, Llama, CodeLlama, DeepSeek models
- Structured JSON action interface for LLM control

**⚡ Performance**
- Multiprocessing-based sandbox with timeout enforcement
- Per-test execution isolation
- 197 unit tests with 83% code coverage
- Automated CI/CD pipeline (Python 3.10, 3.11, 3.12)

**🔄 Developer Friendly**
- Gymnasium standard interface
- Async client for high-performance training
- Sync wrapper for notebooks
- FastAPI HTTP server for remote access
- Complete type hints and documentation

---

## 💻 Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/dhakarshailendra829/codefix-env.git
cd codefix-env
```
### Step 2: Create virual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate (On mac)
.venv/Scripts/activate (On Windows)
```
### Step 3: Intall Package
# For Basic Usage
```bash
pip install -e . 
```
--
# For Developement
```bash
pip install -e ".[dev]"
```
---
# For LLM training (includes transformers, TRL, datasets)
```bash
pip install -e ".[llm]"
```
---
# For everything
```bash
pip install -e ".[dev,llm]"
```
### Step 4: Verify Installation
```bash
python -c "from codefix_env import CodeFixEnvironment; print(' Installation successful!')"
```
```bash
```
```bash
```
