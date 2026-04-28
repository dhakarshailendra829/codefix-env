# CodeFix-Env

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)](https://fastapi.tiangolo.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Compatible-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-197%2F197%20Passing-brightgreen)](tests/)
[![Code Quality](https://img.shields.io/badge/Code%20Quality-A-brightgreen)](https://github.com/dhakarshailendra829/codefix-env)

**A production-ready reinforcement learning environment for training and evaluating large language models on automated code debugging tasks.**

CodeFix-Env is a comprehensive platform designed for machine learning engineers and researchers to build, train, and deploy intelligent code debugging agents. Featuring 21+ curated programming tasks, secure sandbox execution, and seamless integration with state-of-the-art LLMs and training frameworks.

---

## 🎯 What is CodeFix-Env?

CodeFix-Env is a **Gymnasium-compatible RL environment** that simulates real-world Python code debugging workflows. It bridges the gap between academic reinforcement learning and practical software engineering by providing:

- **Real debugging scenarios** with authentic Python bugs across multiple complexity levels
- **LLM-ready interface** compatible with HuggingFace Transformers, OpenAI API, and Anthropic Claude
- **Secure code execution** with sandboxed environments and timeout protection
- **Reward shaping** combining multiple signals (test progress, code similarity, completion bonuses)
- **Production infrastructure** including FastAPI server, async client, and CI/CD pipelines

Perfect for researchers training code repair models, companies building AI-powered debugging tools, and developers benchmarking LLM capabilities.

---

## ✨ Key Capabilities

**🧪 Task Library**
- 21 carefully designed debugging tasks spanning 3 difficulty levels (easy, medium, hard)
- Each task includes buggy code, solution, test cases, and expert hints
- Covers real programming patterns: loops, recursion, data structures, algorithms, decorators

**🤖 LLM Integration**
- Pre-built support for HuggingFace Transformers ecosystem
- Compatible with fine-tuning frameworks: SFT, DPO, GRPO, PPO
- Example implementations for Qwen, Llama, CodeLlama, DeepSeek models
- Structured JSON action interface for reliable LLM control

**⚡ Performance & Reliability**
- Multiprocessing-based sandbox with 5-second timeout enforcement
- Per-test execution isolation (no state leakage between test cases)
- Comprehensive test coverage: 197 tests, 83% coverage
- Automated CI/CD with multi-Python version testing (3.10, 3.11, 3.12)

**🔄 Developer Experience**
- Simple Gymnasium interface matching industry standards
- Async client for high-performance training loops
- Sync wrapper for notebooks and scripts
- FastAPI HTTP server for remote access
- Complete type hints and documentation

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/dhakarshailendra829/codefix-env.git
cd codefix-env

# Create environment
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install package
pip install -e .

# For LLM training support
pip install -e ".[llm]"

# For development
pip install -e ".[dev]"
