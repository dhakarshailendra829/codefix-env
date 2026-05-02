# Changelog

All notable changes to CodeFix-Env are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [0.2.0] — 2026-04-22

### Added
- Full `src/` layout with proper Python packaging
- Pydantic v2 models: `CodeFixAction`, `CodeFixObservation`, `CodeFixState`
- 21 tasks across easy (8), medium (8), hard (5) difficulties
- 11 bug categories: syntax, logic, off-by-one, type, algorithm, scope, etc.
- `CodeFixClient` with async + sync (`.sync()`) support
- `RewardPipeline` blending rule-based + optional neural reward model
- `RewardMLP` — PyTorch neural reward model (8-feature input)
- `SessionManager` for concurrent multi-client server support
- Sandbox executor with subprocess isolation and security policies
- GitHub Actions CI: lint, test (Py 3.10/3.11/3.12), Docker build, type check
- Docker multi-stage production image with non-root user
- Benchmark runner with `RandomAgent`, `GreedyAgent`, `OracleAgent`
- GRPO training example with TRL
- Full API docs and quickstart guide
- OpenEnv YAML manifest

### Changed
- Removed committed `.venv/` directory
- Restructured server into `server/` module
- Separated environment logic from FastAPI app

### Fixed
- Session handling: `X-Session-ID` header pattern
- Reward normalisation clipped to [0, 1]
- Sandbox import whitelist updated to include all stdlib math/collections

---

## [0.1.0] — 2026-03-15

### Added
- Initial MVP: 3 tasks (easy, medium, hard)
- Basic FastAPI server
- `CodeFixAction` with `run_tests`, `edit_line`, `submit_fix`
- Docker deployment
- OpenEnv YAML config

### Features
- Multi-factor reward signals
- Sandboxed code execution with timeout protection
- Support for SFT, DPO, GRPO, PPO training frameworks
