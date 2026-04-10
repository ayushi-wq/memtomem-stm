# Contributing to memtomem-stm

Thank you for your interest in contributing to memtomem-stm!

## Development Setup

```bash
# Clone
git clone https://github.com/memtomem/memtomem-stm.git
cd memtomem-stm

# Install (requires Python 3.12+ and uv)
uv sync

# Run tests
uv run pytest -m "not ollama"          # skip Ollama-dependent tests
uv run pytest                          # full suite (requires running Ollama)

# Lint and format
uv run ruff check src --fix
uv run ruff format src

# Type check
uv run mypy src
```

## Project Structure

- `src/memtomem_stm/` — Core: MCP server, proxy pipeline, compression, surfacing, caching, observability
  - `proxy/` — 4-stage pipeline (CLEAN → COMPRESS → SURFACE → INDEX), privacy scanning
  - `surfacing/` — Memory surfacing engine and relevance gating
  - `observability/` — Langfuse tracing and metrics
  - `cli/` — `mms` / `memtomem-stm-proxy` CLI
  - `utils/` — Circuit breaker and shared helpers
- `tests/` — pytest suite
- `docs/` — Architecture and operations guides

The LTM core lives in a separate repository: [memtomem/memtomem](https://github.com/memtomem/memtomem). Communication between STM and LTM happens entirely through the MCP protocol — there is no Python-level dependency.

## Pull Request Guidelines

1. Create a feature branch from `main`
2. Keep changes focused — one feature or fix per PR
3. Add tests for new functionality
4. Ensure `uv run ruff check src` and `uv run ruff format --check src` pass
5. Ensure `uv run pytest -m "not ollama"` passes
6. `uv run mypy src` is advisory but aim to not introduce new errors
7. Write a clear commit message describing the "why"

## Reporting Issues

Open an issue at https://github.com/memtomem/memtomem-stm/issues with:
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, Python version, memtomem-stm version, upstream MCP server versions)
- Relevant config (`stm_proxy.json` or `mms status` output, with secrets redacted)
