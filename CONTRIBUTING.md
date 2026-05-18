# Contributing to NOESIS PRIME

Thank you for your interest in contributing to **NOESIS PRIME**.

This repository is currently a **research-oriented cognitive architecture prototype**. Contributions are welcome, especially those that improve clarity, reliability, modularity, and reproducibility.

---

## Ways to Contribute

You can help by contributing in any of these areas:

- bug fixes
- documentation improvements
- code cleanup and refactoring
- modularization of the single-file implementation
- test coverage
- performance improvements
- API improvements
- experiment scripts and benchmarks
- architecture diagrams and examples

---

## Before You Start

Please:

1. Read the `README.md` first.
2. Open an issue if you plan to make a large change.
3. Keep pull requests focused and reasonably small.
4. Prefer changes that improve correctness, readability, or reproducibility.

---

## Development Guidelines

### Code style

- Use clear, descriptive names.
- Preserve the architecture-oriented structure and comments where possible.
- Avoid unnecessary rewrites of unrelated sections.
- Keep public behavior backward compatible unless the change is intentional and documented.

### Documentation

If you add or change functionality, also update:

- `README.md`
- inline comments/docstrings
- API examples if endpoint behavior changes

### Testing

There is currently no formal test suite in the repository.

If you contribute logic changes, please include at least one of the following:

- a minimal reproducible example
- a CLI/API usage example
- a proposed test file or test case

---

## Suggested Contribution Areas

High-value contributions include:

- splitting `noesis_prime.py` into modules
- adding `tests/`
- creating `pyproject.toml`
- improving dependency management
- adding Docker support
- adding architecture visualizations
- improving API validation and schemas
- benchmarking memory retrieval and planning behavior

---

## Pull Request Checklist

Before opening a PR, please confirm:

- [ ] the code runs locally
- [ ] the change is scoped and explained clearly
- [ ] documentation is updated if needed
- [ ] no unrelated files were modified unnecessarily
- [ ] environment or dependency changes are documented

---

## Philosophy

NOESIS PRIME is not just an ordinary software project — it is also an exploration of memory, narrative, cognition, and identity in artificial systems.

Please try to preserve:

- conceptual clarity
- architectural coherence
- scientific seriousness
- readable implementation

---

## Questions

If you are unsure whether a contribution fits the direction of the project, open an issue first and describe your proposal.
