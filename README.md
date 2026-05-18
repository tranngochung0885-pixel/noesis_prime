# NOESIS PRIME v1.0 – Narrative-Oriented Emergent Intelligence System

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
![Architecture](https://img.shields.io/badge/architecture-single--file%20system-purple)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-success)

**A unified neuro-inspired cognitive architecture where memory is a living graph, thought is a narrative, and the self is a continuous process.**

NOESIS PRIME is a research-oriented cognitive architecture that integrates episodic memory, predictive coding, neuromodulation, planning, identity continuity, and adaptive LLM routing into a single agentic system. It evolves through a continuous cognitive cycle rather than a sequence of isolated modules.

The core insight: **human cognition is not merely data processing — it is a recursive narrative engine.** Every perception is interpreted through past experience, future expectation, and an evolving self-model.

---

## Table of Contents

- [Overview](#overview)
- [Project Status](#project-status)
- [Key Features](#-key-features)
- [Use Cases](#-use-cases)
- [Scientific Grounding](#-scientific-grounding)
- [Requirements](#-requirements)
- [Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Run Demonstration](#run-demonstration)
  - [Interactive CLI](#interactive-cli)
  - [REST API Server](#rest-api-server)
- [API Overview](#-api-overview)
- [Project Structure](#-project-structure)
- [Architecture Overview](#-architecture-overview)
- [Configuration](#️-configuration)
- [Telemetry & Introspection](#-telemetry--introspection)
- [Development](#-development)
- [Extending NOESIS](#-extending-noesis)
- [Limitations](#-limitations)
- [Roadmap Ideas](#-roadmap-ideas)
- [Citation](#-citation)
- [Author](#-author)
- [License](#-license)

---

## Overview

NOESIS PRIME is the culmination of multiple generations of cognitive architecture ideas (AXIOM → NEXUS → HELIOS → MNEMOS → NOESIS). The repository currently implements the full system in a single Python file, with a unified internal state and tightly integrated subsystems.

At a high level, the architecture combines:

- **Temporal Narrative Graph (TNG)** for episodic memory
- **Hierarchical predictive coding** for world modeling
- **Neuromodulatory dynamics** for cognitive state control
- **MCTS + actor-critic planning** for action selection
- **Identity continuity** for persistent self-modeling
- **Adaptive LLM routing** across local and cloud backends
- **CLI + REST API** interfaces for experimentation
- **SQLite-backed persistence** for long-term state and memory

This repository is best understood as a **research prototype**, **architectural exploration**, and **technical demonstration** of a narrative-first model of machine cognition.

---

## Project Status

**Status:** Active research prototype  
**Version:** 1.0  
**Implementation style:** Single-file reference implementation  
**Primary language:** Python  
**Interface modes:** Demo, CLI, REST API  
**Project maturity:** Early open-source / research-grade foundation

This project is suitable for:
- experimentation
- architecture study
- cognitive systems prototyping
- memory/planning/self-model research
- inspiration for modular multi-agent or embodied-agent systems

It is **not yet positioned as a production framework**. The current implementation prioritizes conceptual completeness and integrative design over full modular decomposition and production hardening.

---

## ✨ Key Features

- **Temporal Narrative Graph (TNG)** – Episodic memories stored as nodes in a causal-temporal graph, not a flat list. Edges include `TEMPORAL`, `CAUSAL`, `CONTRAST`, `REFLECTS`, `ASSOCIATIVE`.
- **4-Level Hierarchical World Model** – Predictive coding across four levels: sensory (L0), conceptual (L1), narrative (L2), self-model (L3).
- **Reconsolidation** – Retrieved memories enter a 2-hour window during which they can be modified.
- **Nostalgia Engine** – Deliberate replay of past engrams into a coherent narrative journey; extracts insights via spreading activation.
- **Neuromodulatory Fabric** – DA/NE/ACh/5HT/GABA/GLU with metaplasticity tracking.
- **MCTS Planner** – Monte Carlo Tree Search (PUCT) with learned priors from the actor-critic.
- **Identity Continuity Layer** – Persistent self-model (personality, core values, formative events) saved across sessions.
- **Meta-Cognitive Monitor** – Confidence calibration, uncertainty estimation, dynamic MCTS budget, performance alerts.
- **Adaptive LLM Bridge** – Intelligent routing: local GGUF for fast operations, cloud Anthropic/OpenAI for deeper reasoning, with fallback chain.
- **Unified Internal State (IST)** – All modules read from and propose updates to a single internal state tensor.
- **FastAPI Interface** – REST endpoints and WebSocket chat for interactive experimentation.
- **Persistence Layer** – SQLite-backed episodic and semantic memory persistence.

---

## 🎯 Use Cases

NOESIS PRIME can be used as a foundation or reference for:

- **Cognitive architecture experiments**
  - testing narrative-based memory models
  - studying self-model persistence and autobiographical continuity

- **Agent memory research**
  - episodic recall
  - memory reconsolidation
  - graph-based associative retrieval

- **Planning + reflection agents**
  - combining MCTS planning with memory and self-critique
  - adaptive action selection under uncertainty

- **Interactive AI demos**
  - CLI-based cognitive cycles
  - API-driven chat and introspection endpoints

- **Research-inspired LLM systems**
  - hybrid local/cloud routing
  - narrative prompting grounded in internal memory state

---

## 🔬 Scientific Grounding

| Concept | Implementation |
|---------|----------------|
| Active Inference / Free Energy Principle | `HierarchicalWorldModel` – precision-weighted prediction errors across 4 levels |
| Complementary Learning Systems | `TemporalNarrativeGraph` (fast episodic) + `ResonanceMemoryEngine` (slow semantic) |
| Temporal Memory Indexing | TNG: engrams as nodes, temporal/causal edges |
| Neuromodulatory Gain Control | `NeuromodulatoryFabric` – DA (learning rate), NE (arousal), ACh (precision), 5HT (stability) |
| Memory Reconsolidation | `Reconsolidation` window – retrieved engrams can be blended with new context |
| Predictive Coding | 4-level hierarchy with BCM-gated Hebbian learning |
| MCTS + Alpha-style Planning | `DeliberativeCortex` – PUCT selection, value rollouts |
| Engram Allocation & Competition | TNG – pattern separation via cosine threshold; edges encode relationships |
| World Models / JEPA | World model predicts next observation; free energy drives learning |
| Transformer Attention | `ThalamicRouter` – multi-head attention over feature sources |
| BCM Plasticity | `PredictiveCodingLevel` – sliding threshold maintains synaptic stability |
| Metaplasticity | `NeuromodulatoryFabric` – accumulated excess activity reduces baseline |
| Spreading Activation | TNG – BFS through graph with edge-type weights |
| Nostalgia as Replay | `NostalgiaTrace` – curated journey through TNG, re-narrated by LLM |

---

## 📋 Requirements

### Minimum

- **Python 3.10+** recommended
- `numpy`
- `scipy`

### Optional dependencies

Install these for full functionality:

- `sentence-transformers`
- `faiss-cpu`
- `torch`
- `anthropic`
- `openai`
- `llama-cpp-python`
- `fastapi`
- `uvicorn`
- `pydantic`

### Runtime notes

- If `sentence-transformers` is unavailable, the system falls back to deterministic hash-based embeddings.
- If cloud API keys are not set, LLM generation falls back to local or mock behavior.
- If `FastAPI` is not installed, API mode will not be available.
- Persistent data is stored under `~/.noesis_prime` by default.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/tranngochung0885-pixel/noesis_prime.git
cd noesis_prime

# Install core dependencies
pip install -r requirements.txt

# Optional editable install for development
pip install -e .[dev]
```

### Environment Variables

```bash
# For cloud LLM backends
export ANTHROPIC_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."

# Select backend/model (optional)
export NOESIS_BACKEND="anthropic"
export NOESIS_MODEL="claude-sonnet-4-20250514"

# Logging level (optional)
export NOESIS_LOG="INFO"

# For local GGUF (optional)
export NOESIS_LOCAL_PATH="/path/to/model.gguf"
```

### Run Demonstration

```bash
python noesis_prime.py demo
```

This demonstration runs:

1. Knowledge injection – 20 neuroscience/AI facts stored in TNG.
2. Goal setting – adds two persistent goals.
3. Cognitive cycles – multiple queries processed with predictive coding, memory retrieval, planning, and optional LLM generation.
4. Nostalgia demonstration – revisits the theme `hippocampus memory consolidation`.
5. Consolidation – offline memory strengthening and pruning.
6. Final introspection – full state snapshot.

### Interactive CLI

```bash
python noesis_prime.py cli
```

Useful commands:

- `/inject <fact>`
- `/recall <query>`
- `/nostalgize <theme>`
- `/goal <description>`
- `/introspect`
- `/telemetry`
- `/save`
- `/consolidate`
- `/reset_wm`
- `/session <id>`

Example session:

```text
noesis> What is reconsolidation?
[RESPOND | mode=focused | intent=factual | conf=0.87 | surp=0.23]
Reconsolidation is the process by which a retrieved memory becomes temporarily
modifiable before being re-stabilised. This window allows the memory to be
updated with new information.

Memory context:
  • Reconsolidation allows retrieved memories to be modified before re-stabilisation.
```

### REST API Server

```bash
python noesis_prime.py serve --host 127.0.0.1 --port 8000
```

Then call the API:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test","message":"What is predictive coding?"}'
```

---

## 🔌 API Overview

Available endpoints include:

- `GET /healthz`
- `POST /step`
- `POST /chat`
- `POST /inject`
- `POST /search`
- `POST /nostalgize`
- `POST /goal`
- `GET /introspect/{session_id}`
- `GET /sessions`
- `DELETE /sessions/{session_id}`
- `WS /ws/chat`

### Example request

```json
{
  "session_id": "test",
  "message": "What is predictive coding?",
  "reward": 0.0
}
```

### Example response

```json
{
  "response": "Predictive coding is a framework in which the brain or agent continuously generates predictions and updates them using prediction errors.",
  "session_id": "test"
}
```

For more details, see:
- `docs/api.md`
- `docs/architecture.md`

---

## 📁 Project Structure

```text
.
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── workflows/
│   └── pull_request_template.md
├── docs/
│   ├── api.md
│   └── architecture.md
├── tests/
│   ├── test_config.py
│   ├── test_import.py
│   └── test_vec.py
├── .gitignore
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── SECURITY.md
├── noesis_prime.py
├── pyproject.toml
└── requirements.txt
```

---

## 🧠 Architecture Overview

NOESIS PRIME runs one cognitive cycle per `step()` call:

1. **Perception** – Encode raw input into an embedding; detect intent, sentiment, and value conflict.
2. **World Model** – Run 4-level predictive coding to update representations and compute free energy.
3. **Neuromodulation** – Update DA/NE/ACh/5HT/GABA/GLU based on surprise, reward, and threat.
4. **Memory Retrieval** – Query episodic and semantic memory using a blended observation/working-memory cue.
5. **Reconsolidation** – Retrieved engrams may be updated within the reconsolidation window.
6. **Nostalgia Check** – Trigger narrative replay when nostalgic conditions are met.
7. **Identity Probe** – Estimate alignment between current observation and self-model.
8. **Planning** – Use MCTS or fast policy selection depending on cognitive mode.
9. **Action Execution** – Dispatch to handlers such as `SPEAK`, `RECALL`, `REFLECT`, `PLAN`, `IMAGINE`, `NOSTALGIZE`.
10. **Self-Critique** – Refine low-confidence outputs.
11. **Reward + Learning** – Apply TD update and update memory value traces.
12. **Memory Write** – Store current episode and extract concepts.
13. **Telemetry + Consolidation** – Record metrics and run periodic consolidation.

All modules read from and propose updates to the **Internal State Tensor (IST)** — the shared source of truth for the current cognitive state.

For a higher-level architectural summary, see `docs/architecture.md`.

---

## ⚙️ Configuration

All major hyperparameters are defined in `NOESISConfig`.

Example:

```python
cfg = NOESISConfig(
    embed_dim=256,
    wm_dims=(256, 128, 64, 32),
    mcts_simulations=32,
    tng_causal_threshold=0.35,
    reconsolidation_window_h=2.0,
    llm_backend="anthropic",
)
```

Key configuration groups include:

- identity and persistence
- embedding setup
- Temporal Narrative Graph limits
- world model dimensions and learning rates
- memory and reconsolidation parameters
- neuromodulatory baseline dynamics
- MCTS and planning controls
- LLM backend/model settings
- logging and performance options

---

## 📊 Telemetry & Introspection

Built-in observability features include:

- `agent.monitor.summary()` – rolling performance metrics
- `agent.introspect()` – full internal snapshot
- `agent.search_memory(query)` – episodic memory search
- `agent.nostalgize(theme)` – nostalgia trace generation
- `/introspect/{session_id}` – REST inspection endpoint
- `/sessions` – session overview

The introspection output includes:
- identity state
- cognition metrics
- world model free energies
- neuromodulator snapshot
- memory statistics
- LLM routing statistics
- current plan and cumulative reward

---

## 🛠 Development

### Install development dependencies

```bash
pip install -e .[dev]
```

### Run tests

```bash
pytest -q
```

### Run lint

```bash
ruff check .
```

### Continuous Integration

The repository includes a GitHub Actions workflow for:
- multi-version Python checks
- linting with Ruff
- running tests with Pytest

---

## 🧪 Extending NOESIS

Ways to extend the current system:

- add new `ActionKind` values and implement handlers in `_execute()`
- add new `EdgeKind` values in the Temporal Narrative Graph
- add alternative embedding pipelines
- introduce new memory policies or replay heuristics
- add more neuromodulators to `NTXState`
- support additional LLM providers
- modularize the single-file implementation into packages
- add benchmark harnesses or evaluation datasets

---

## ⚠️ Limitations

Current limitations of the repository:

- **Single-file implementation** – powerful as a reference, but harder to maintain at scale.
- **Early packaging stage** – `requirements.txt` and `pyproject.toml` are now included, but packaging can still be improved further.
- **Basic test coverage only** – test scaffolding exists, but coverage is still minimal and should be expanded.
- **Research-first behavior** – several mechanisms are conceptually rich but not benchmarked against standardized tasks.
- **Optional dependency variability** – behavior changes depending on which optional packages are installed.
- **Production hardening not included** – authentication, rate limiting, deployment guides, and persistence migrations are not yet formalized.

---

## 🛣️ Roadmap Ideas

Natural next steps for making the repository even stronger:

- split `noesis_prime.py` into modular packages
- expand automated tests for memory, planning, and API endpoints
- provide Docker support
- include architecture diagrams and benchmark results
- add reproducible experiments and evaluation scripts
- expose richer API schemas and OpenAPI examples
- add persistent session export/import utilities
- publish formal releases and tagged milestones

---

## 📚 Citation

```bibtex
@misc{noesis2026,
  author = {tranngochung0885-pixel},
  title = {NOESIS PRIME: Narrative-Oriented Emergent Intelligence System},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tranngochung0885-pixel/noesis_prime}}
}
```

---

## 👤 Author

**GitHub:** `tranngochung0885-pixel`  
**Repository:** `tranngochung0885-pixel/noesis_prime`

If you want this section to look even more professional later, you can replace the GitHub username with:
- your real name
- a short bio
- contact email
- project website or paper link

---

## 📜 License

MIT License. See the `LICENSE` file.

---

Built with ✨ by **tranngochung0885-pixel**, synthesising ideas from cognitive science, neuroscience, reinforcement learning, and multi-layer cognitive architecture design.
