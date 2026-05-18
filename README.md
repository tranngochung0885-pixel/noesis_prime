# NOESIS PRIME v1.0 – Narrative-Oriented Emergent Intelligence System

**A unified neuro‐inspired cognitive architecture where memory is a living graph, thought is a narrative, and the self is a continuous process.**

NOESIS PRIME is the culmination of three generations of cognitive architectures (AXIOM → NEXUS → HELIOS → MNEMOS → NOESIS). It implements a single, coherent mind through ten deeply integrated modules that read from and write to a shared internal state (the Internal State Tensor, IST).

The core insight: **human cognition is not data processing – it is a continuous, recursive narrative engine.** Every perception is interpreted through the lens of past experience, future expectation, and self‐model. Memory is not storage; it is reconstruction. Thought is not computation; it is a living story being rewritten in real‐time.

---

## ✨ Key Features

- **Temporal Narrative Graph (TNG)** – Episodic memories stored as nodes in a causal‐temporal graph, not a flat list. Edges include `TEMPORAL`, `CAUSAL`, `CONTRAST`, `REFLECTS`, `ASSOCIATIVE`.
- **4‐Level Hierarchical World Model** – Predictive coding across four levels: sensory (L0), conceptual (L1), narrative (L2), self‐model (L3).
- **Reconsolidation** – Retrieved memories enter a 2‐hour window during which they can be modified (Nader et al. 2000).
- **Nostalgia Engine** – Deliberate replay of past engrams into a coherent narrative journey; extracts insights via spreading activation.
- **Neuromodulatory Fabric** – DA/NE/ACh/5HT/GABA/GLU with metaplasticity tracking (sustained high activity leads to compensatory baseline shifts).
- **MCTS Planner** – Monte Carlo Tree Search (PUCT) with learned priors from the actor‐critic, inspired by AlphaZero.
- **Identity Continuity Layer** – Persistent self‐model (personality, core values, formative events) saved across sessions.
- **Meta‐Cognitive Monitor** – Confidence calibration, uncertainty estimation, dynamic MCTS budget, performance alerts.
- **Adaptive LLM Bridge** – Intelligent routing: local GGUF for fast ops, cloud Anthropic/OpenAI for deep reasoning, with fallback chain.
- **Unified Internal State (IST)** – All modules read from and propose updates to a single tensor; the Agency Core integrates them each cycle.

---

## 🔬 Scientific Grounding

| Concept | Implementation |
|---------|----------------|
| Active Inference / Free Energy Principle | `HierarchicalWorldModel` – precision‐weighted prediction errors across 4 levels |
| Complementary Learning Systems | `TemporalNarrativeGraph` (fast episodic) + `ResonanceMemoryEngine` (slow semantic) |
| Temporal Memory Indexing | TNG: engrams as nodes, temporal/causal edges |
| Neuromodulatory Gain Control | `NeuromodulatoryFabric` – DA (learning rate), NE (arousal), ACh (precision), 5HT (stability) |
| Memory Reconsolidation | `Reconsolidation` window – retrieved engrams can be blended with new context |
| Predictive Coding | 4‐level hierarchy with BCM‐gated Hebbian learning |
| MCTS + Alpha‐style Planning | `DeliberativeCortex` – PUCT selection, value rollouts |
| Engram Allocation & Competition | TNG – pattern separation via cosine threshold; edges encode relationships |
| World Models / JEPA | World model predicts next observation; free energy drives learning |
| Transformer Attention | `ThalamicRouter` – multi‐head attention over feature sources |
| BCM Plasticity | `PredictiveCodingLevel` – sliding threshold maintains synaptic stability |
| Metaplasticity | `NeuromodulatoryFabric` – accumulated excess activity reduces baseline |
| Spreading Activation | TNG – BFS through graph with edge‐type weights |
| Nostalgia as Replay | `NostalgiaTrace` – curated journey through TNG, re‐narrated by LLM |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourname/noesis-prime.git
cd noesis-prime

# Install core dependencies
pip install numpy scipy

# For full capabilities:
pip install sentence-transformers faiss-cpu anthropic openai llama-cpp-python fastapi uvicorn
```

Environment Variables

```bash
# For cloud LLMs
export ANTHROPIC_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."

# For local GGUF (optional)
export NOESIS_LOCAL_PATH="/path/to/model.gguf"
```

Run Demonstration

```bash
python noesis_prime.py demo
```

This runs:

1. Knowledge injection – 20 neuroscience/AI facts stored in TNG.
2. Goal setting – adds two persistent goals.
3. Cognitive cycles – 8 queries processed, each with full predictive coding, MCTS, and optional LLM generation.
4. Nostalgia demonstration – revisits the theme "hippocampus memory consolidation".
5. Consolidation – offline memory strengthening and pruning.
6. Final introspection – full state snapshot.

Interactive CLI

```bash
python noesis_prime.py cli
```

Type /help for commands. Example session:

```
noesis> What is reconsolidation?
[RESPOND | mode=focused | intent=factual | conf=0.87 | surp=0.23]
Reconsolidation is the process by which a retrieved memory becomes temporarily 
modifiable before being re-stabilised. This window (about 2 hours) allows the 
memory to be updated with new information, which is crucial for learning and 
therapeutic interventions (Nader et al. 2000).

  Memory context:
    • Reconsolidation allows retrieved memories to be modified before re-stabilisation.

Reward? [Enter=0, +/good/-/bad/value]: +
```

REST API Server

```bash
python noesis_prime.py serve --port 8000
```

Then use the API:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test","message":"What is predictive coding?"}'
```

---

🧠 Architecture Overview

NOESIS PRIME runs one cognitive cycle per step() call:

1. Perception – Encode raw input → embed_dim vector, detect intent, sentiment, value conflict.
2. World Model – 4‐level predictive coding → free energy, surprise, self‐model update.
3. Neuromodulation – Update DA/NE/ACh/5HT/GABA/GLU based on surprise, reward, threat.
4. Memory Retrieval – Query TNG + semantic index with weighted cue (obs + WM context).
5. Reconsolidation – For each retrieved engram, if within 2‐hour window, blend with current context.
6. Nostalgia Check – If mode is NOSTALGIC or intent is NOSTALGIC, generate nostalgia trace.
7. Identity Probe – Compute alignment between current observation and self‐model.
8. Planning (MCTS) – Build state vector (obs + narrative + WM), run PUCT search, select action.
9. Action Execution – Dispatch to handler (SPEAK, RECALL, REFLECT, PLAN, IMAGINE, NOSTALGIZE, etc.).
10. Self‐Critique – If confidence < threshold, run second LLM call to refine response.
11. Reward + Learning – TD update, update TNG Q‐values, neuromod reward signals.
12. Memory Write – Store experience as new engram; extract concepts; add reflection/associative edges.
13. Telemetry + Consolidation – Record metrics; periodic consolidation (pruning, cascade updates).

All modules read from and propose updates to the Internal State Tensor (IST) – the single source of truth.

---

⚙️ Configuration

All hyperparameters are in NOESISConfig. Example:

```python
cfg = NOESISConfig(
    embed_dim=256,
    wm_dims=(256, 128, 64, 32),        # 4‐level world model
    mcts_simulations=32,
    tng_causal_threshold=0.35,
    reconsolidation_window_h=2.0,
    llm_backend="anthropic",
)
```

See the dataclass for full documentation.

---

📁 Code Structure

· noesis_prime.py – the entire system (single file, 5000+ lines).
· NOESISAgent – main cognitive core.
· TemporalNarrativeGraph – episodic store with causal/temporal edges.
· HierarchicalWorldModel – 4‐level predictive coding.
· ResonanceMemoryEngine – semantic + working memory + nostalgia.
· NeuromodulatoryFabric – DA/NE/ACh/5HT/GABA/GLU dynamics.
· DeliberativeCortex – MCTS planner + actor‐critic.
· IdentityContinuityLayer – persistent self‐model.
· MetaCognitiveMonitor – uncertainty, calibration, alerts.
· AdaptiveLLMBridge – local + cloud LLM routing.
· SessionManager – multi‐session orchestration.
· run_demo(), run_cli(), build_api() – entry points.

---

🧪 Extending NOESIS

· New action types – add to ActionKind enum and implement handler in _execute().
· New edge types – add to EdgeKind, modify edge creation logic.
· Custom environment – subclass BaseEnvironment (not included – use directly with agent.step()).
· Additional neuromodulators – add fields to NTXState and update dynamics in NeuromodulatoryFabric.
· Different LLM backends – subclass LLMProvider and add to AdaptiveLLMBridge.

---

📊 Telemetry & Introspection

· agent.monitor.summary() – rolling performance metrics (avg reward, TD error, confidence, calibration error).
· agent.introspect() – full snapshot of identity, cognition, world model, memory, neuromodulators.
· agent.monitor.plot_ascii("confidence") – ASCII time‐series for quick inspection.

---

📜 License

MIT License. See LICENSE file.

---

📚 Citation

```bibtex
@misc{noesis2026,
  author = {Your Name},
  title = {NOESIS PRIME: Narrative-Oriented Emergent Intelligence System},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourname/noesis-prime}}
}
```

---

Built with ✨ by synthesising ideas from cognitive science, neuroscience, reinforcement learning, and three generations of cognitive architectures.

