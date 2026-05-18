# Architecture Overview

This document provides a high-level architectural view of **NOESIS PRIME**.

## Core Idea

NOESIS PRIME models cognition as a continuous loop integrating:

- perception
- predictive world modeling
- neuromodulation
- memory retrieval and reconsolidation
- planning
- response generation
- self-critique
- long-term memory writing

## Main Components

### 1. CognitiveEncoder
Encodes raw input into a unified embedding space.

### 2. HierarchicalWorldModel
A 4-level predictive coding hierarchy:
- L0: perception
- L1: concept
- L2: narrative
- L3: self-model

### 3. TemporalNarrativeGraph
Stores episodic memories as nodes in a causal-temporal graph.

### 4. ResonanceMemoryEngine
Handles episodic recall, semantic concepts, working memory, nostalgia traces, and reconsolidation.

### 5. NeuromodulatoryFabric
Tracks DA/NE/ACh/5HT/GABA/GLU dynamics and affects learning, attention, and mode selection.

### 6. DeliberativeCortex
Combines actor-critic learning and MCTS-style planning for action selection.

### 7. IdentityContinuityLayer
Maintains persistent identity across sessions.

### 8. AdaptiveLLMBridge
Routes between local GGUF models, cloud LLMs, or mock mode.

### 9. MetaCognitiveMonitor
Tracks confidence, surprise, calibration, and performance alerts.

### 10. NOESISAgent
The unified orchestration layer tying all modules together into a single cognitive cycle.

## Cognitive Cycle

A typical `step()` performs:

1. Encode observation
2. Update world model
3. Update neuromodulators
4. Retrieve memory
5. Reconsolidate if needed
6. Generate nostalgia if triggered
7. Probe identity alignment
8. Plan and choose action
9. Execute action
10. Self-critique response
11. Learn from reward
12. Write memory
13. Record telemetry

## Persistence

By default, persistent runtime state is stored under:

```text
~/.noesis_prime/
```

This includes:
- SQLite databases
- checkpoints
- logs
- identity state

## Implementation Note

The current repository uses a **single-file reference implementation** in `noesis_prime.py`. This is useful for readability and conceptual coherence, but future modularization would improve maintainability.