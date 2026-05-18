#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║   N O E S I S   P R I M E   v1.0                                                    ║
║   Narrative-Oriented Emergent Intelligence System                                    ║
║                                                                                      ║
║   Philosophy:                                                                        ║
║   A human mind does not process data through sequential modules — it is a           ║
║   continuous, recursive narrative engine. Every perception is interpreted           ║
║   through the lens of past experience, future expectation, and self-model.          ║
║   Memory is not storage: it is reconstruction. Thought is not computation:          ║
║   it is a living story being rewritten in real-time.                                ║
║                                                                                      ║
║   NOESIS PRIME instantiates this as:                                                 ║
║     1. Cognitive Substrate      — unified latent engram space, all modalities       ║
║     2. Temporal Narrative Graph — episodic events as causal graph, not flat list    ║
║     3. Hierarchical World Model — 4-level predictive processing hierarchy           ║
║     4. Resonance Memory Engine  — consolidation, nostalgia, creative recombination  ║
║     5. Neuromodulatory Fabric   — DA/NE/ACh/5HT + metaplasticity dynamics          ║
║     6. Deliberative Cortex      — MCTS-guided planning, CoT, self-critique loops   ║
║     7. Identity Continuity Layer — autobiographical self-model, value alignment     ║
║     8. Adaptive LLM Bridge      — local GGUF + cloud with intelligent routing      ║
║     9. Meta-Cognitive Monitor   — uncertainty, calibration, resource allocation     ║
║    10. Unified Agency Core      — integrates all streams, drives coherent action    ║
║                                                                                      ║
║   Scientific Foundations:                                                            ║
║   • Active Inference / Free Energy Principle (Friston 2010–2024)                   ║
║   • Complementary Learning Systems (McClelland et al.; Kumaran et al. 2016)        ║
║   • Temporal Memory Indexing (Teyler & DiScenna 1986; Rubin et al. 2024)          ║
║   • Neuromodulatory Gain Control (Dayan & Yu 2006; Friston et al. 2012)           ║
║   • Memory Reconsolidation (Nader et al. 2000; Haubrich & Bhatt 2020)             ║
║   • Predictive Coding (Rao & Ballard 1999; Clark 2013; Hohwy 2020)                ║
║   • MCTS + Alpha-family planning (Silver et al. 2016–2023)                         ║
║   • Engram Allocation & Competition (Tonegawa et al. 2015; Bhatt et al. 2020)     ║
║   • World Models / JEPA (LeCun 2022; Assran et al. 2023)                           ║
║   • Transformer attention as dynamic binding (Ramsauer et al. 2020)               ║
║                                                                                      ║
║   Differences from CORTICEX / HELIOS:                                               ║
║   • Temporal graph of episodes (not flat list) — causal & temporal edges           ║
║   • Reconsolidation: retrieved memories are modified, not just activated           ║
║   • Nostalgia engine: actively replays past to extract new meaning                 ║
║   • 4-level world model with explicit narrative layer (L3)                         ║
║   • MCTS planning with learned priors from episodic memory                         ║
║   • Identity continuity: self-model persists across sessions                       ║
║   • Intelligent LLM routing: local for fast ops, cloud for deep reasoning          ║
║   • Metaplasticity: learning rates shift based on cumulative history               ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# §0  STDLIB IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import hashlib
import json
import logging
import math
import os
import random
import re
import sqlite3
import sys
import threading
import time
import uuid
import warnings
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

warnings.filterwarnings("ignore", category=UserWarning)

T = TypeVar("T")
EPS = 1e-12
INF = float("inf")
TAU = 2.0 * math.pi

# ═══════════════════════════════════════════════════════════════════════════════
# §1  OPTIONAL DEPENDENCY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _try_import(module_name: str, package: str = "") -> Optional[Any]:
    """Safe import with informative logging."""
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError:
        return None

_np_mod   = _try_import("numpy")
_sp_mod   = _try_import("scipy")
_st_mod   = _try_import("sentence_transformers")
_faiss_mod = _try_import("faiss")
_torch_mod = _try_import("torch")
_anthropic_mod = _try_import("anthropic")
_openai_mod    = _try_import("openai")
_fastapi_mod   = _try_import("fastapi")
_uvicorn_mod   = _try_import("uvicorn")
_llama_mod     = _try_import("llama_cpp")

HAS_NUMPY       = _np_mod is not None
HAS_SCIPY       = _sp_mod is not None
HAS_ST          = _st_mod is not None
HAS_FAISS       = _faiss_mod is not None
HAS_TORCH       = _torch_mod is not None
HAS_ANTHROPIC   = _anthropic_mod is not None
HAS_OPENAI      = _openai_mod is not None
HAS_FASTAPI     = _fastapi_mod is not None
HAS_LLAMA       = _llama_mod is not None

if HAS_NUMPY:
    import numpy as np
else:
    np = None  # type: ignore

if HAS_FAISS:
    import faiss  # type: ignore

if HAS_ANTHROPIC:
    import anthropic as _anthropic  # type: ignore

if HAS_OPENAI:
    import openai as _openai  # type: ignore

if HAS_LLAMA:
    from llama_cpp import Llama  # type: ignore

# ═══════════════════════════════════════════════════════════════════════════════
# §2  LOGGING SYSTEM — structured, context-aware
# ═══════════════════════════════════════════════════════════════════════════════

_LOG_FORMAT = (
    "%(asctime)s │ %(levelname)-8s │ [%(component)-16s] │ %(message)s"
)
_DATE_FORMAT = "%H:%M:%S"

class ContextualLogger:
    """
    Logger that automatically injects component name, session id,
    and step count into every message.
    """
    _root_logger: ClassVar[logging.Logger] = logging.getLogger("NOESIS")
    _handler_installed: ClassVar[bool] = False

    def __init__(self, component: str):
        self.component = component
        self._logger = logging.getLogger(f"NOESIS.{component}")
        self._context: Dict[str, Any] = {}
        self._install_handler()

    @classmethod
    def _install_handler(cls):
        if cls._handler_installed:
            return
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        cls._root_logger.addHandler(h)
        cls._root_logger.setLevel(logging.INFO)
        cls._handler_installed = True

    def set_context(self, **kwargs):
        self._context.update(kwargs)

    def _fmt(self, msg: str) -> str:
        ctx_parts = [f"{k}={v}" for k, v in self._context.items()]
        if ctx_parts:
            return f"{msg} [{', '.join(ctx_parts)}]"
        return msg

    def debug(self, msg: str, *a, **kw):
        self._logger.debug(self._fmt(msg), *a, extra={"component": self.component}, **kw)

    def info(self, msg: str, *a, **kw):
        self._logger.info(self._fmt(msg), *a, extra={"component": self.component}, **kw)

    def warning(self, msg: str, *a, **kw):
        self._logger.warning(self._fmt(msg), *a, extra={"component": self.component}, **kw)

    def error(self, msg: str, *a, **kw):
        self._logger.error(self._fmt(msg), *a, extra={"component": self.component}, **kw)

    def exception(self, msg: str, *a, **kw):
        self._logger.exception(self._fmt(msg), *a, extra={"component": self.component}, **kw)


def set_log_level(level: str):
    logging.getLogger("NOESIS").setLevel(getattr(logging, level.upper(), logging.INFO))

# ═══════════════════════════════════════════════════════════════════════════════
# §3  ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class IntentClass(str, Enum):
    FACTUAL       = "factual"
    ANALYTICAL    = "analytical"
    CREATIVE      = "creative"
    PROCEDURAL    = "procedural"
    SOCIAL        = "social"
    EMOTIONAL     = "emotional"
    METACOGNITIVE = "metacognitive"
    NOSTALGIC     = "nostalgic"     # user wants to revisit past
    ADVERSARIAL   = "adversarial"   # testing / probing
    AMBIGUOUS     = "ambiguous"

class ActionKind(str, Enum):
    SPEAK         = "speak"
    RECALL        = "recall"
    REFLECT       = "reflect"
    PLAN          = "plan"
    IMAGINE       = "imagine"        # mental simulation / counterfactual
    STORE         = "store"
    CONSOLIDATE   = "consolidate"
    SEARCH        = "search"
    TOOL_CALL     = "tool_call"
    NOSTALGIZE    = "nostalgize"     # deliberate nostalgia replay
    NOOP          = "noop"

class MemoryKind(str, Enum):
    EPISODIC      = "episodic"
    SEMANTIC      = "semantic"
    WORKING       = "working"
    PROCEDURAL    = "procedural"
    AUTOBIO       = "autobiographical"

class ConsolidationPhase(str, Enum):
    LABILE        = "labile"         # < 2h, easily overwritten
    EARLY         = "early"          # 2–12h, CREB cascade
    STABLE        = "stable"         # 12–72h, protein synthesis complete
    REMOTE        = "remote"         # > 72h, neocortically resident
    RECONSOLIDATING = "reconsolidating"  # active rewrite in progress

class NTX(str, Enum):
    DA   = "dopamine"
    NE   = "norepinephrine"
    ACH  = "acetylcholine"
    HT   = "serotonin"
    GABA = "gaba"           # inhibitory tone
    GLU  = "glutamate"      # excitatory gain

class CognitiveMode(str, Enum):
    REFLEX     = "reflex"       # fast, sub-threshold event
    FOCUSED    = "focused"      # deliberate single-thread
    EXPANSIVE  = "expansive"    # broad search, creative
    CRITICAL   = "critical"     # self-monitoring, high scrutiny
    NOSTALGIC  = "nostalgic"    # replay + reanalysis
    DREAMING   = "dreaming"     # offline consolidation

class LLMBackend(str, Enum):
    LOCAL_GGUF  = "local_gguf"
    ANTHROPIC   = "anthropic"
    OPENAI      = "openai"
    MOCK        = "mock"

class EdgeKind(str, Enum):
    """Types of edges in the Temporal Narrative Graph."""
    TEMPORAL    = "temporal"       # event A happened before B
    CAUSAL      = "causal"         # A caused B
    ASSOCIATIVE = "associative"    # A co-occurred with B (Hebbian)
    CONTRAST    = "contrast"       # A contradicts B
    SUBSUMES    = "subsumes"       # A generalises B (hierarchical)
    REFLECTS    = "reflects"       # A is a reflection/insight about B

# ═══════════════════════════════════════════════════════════════════════════════
# §4  MASTER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NOESISConfig:
    """
    All hyperparameters, paths, and feature flags in one place.
    Every field has a docstring-style comment and an env-var override.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    agent_name: str          = "Noesis"
    agent_version: str       = "1.0.0"
    session_persistent: bool = True   # persist state across runs

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_root: Path = Path("~/.noesis_prime").expanduser()

    # ── Embedding ─────────────────────────────────────────────────────────────
    embed_dim: int           = 768       # unified latent space dimension
    embed_model: str         = "all-mpnet-base-v2"   # sentence-transformers
    embed_cache_size: int    = 20_000

    # ── Temporal Narrative Graph ──────────────────────────────────────────────
    tng_max_nodes: int       = 100_000
    tng_max_edges: int       = 500_000
    tng_edge_decay: float    = 0.002      # per-hour decay on associative edges
    tng_causal_threshold: float = 0.35    # cosine sim for causal inference
    tng_temporal_window_s: float = 120.0  # events within this window get temporal links

    # ── World Model (4 levels) ────────────────────────────────────────────────
    wm_dims: Tuple = (768, 384, 192, 96)   # L0=perception, L1=concept, L2=narrative, L3=self
    wm_lr: Tuple   = (0.08, 0.04, 0.02, 0.01)
    wm_precision: Tuple = (1.0, 1.5, 2.5, 4.0)   # higher = sharper prior
    wm_habituation_window: int = 32

    # ── Resonance Memory Engine ────────────────────────────────────────────────
    episodic_capacity: int   = 80_000
    semantic_capacity: int   = 30_000
    replay_batch_size: int   = 48
    nostalgia_depth: int     = 8         # episodes per nostalgia trace
    reconsolidation_window_h: float = 2.0  # time window where memory is modifiable
    consolidation_interval: int = 250    # steps between full consolidation runs
    engram_halflife_h: float = 96.0      # Ebbinghaus half-life for raw episodic

    # ── Neuromodulatory Fabric ────────────────────────────────────────────────
    ntx_reuptake: float      = 0.12      # per-step exponential decay toward baseline
    ntx_baseline: Dict = field(default_factory=lambda: {
        "DA": 0.40, "NE": 0.28, "ACH": 0.38, "HT": 0.48, "GABA": 0.50, "GLU": 0.45
    })
    metaplasticity_window: int = 500   # steps over which metaplasticity is tracked

    # ── Deliberative Cortex (MCTS) ────────────────────────────────────────────
    mcts_simulations: int    = 64
    mcts_c_puct: float       = 1.4
    mcts_max_depth: int      = 6
    mcts_temperature: float  = 0.8
    plan_horizon: int        = 4

    # ── Working Memory ─────────────────────────────────────────────────────────
    wm_slots: int            = 7       # Miller's Law
    wm_decay_s: float        = 480.0   # TTL for working memory slots

    # ── Identity / Autobiographical ────────────────────────────────────────────
    autobio_max_events: int  = 5_000
    self_model_dim: int      = 96      # same as WM L3
    identity_update_lr: float = 0.005

    # ── Meta-Cognitive Monitor ─────────────────────────────────────────────────
    confidence_ema_alpha: float = 0.15
    calibration_window: int  = 100
    uncertainty_threshold: float = 0.35

    # ── LLM Configuration ─────────────────────────────────────────────────────
    llm_backend: str         = os.getenv("NOESIS_BACKEND", "anthropic")
    llm_cloud_model: str     = os.getenv("NOESIS_MODEL", "claude-sonnet-4-20250514")
    llm_local_model_path: str = os.getenv("NOESIS_LOCAL_PATH", "")
    llm_max_tokens: int      = 1536
    llm_temperature: float   = 0.72
    llm_timeout_s: float     = 30.0
    llm_local_ctx_size: int  = 4096   # context window for GGUF models

    # ── Performance ────────────────────────────────────────────────────────────
    async_enabled: bool      = True
    faiss_nlist: int         = 128
    faiss_nprobe: int        = 16

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str           = os.getenv("NOESIS_LOG", "INFO")

    def __post_init__(self):
        self.data_root = Path(self.data_root).expanduser()
        self.data_root.mkdir(parents=True, exist_ok=True)
        (self.data_root / "db").mkdir(exist_ok=True)
        (self.data_root / "checkpoints").mkdir(exist_ok=True)
        (self.data_root / "logs").mkdir(exist_ok=True)
        assert len(self.wm_dims) == 4, "World model needs exactly 4 levels"
        assert len(self.wm_lr) == 4
        assert len(self.wm_precision) == 4
        set_log_level(self.log_level)

_GLOBAL_CFG: Optional[NOESISConfig] = None

def get_cfg() -> NOESISConfig:
    global _GLOBAL_CFG
    if _GLOBAL_CFG is None:
        _GLOBAL_CFG = NOESISConfig()
    return _GLOBAL_CFG

def set_cfg(cfg: NOESISConfig):
    global _GLOBAL_CFG
    _GLOBAL_CFG = cfg

# ═══════════════════════════════════════════════════════════════════════════════
# §5  VECTOR MATHEMATICS — the pure numeric core
# ═══════════════════════════════════════════════════════════════════════════════

class Vec:
    """
    All vector / matrix math lives here.
    Pure-Python fallbacks ensure correctness without numpy.
    NumPy accelerates everything when available.

    Design principle: every method accepts list[float] or np.ndarray
    and returns ndarray when numpy is present, else list[float].
    """

    # ── construction ───────────────────────────────────────────────────────────

    @staticmethod
    def zeros(n: int) -> Any:
        if HAS_NUMPY: return np.zeros(n, dtype=np.float32)
        return [0.0] * n

    @staticmethod
    def ones(n: int) -> Any:
        if HAS_NUMPY: return np.ones(n, dtype=np.float32)
        return [1.0] * n

    @staticmethod
    def rand(n: int, seed: Optional[int] = None) -> Any:
        if HAS_NUMPY:
            rng = np.random.default_rng(seed)
            return rng.standard_normal(n).astype(np.float32)
        rng = random.Random(seed)
        return [rng.gauss(0.0, 1.0) for _ in range(n)]

    @staticmethod
    def rand_unit(n: int, seed: Optional[int] = None) -> Any:
        v = Vec.rand(n, seed)
        return Vec.normalize(v)

    @staticmethod
    def hash_unit(text: str, n: int) -> Any:
        """Deterministic unit vector from text, no model needed."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big")
        return Vec.rand_unit(n, seed)

    # ── shape ops ──────────────────────────────────────────────────────────────

    @staticmethod
    def pad_trim(v: Any, n: int) -> Any:
        if HAS_NUMPY:
            a = np.asarray(v, dtype=np.float32).flatten()
            if len(a) >= n: return a[:n]
            return np.pad(a, (0, n - len(a)))
        lst = list(v)
        if len(lst) >= n: return lst[:n]
        return lst + [0.0] * (n - len(lst))

    @staticmethod
    def flatten(v: Any) -> Any:
        if HAS_NUMPY: return np.asarray(v, dtype=np.float32).flatten()
        return [float(x) for x in v]

    # ── arithmetic ─────────────────────────────────────────────────────────────

    @staticmethod
    def add(a: Any, b: Any) -> Any:
        if HAS_NUMPY:
            return np.asarray(a, np.float32) + np.asarray(b, np.float32)
        return [float(ai) + float(bi) for ai, bi in zip(a, b)]

    @staticmethod
    def scale(v: Any, s: float) -> Any:
        if HAS_NUMPY: return np.asarray(v, np.float32) * s
        return [float(x) * s for x in v]

    @staticmethod
    def lerp(a: Any, b: Any, t: float) -> Any:
        """Linear interpolation: a + t*(b-a)."""
        if HAS_NUMPY:
            av = np.asarray(a, np.float32)
            bv = np.asarray(b, np.float32)
            return av + t * (bv - av)
        return [float(ai) + t * (float(bi) - float(ai)) for ai, bi in zip(a, b)]

    @staticmethod
    def dot(a: Any, b: Any) -> float:
        if HAS_NUMPY:
            av = np.asarray(a, np.float32).flatten()
            bv = np.asarray(b, np.float32).flatten()
            n = min(len(av), len(bv))
            return float(np.dot(av[:n], bv[:n]))
        la, lb = list(a), list(b)
        return sum(float(ai) * float(bi) for ai, bi in zip(la, lb))

    @staticmethod
    def norm(v: Any) -> float:
        if HAS_NUMPY:
            return float(np.linalg.norm(np.asarray(v, np.float32)))
        return math.sqrt(sum(float(x) ** 2 for x in v))

    @staticmethod
    def normalize(v: Any) -> Any:
        if HAS_NUMPY:
            a = np.asarray(v, np.float32)
            n = float(np.linalg.norm(a))
            return a / n if n > EPS else a
        lst = list(v)
        n = math.sqrt(sum(float(x) ** 2 for x in lst))
        return [float(x) / n for x in lst] if n > EPS else lst

    @staticmethod
    def cosine(a: Any, b: Any) -> float:
        if HAS_NUMPY:
            av = np.asarray(a, np.float32).flatten()
            bv = np.asarray(b, np.float32).flatten()
            n = min(len(av), len(bv))
            av, bv = av[:n], bv[:n]
            na = float(np.linalg.norm(av))
            nb = float(np.linalg.norm(bv))
            if na < EPS or nb < EPS: return 0.0
            return float(np.clip(np.dot(av, bv) / (na * nb), -1.0, 1.0))
        la, lb = list(a), list(b)
        n = min(len(la), len(lb))
        dot = sum(float(la[i]) * float(lb[i]) for i in range(n))
        na = math.sqrt(sum(float(x) ** 2 for x in la[:n]))
        nb = math.sqrt(sum(float(x) ** 2 for x in lb[:n]))
        if na < EPS or nb < EPS: return 0.0
        return max(-1.0, min(1.0, dot / (na * nb)))

    @staticmethod
    def l2_dist(a: Any, b: Any) -> float:
        if HAS_NUMPY:
            av = np.asarray(a, np.float32).flatten()
            bv = np.asarray(b, np.float32).flatten()
            n = min(len(av), len(bv))
            return float(np.linalg.norm(av[:n] - bv[:n]))
        la, lb = list(a), list(b)
        n = min(len(la), len(lb))
        return math.sqrt(sum((float(la[i]) - float(lb[i])) ** 2 for i in range(n)))

    # ── weighted combination ───────────────────────────────────────────────────

    @staticmethod
    def weighted_mean(vecs: List[Any], weights: List[float]) -> Any:
        if not vecs: return Vec.zeros(get_cfg().embed_dim)
        total = sum(weights) + EPS
        if HAS_NUMPY:
            dim = max(len(np.asarray(v).flatten()) for v in vecs)
            result = np.zeros(dim, dtype=np.float32)
            for v, w in zip(vecs, weights):
                arr = np.asarray(v, np.float32).flatten()
                result[:len(arr)] += (w / total) * arr
            return Vec.normalize(result)
        dim = max(len(list(v)) for v in vecs)
        result = [0.0] * dim
        for v, w in zip(vecs, weights):
            lst = list(v)
            for i in range(min(dim, len(lst))):
                result[i] += (w / total) * float(lst[i])
        return Vec.normalize(result)

    # ── probability ops ────────────────────────────────────────────────────────

    @staticmethod
    def softmax(logits: Any, temperature: float = 1.0) -> Any:
        if HAS_NUMPY:
            a = np.asarray(logits, np.float64) / max(temperature, EPS)
            a -= a.max()
            e = np.exp(a)
            return (e / (e.sum() + EPS)).astype(np.float32)
        mx = max(logits)
        exps = [math.exp((x - mx) / max(temperature, EPS)) for x in logits]
        s = sum(exps) + EPS
        return [e / s for e in exps]

    @staticmethod
    def entropy(probs: Any) -> float:
        if HAS_NUMPY:
            p = np.asarray(probs, np.float64) + EPS
            return float(-np.sum(p * np.log(p)))
        return -sum((p + EPS) * math.log(p + EPS) for p in probs)

    @staticmethod
    def kl(p: Any, q: Any) -> float:
        if not HAS_NUMPY: return 0.5
        pv = np.asarray(p, np.float64).flatten() + EPS
        qv = np.asarray(q, np.float64).flatten() + EPS
        n = min(len(pv), len(qv))
        pn = pv[:n] / pv[:n].sum()
        qn = qv[:n] / qv[:n].sum()
        return float(np.sum(pn * np.log(pn / qn)))

    # ── matrix ops ────────────────────────────────────────────────────────────

    @staticmethod
    def rand_mat(rows: int, cols: int, scale: float = 0.02) -> Any:
        if HAS_NUMPY:
            return (np.random.randn(rows, cols) * scale).astype(np.float32)
        return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def zeros_mat(rows: int, cols: int) -> Any:
        if HAS_NUMPY: return np.zeros((rows, cols), dtype=np.float32)
        return [[0.0] * cols for _ in range(rows)]

    @staticmethod
    def matvec(M: Any, v: Any) -> Any:
        if HAS_NUMPY:
            return np.asarray(M, np.float32) @ np.asarray(v, np.float32)
        rows = list(M)
        cols = len(rows[0])
        return [sum(float(rows[i][j]) * float(v[j]) for j in range(min(cols, len(v))))
                for i in range(len(rows))]

    @staticmethod
    def outer(a: Any, b: Any) -> Any:
        if HAS_NUMPY:
            return np.outer(np.asarray(a, np.float32), np.asarray(b, np.float32))
        la, lb = list(a), list(b)
        return [[float(ai) * float(bi) for bi in lb] for ai in la]

    # ── predictive coding helpers ──────────────────────────────────────────────

    @staticmethod
    def prediction_error(obs: Any, pred: Any, precision: float = 1.0) -> Tuple[Any, float]:
        """Returns (error_vector, scalar_magnitude²)."""
        if HAS_NUMPY:
            o = np.asarray(obs,  np.float32).flatten()
            p = np.asarray(pred, np.float32).flatten()
            n = min(len(o), len(p))
            err = o[:n] - p[:n]
            mag = float(0.5 * precision * np.dot(err, err))
            return err, mag
        o, p = list(obs), list(pred)
        n = min(len(o), len(p))
        err = [float(o[i]) - float(p[i]) for i in range(n)]
        mag = 0.5 * precision * sum(e * e for e in err)
        return err, mag

    @staticmethod
    def free_energy(obs: Any, pred: Any, precision: float = 1.0) -> float:
        """Variational free energy ≈ precision-weighted squared error."""
        _, mag = Vec.prediction_error(obs, pred, precision)
        return mag

    @staticmethod
    def td_error(r: float, v_next: float, v_curr: float, gamma: float = 0.97) -> float:
        return r + gamma * v_next - v_curr

    # ── information theory ─────────────────────────────────────────────────────

    @staticmethod
    def surprise(obs: Any, pred: Any, precision: float = 1.0) -> float:
        """Novelty signal in [0, 1]."""
        fe = Vec.free_energy(obs, pred, precision)
        return 1.0 - math.exp(-fe)

    @staticmethod
    def mutual_info_approx(a: Any, b: Any) -> float:
        """Cosine-based approximate mutual information proxy."""
        sim = Vec.cosine(a, b)
        return max(0.0, sim)

# ═══════════════════════════════════════════════════════════════════════════════
# §6  VECTOR INDEX — ANN search (FAISS / brute force)
# ═══════════════════════════════════════════════════════════════════════════════

class VectorIndex:
    """
    Approximate nearest-neighbour search over dense embeddings.
    Uses FAISS IVFFlat when available; gracefully falls back to
    NumPy batched cosine or pure-Python exhaustive search.

    Thread-safe for concurrent reads and writes.
    """

    def __init__(
        self,
        dim: int,
        nlist: int = 128,
        nprobe: int = 16,
        metric: str = "cosine",   # "cosine" or "l2"
    ):
        self.dim    = dim
        self.nlist  = nlist
        self.nprobe = nprobe
        self.metric = metric
        self._lock  = threading.RLock()
        self._store: Dict[str, Any] = {}   # uid → normed vector
        self._faiss = None
        self._faiss_uids: List[str] = []
        self._dirty = True
        self._rebuild_at = 512
        self._log = ContextualLogger("VectorIndex")

    def add(self, uid: str, vec: Any) -> None:
        v = Vec.normalize(Vec.pad_trim(vec, self.dim))
        with self._lock:
            self._store[uid] = v
            self._dirty = True

    def remove(self, uid: str) -> None:
        with self._lock:
            self._store.pop(uid, None)
            self._dirty = True

    def search(self, query: Any, k: int = 10) -> List[Tuple[str, float]]:
        q = Vec.normalize(Vec.pad_trim(query, self.dim))
        with self._lock:
            if self._dirty and len(self._store) >= self._rebuild_at:
                self._rebuild_faiss()
            if self._faiss is not None and not self._dirty:
                return self._faiss_search(q, k)
            return self._brute_search(q, k)

    def _brute_search(self, q: Any, k: int) -> List[Tuple[str, float]]:
        if not self._store: return []
        if HAS_NUMPY:
            uids = list(self._store.keys())
            mat  = np.stack([np.asarray(self._store[u], np.float32) for u in uids])
            qv   = np.asarray(q, np.float32)
            if self.metric == "cosine":
                sims = mat @ qv
            else:
                sims = -np.linalg.norm(mat - qv[None, :], axis=1)
            idx = np.argsort(-sims)[:k]
            return [(uids[i], float(sims[i])) for i in idx]
        scores = [
            (uid, Vec.cosine(q, v) if self.metric == "cosine" else -Vec.l2_dist(q, v))
            for uid, v in self._store.items()
        ]
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    def _rebuild_faiss(self) -> None:
        if not HAS_FAISS or not HAS_NUMPY: self._dirty = False; return
        try:
            uids = list(self._store.keys())
            n    = len(uids)
            vecs = np.stack([np.asarray(self._store[u], np.float32) for u in uids])
            nlist = max(1, min(self.nlist, n // 8))
            quantiser = faiss.IndexFlatIP(self.dim)
            idx = faiss.IndexIVFFlat(quantiser, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            if n >= nlist * 4:
                idx.train(vecs)
                idx.add(vecs)
                idx.nprobe = self.nprobe
                self._faiss = idx
                self._faiss_uids = uids
            self._dirty = False
        except Exception as exc:
            self._log.debug("FAISS rebuild failed: %s", exc)
            self._dirty = False

    def _faiss_search(self, q: Any, k: int) -> List[Tuple[str, float]]:
        try:
            qv = np.asarray(q, np.float32).reshape(1, -1)
            n_ret = min(k, len(self._faiss_uids))
            D, I  = self._faiss.search(qv, n_ret)
            return [
                (self._faiss_uids[i], float(D[0][j]))
                for j, i in enumerate(I[0])
                if 0 <= i < len(self._faiss_uids)
            ]
        except Exception:
            return self._brute_search(q, k)

    def __len__(self) -> int:
        return len(self._store)

    def batch_similarities(self, query: Any, uids: List[str]) -> List[float]:
        """Compute similarities against a specific subset of UIDs."""
        q = Vec.normalize(Vec.pad_trim(query, self.dim))
        results = []
        with self._lock:
            for uid in uids:
                if uid in self._store:
                    results.append(Vec.cosine(q, self._store[uid]))
                else:
                    results.append(0.0)
        return results

# ═══════════════════════════════════════════════════════════════════════════════
# §7  ENCODER — hierarchical multi-modal representation
# ═══════════════════════════════════════════════════════════════════════════════

class EncoderCache:
    """Thread-safe LRU cache for encoded embeddings."""
    def __init__(self, capacity: int):
        self._cap = capacity
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
        return None

    def put(self, key: str, val: Any) -> None:
        with self._lock:
            self._store[key] = val
            self._store.move_to_end(key)
            if len(self._store) > self._cap:
                self._store.popitem(last=False)

    def __len__(self) -> int: return len(self._store)


class CognitiveEncoder:
    """
    Multi-level encoder that maps raw inputs to the unified engram space.

    Pipeline:
      1. Raw → embed_dim vector (SentenceTransformer or hash fallback).
      2. That vector is projected through 4 world model levels via
         Oja-updated linear maps (online PCA), yielding a multi-scale
         representation useful for predictive coding.

    All levels share the same cache, keyed by text.
    """

    def __init__(self, cfg: Optional[NOESISConfig] = None):
        self.cfg   = cfg or get_cfg()
        self.log   = ContextualLogger("Encoder")
        self._model = None
        self._cache = EncoderCache(self.cfg.embed_cache_size)
        self._lock  = threading.Lock()

        # Oja projection matrices: W[l] shape (wm_dims[l+1], wm_dims[l])
        # L0 input = embed_dim, subsequent inputs are wm_dims[l-1]
        dims = [self.cfg.embed_dim] + list(self.cfg.wm_dims)
        self._oja_W: List[Any] = []
        self._oja_mom: List[Any] = []
        for l in range(len(self.cfg.wm_dims)):
            din  = dims[l]
            dout = dims[l + 1]
            self._oja_W.append(Vec.rand_mat(dout, din, scale=1.0 / math.sqrt(din)))
            self._oja_mom.append(Vec.zeros_mat(dout, din))

        self._oja_lr  = 0.004
        self._oja_mom_rate = 0.08

        self._try_load_model()
        self.log.info(
            "CognitiveEncoder ready — embed_dim=%d levels=%d model=%s",
            self.cfg.embed_dim, len(self.cfg.wm_dims),
            "SentenceTransformer" if self._model else "HashFallback"
        )

    def _try_load_model(self) -> None:
        if not HAS_ST: return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.cfg.embed_model)
            self.log.info("SentenceTransformer loaded: %s", self.cfg.embed_model)
        except Exception as exc:
            self.log.warning("ST load failed (%s) — using hash embeddings", exc)

    def encode(self, text: str) -> Any:
        """Encode text → embed_dim unit vector (cached)."""
        key = hashlib.sha1(text.encode()).hexdigest()
        cached = self._cache.get(key)
        if cached is not None: return cached
        vec = self._encode_raw(text)
        self._cache.put(key, vec)
        return vec

    def _encode_raw(self, text: str) -> Any:
        if self._model is not None:
            try:
                raw = self._model.encode(text, convert_to_numpy=True)
                return Vec.normalize(Vec.pad_trim(raw, self.cfg.embed_dim))
            except Exception:
                pass
        return Vec.hash_unit(text, self.cfg.embed_dim)

    def encode_obs(self, obs: Any) -> Any:
        """Polymorphic encoder: str, dict, numeric array."""
        if isinstance(obs, str):      return self.encode(obs)
        if isinstance(obs, dict):     return self.encode(json.dumps(obs, sort_keys=True, default=str))
        if isinstance(obs, (int, float)): return self.encode(str(obs))
        try:
            arr = Vec.pad_trim(obs, self.cfg.embed_dim)
            return Vec.normalize(arr)
        except Exception:
            return self.encode(str(obs))

    def project_levels(self, raw: Any, update_weights: bool = False) -> List[Any]:
        """
        Project embedding through all Oja levels.
        Returns list of per-level activations.
        If update_weights=True, apply one Oja learning step.
        """
        x = Vec.pad_trim(raw, self.cfg.embed_dim)
        activations: List[Any] = []
        for l in range(len(self.cfg.wm_dims)):
            W   = self._oja_W[l]
            y   = Vec.normalize(Vec.matvec(W, x))
            activations.append(y)
            if update_weights:
                self._oja_update(l, x, y)
            x = y
        return activations

    def _oja_update(self, level: int, x: Any, y: Any) -> None:
        """
        Oja's rule (momentum variant):
          ΔW_i = η (y_i * x - y_i² * W_i)
        Applied with momentum for stability.
        """
        lr  = self._oja_lr
        mom = self._oja_mom_rate
        W   = self._oja_w[level]
        Wm  = self._oja_mom[level]
        if HAS_NUMPY:
            xv  = np.asarray(x, np.float32)
            yv  = np.asarray(y, np.float32)
            Wn  = np.asarray(W,  np.float32)
            Wmn = np.asarray(Wm, np.float32)
            dW  = np.outer(yv, xv) - (yv[:, None] ** 2) * Wn
            new_mom = mom * Wmn + lr * dW
            self._oja_w[level]   = (Wn + new_mom).tolist()
            self._oja_mom[level] = new_mom.tolist()
        # Python fallback omitted for brevity (Oja is numpy-centric anyway)

    def tokenise(self, text: str) -> List[str]:
        """Simple BM25-ready token list (stopword-filtered)."""
        STOPS = {
            "the","a","an","is","are","was","were","be","been","have","has",
            "do","does","did","will","would","could","should","may","might",
            "i","you","he","she","it","we","they","and","or","but","in",
            "on","at","to","for","of","with","from","by","as","so","if",
            "not","no","this","that","these","those","my","your","our","their",
        }
        tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
        return [t for t in tokens if t not in STOPS and len(t) > 1][:512]

    @property
    def cache_stats(self) -> Dict[str, int]:
        return {"cached_embeddings": len(self._cache)}


# ═══════════════════════════════════════════════════════════════════════════════
# §8  TEMPORAL NARRATIVE GRAPH (TNG)
#     Episodes as nodes in a causal-temporal graph, not a flat list.
#     This is the core differentiator from CORTICEX/HELIOS:
#     Memory = living graph, not an indexed array.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Engram:
    """
    A single memory trace — the atomic unit of experience.

    Grounded in engram cell theory (Tonegawa et al. 2015):
    a sparse set of neurons that when reactivated, reconstitutes the
    experience. Here, embedding is the compressed pattern, and
    reconsolidation_window tracks whether the engram is actively
    open for modification.

    Extended with narrative fields that CORTICEX/HELIOS lack:
    - narrative_role: how this engram fits into the life story
    - emotional_valence: affective tag
    - autobio_salience: importance to self-identity
    """
    uid:               str
    content:           str
    embedding:         Any       # shape: (embed_dim,)
    kind:              MemoryKind = MemoryKind.EPISODIC
    session_id:        str = "default"
    created_at:        float = field(default_factory=time.time)
    last_accessed:     float = field(default_factory=time.time)
    access_count:      int   = 0
    importance:        float = 0.5
    emotional_valence: float = 0.0   # [-1, 1]
    arousal:           float = 0.5   # [0, 1]
    surprise_at_enc:   float = 0.0
    autobio_salience:  float = 0.0   # [0,1] — how identity-defining this is
    narrative_role:    str   = "event"   # event | insight | decision | turning_point | habit
    consolidation:     ConsolidationPhase = ConsolidationPhase.LABILE
    cortical_strength: float = 0.0   # 0=hippocampal → 1=neocortical
    reconsolidation_open: bool = False   # True = currently modifiable
    stability_h:       float = 4.0   # Ebbinghaus stability parameter
    q_value:           float = 0.5   # RL value of recalling this engram
    tags:              List[str] = field(default_factory=list)
    source_step:       int   = 0
    metadata:          Dict  = field(default_factory=dict)

    @property
    def age_h(self) -> float:
        return (time.time() - self.created_at) / 3600.0

    @property
    def retrievability(self) -> float:
        """
        Ebbinghaus forgetting curve: R = e^{-t/S}
        Modified with cortical_strength boost (consolidated memories resist forgetting).
        """
        elapsed_h = (time.time() - self.last_accessed) / 3600.0
        base_r = math.exp(-elapsed_h / max(self.stability_h, EPS))
        return min(1.0, base_r + 0.3 * self.cortical_strength)

    @property
    def composite_priority(self) -> float:
        """Multi-factor priority for replay selection."""
        return (
            0.30 * self.importance +
            0.20 * (1.0 - self.cortical_strength) +  # unconsolidated = higher priority
            0.20 * abs(self.emotional_valence) +
            0.15 * self.retrievability +
            0.10 * min(1.0, self.access_count / 10.0) +
            0.05 * self.autobio_salience
        )

    def activate(self, retrieval_strength: float = 1.0) -> None:
        """Record retrieval; update stability via spaced repetition (SM-2-like)."""
        self.last_accessed = time.time()
        self.access_count  += 1
        q = min(1.0, retrieval_strength)
        if q < 0.3:
            self.stability_h = max(0.5, self.stability_h * 0.5)
        elif q < 0.6:
            self.stability_h *= (0.8 + q * 0.4)
        else:
            self.stability_h *= (1.2 + q * 1.8)
        self.importance = min(1.0, self.importance + 0.008 * retrieval_strength)

    def apply_consolidation_cascade(self) -> None:
        """
        Simulate molecular consolidation cascade over time.
        CREB → CAMTA1 → TCF4 → ASH1L / BDNF → protein synthesis → cortical transfer.
        Reference: Bhatt et al. (2020); Haubrich & Bhatt (2020).
        """
        age = self.age_h
        # Each cascade stage has a time window and decay
        creb   = max(0.0, 1.0 - age / 3.0)          # 0-3h
        camta1 = min(1.0, age / 2.0) * max(0.0, 1.0 - age / 8.0)  # 2-8h
        tcf4   = min(1.0, max(0.0, (age - 4.0) / 6.0)) * max(0.0, 1.0 - age / 30.0)
        ash1l  = min(1.0, max(0.0, (age - 20.0) / 100.0))  # 20h+

        cascade = 0.15 * creb + 0.30 * camta1 + 0.30 * tcf4 + 0.25 * ash1l
        self.cortical_strength = min(1.0, cascade)

        if age >= 168:    self.consolidation = ConsolidationPhase.REMOTE
        elif tcf4 > 0.2:  self.consolidation = ConsolidationPhase.STABLE
        elif camta1 > 0.2: self.consolidation = ConsolidationPhase.EARLY
        else:              self.consolidation = ConsolidationPhase.LABILE


@dataclass
class TNGEdge:
    """
    An edge in the Temporal Narrative Graph.
    src → dst with typed relationship and weighted strength.
    """
    src:       str
    dst:       str
    kind:      EdgeKind
    weight:    float = 1.0
    created_at: float = field(default_factory=time.time)
    last_activated: float = field(default_factory=time.time)
    metadata:  Dict  = field(default_factory=dict)

    @property
    def age_h(self) -> float:
        return (time.time() - self.last_activated) / 3600.0


class TemporalNarrativeGraph:
    """
    The Temporal Narrative Graph (TNG):
    Episodes (Engrams) as nodes; temporal, causal, associative, contrast,
    and reflective edges connect them into a coherent life story.

    This is the central innovation over flat episodic stores:
    - Causal inference: when a new engram is stored, we check if it was
      caused by recent engrams (high temporal proximity + semantic sim).
    - Contrast detection: when a new engram contradicts a recent one
      (negative cosine), a CONTRAST edge is created — this is the
      source of cognitive dissonance and triggers reflection.
    - Spreading activation: recall a node → activate its neighbourhood →
      surfaces related memories that are semantically + causally linked.
    - Narrative coherence: the path through the graph from earliest to
      latest engram is the agent's "life story".

    Backed by SQLite for persistence; VectorIndex for ANN retrieval.
    """

    def __init__(self, cfg: Optional[NOESISConfig] = None):
        self.cfg   = cfg or get_cfg()
        self.log   = ContextualLogger("TNG")
        self._lock = threading.RLock()

        # In-memory stores
        self._nodes: Dict[str, Engram]  = {}   # uid → Engram
        self._edges: Dict[str, TNGEdge] = {}   # edge_uid → TNGEdge
        self._adj_out:  Dict[str, Set[str]] = defaultdict(set)  # node → set(edge_uid)
        self._adj_in:   Dict[str, Set[str]] = defaultdict(set)
        self._timeline: deque[str] = deque()   # ordered insertion (temporal axis)

        # Semantic index for ANN retrieval
        self._index = VectorIndex(
            dim=self.cfg.embed_dim,
            nlist=self.cfg.faiss_nlist,
            nprobe=self.cfg.faiss_nprobe,
        )

        # SQLite persistence
        db_path = str(self.cfg.data_root / "db" / "tng.db")
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._setup_db()
        self._load_from_db()
        self.log.info("TNG ready — %d nodes, %d edges", len(self._nodes), len(self._edges))

    def _setup_db(self) -> None:
        c = self._conn
        c.execute("""CREATE TABLE IF NOT EXISTS engrams (
            uid TEXT PRIMARY KEY, content TEXT, kind TEXT, session_id TEXT,
            created_at REAL, last_accessed REAL, access_count INTEGER,
            importance REAL, emotional_valence REAL, arousal REAL, surprise_at_enc REAL,
            autobio_salience REAL, narrative_role TEXT, consolidation TEXT,
            cortical_strength REAL, stability_h REAL, q_value REAL,
            tags TEXT, source_step INTEGER, metadata TEXT, embedding BLOB
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS tng_edges (
            uid TEXT PRIMARY KEY, src TEXT, dst TEXT, kind TEXT,
            weight REAL, created_at REAL, last_activated REAL, metadata TEXT
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tng_src ON tng_edges(src)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tng_dst ON tng_edges(dst)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_eng_session ON engrams(session_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_eng_importance ON engrams(importance DESC)")
        c.commit()

    def _load_from_db(self) -> None:
        """Load recent engrams and edges into memory on startup."""
        cur = self._conn.execute(
            "SELECT uid, embedding FROM engrams ORDER BY created_at DESC LIMIT 8000"
        )
        for uid, blob in cur.fetchall():
            if blob and HAS_NUMPY:
                vec = np.frombuffer(blob, np.float32).copy()
            else:
                vec = Vec.zeros(self.cfg.embed_dim)
            self._index.add(uid, vec)
        # Load full recent nodes into hot cache
        cur2 = self._conn.execute(
            "SELECT * FROM engrams ORDER BY last_accessed DESC LIMIT 2000"
        )
        cols = [d[0] for d in cur2.description]
        for row in cur2.fetchall():
            d = dict(zip(cols, row))
            eng = self._row_to_engram(d)
            if eng:
                with self._lock:
                    self._nodes[eng.uid] = eng
                    self._timeline.appendleft(eng.uid)
        # Load edges
        cur3 = self._conn.execute("SELECT * FROM tng_edges ORDER BY created_at DESC LIMIT 20000")
        cols3 = [d[0] for d in cur3.description]
        for row in cur3.fetchall():
            d = dict(zip(cols3, row))
            edge = TNGEdge(
                src=d["src"], dst=d["dst"],
                kind=EdgeKind(d["kind"]), weight=float(d["weight"]),
                created_at=float(d["created_at"]),
                last_activated=float(d["last_activated"]),
                metadata=json.loads(d.get("metadata", "{}")),
            )
            with self._lock:
                self._edges[d["uid"]] = edge
                self._adj_out[d["src"]].add(d["uid"])
                self._adj_in[d["dst"]].add(d["uid"])

    def _row_to_engram(self, d: Dict) -> Optional[Engram]:
        try:
            blob = d.pop("embedding", None)
            if blob and HAS_NUMPY:
                vec = np.frombuffer(blob, np.float32).copy()
            else:
                vec = Vec.zeros(self.cfg.embed_dim)
            return Engram(
                uid=d["uid"], content=d["content"],
                kind=MemoryKind(d.get("kind", "episodic")),
                embedding=vec, session_id=d.get("session_id", "default"),
                created_at=float(d.get("created_at", time.time())),
                last_accessed=float(d.get("last_accessed", time.time())),
                access_count=int(d.get("access_count", 0)),
                importance=float(d.get("importance", 0.5)),
                emotional_valence=float(d.get("emotional_valence", 0.0)),
                arousal=float(d.get("arousal", 0.5)),
                surprise_at_enc=float(d.get("surprise_at_enc", 0.0)),
                autobio_salience=float(d.get("autobio_salience", 0.0)),
                narrative_role=d.get("narrative_role", "event"),
                consolidation=ConsolidationPhase(d.get("consolidation", "labile")),
                cortical_strength=float(d.get("cortical_strength", 0.0)),
                stability_h=float(d.get("stability_h", 4.0)),
                q_value=float(d.get("q_value", 0.5)),
                tags=json.loads(d.get("tags", "[]")),
                source_step=int(d.get("source_step", 0)),
                metadata=json.loads(d.get("metadata", "{}")),
            )
        except Exception as exc:
            self.log.warning("Failed to load engram: %s", exc)
            return None

    # ── core operations ────────────────────────────────────────────────────────

    def store(
        self,
        content: str,
        embedding: Any,
        kind: MemoryKind = MemoryKind.EPISODIC,
        session_id: str = "default",
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        arousal: float = 0.5,
        surprise: float = 0.0,
        autobio_salience: float = 0.0,
        narrative_role: str = "event",
        tags: Optional[List[str]] = None,
        source_step: int = 0,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Store a new engram. Automatically creates TNG edges by:
        1. Pattern separation: if too similar to a recent engram → update that one.
        2. Temporal edge to the most recent engram in the same session.
        3. Causal edge to recent engrams with high semantic similarity.
        4. Contrast edge if semantic similarity is strongly negative.
        5. Associative edges via Hebbian co-activation with retrieved context.
        """
        # Pattern separation
        neighbors = self._index.search(embedding, k=3)
        if neighbors and neighbors[0][1] > 0.88:
            # Too similar — update instead of creating new
            uid = neighbors[0][0]
            eng = self._load_engram(uid)
            if eng:
                eng.activate(importance)
                eng.importance = min(1.0, eng.importance + 0.04)
                self._persist_engram(eng)
                return uid

        uid = f"eng_{uuid.uuid4().hex[:18]}"
        eng = Engram(
            uid=uid, content=content, embedding=embedding,
            kind=kind, session_id=session_id,
            importance=importance, emotional_valence=emotional_valence,
            arousal=arousal, surprise_at_enc=surprise,
            autobio_salience=autobio_salience,
            narrative_role=narrative_role,
            tags=tags or [], source_step=source_step,
            metadata=metadata or {},
        )
        eng.apply_consolidation_cascade()
        self._persist_engram(eng)
        self._index.add(uid, embedding)
        with self._lock:
            self._nodes[uid] = eng
            self._timeline.append(uid)
            # Evict oldest from hot cache if too large
            if len(self._nodes) > self.cfg.tng_max_nodes // 2:
                oldest = next(iter(self._timeline))
                self._nodes.pop(oldest, None)
                self._timeline.popleft()

        # ── create narrative edges ─────────────────────────────────────────────
        recent_uids = list(self._timeline)[-8:]  # last 8 nodes

        # Temporal: link to immediately preceding
        if len(recent_uids) >= 2:
            prev_uid = recent_uids[-2]
            self._add_edge(prev_uid, uid, EdgeKind.TEMPORAL, weight=1.0)

        # Causal & contrast: check semantic relationships with recent
        for r_uid in recent_uids[:-1]:
            r_eng = self._load_engram(r_uid)
            if r_eng is None: continue
            sim = Vec.cosine(embedding, r_eng.embedding)
            time_diff = abs(eng.created_at - r_eng.created_at)
            if sim > self.cfg.tng_causal_threshold and time_diff < self.cfg.tng_temporal_window_s:
                self._add_edge(r_uid, uid, EdgeKind.CAUSAL, weight=sim)
            elif sim < -0.15:
                self._add_edge(r_uid, uid, EdgeKind.CONTRAST, weight=abs(sim))

        return uid

    def retrieve(
        self,
        cue: Any,
        k: int = 10,
        session_id: Optional[str] = None,
        min_importance: float = 0.0,
        kinds: Optional[List[MemoryKind]] = None,
    ) -> List[Tuple[Engram, float]]:
        """
        Pattern completion: retrieve k most relevant engrams.
        Scoring = cosine_similarity × retrievability × importance_bonus × cortical_bonus.
        """
        candidates = self._index.search(cue, k=k * 4)
        results: List[Tuple[Engram, float]] = []
        for uid, raw_sim in candidates:
            eng = self._load_engram(uid)
            if eng is None: continue
            if session_id and eng.session_id != session_id: continue
            if eng.importance < min_importance: continue
            if kinds and eng.kind not in kinds: continue
            score = (
                raw_sim *
                (0.6 + 0.25 * eng.retrievability) *
                (1.0 + 0.2 * eng.importance) *
                (1.0 + 0.15 * eng.cortical_strength)
            )
            results.append((eng, score))
        results.sort(key=lambda x: -x[1])
        # Activate top results
        for eng, score in results[:k]:
            eng.activate(score)
            self._persist_engram(eng)
        return results[:k]

    def spreading_activation(
        self, seed_uid: str, hops: int = 3, decay: float = 0.6
    ) -> Dict[str, float]:
        """
        BFS spreading activation through the TNG.
        Returns {uid: activation_strength}.
        Used for nostalgia replay and contextual priming.
        """
        visited: Dict[str, float] = {seed_uid: 1.0}
        frontier = [(seed_uid, 1.0)]
        for hop in range(hops):
            next_f = []
            for uid, strength in frontier:
                with self._lock:
                    out_edges = list(self._adj_out.get(uid, set()))
                    in_edges  = list(self._adj_in.get(uid, set()))
                for eid in out_edges + in_edges:
                    edge = self._edges.get(eid)
                    if edge is None: continue
                    nbr = edge.dst if edge.src == uid else edge.src
                    if nbr in visited: continue
                    # Weight decay by hop depth + edge type
                    type_weight = {
                        EdgeKind.CAUSAL: 1.0, EdgeKind.TEMPORAL: 0.8,
                        EdgeKind.ASSOCIATIVE: 0.7, EdgeKind.REFLECTS: 0.9,
                        EdgeKind.CONTRAST: 0.5, EdgeKind.SUBSUMES: 0.6,
                    }.get(edge.kind, 0.5)
                    new_strength = strength * decay * edge.weight * type_weight
                    visited[nbr] = new_strength
                    next_f.append((nbr, new_strength))
            frontier = sorted(next_f, key=lambda x: -x[1])[:24]
        return visited

    def narrative_path(
        self, start_uid: str, end_uid: str
    ) -> List[Tuple[str, EdgeKind]]:
        """
        Find shortest narrative path between two engrams in the TNG.
        Uses BFS over TEMPORAL and CAUSAL edges.
        Returns list of (uid, edge_kind) pairs.
        """
        if start_uid == end_uid: return [(start_uid, EdgeKind.TEMPORAL)]
        queue: deque[List[Tuple[str, Optional[EdgeKind]]]] = deque()
        queue.append([(start_uid, None)])
        visited: Set[str] = {start_uid}
        while queue:
            path = queue.popleft()
            curr_uid = path[-1][0]
            with self._lock:
                edge_uids = list(self._adj_out.get(curr_uid, set()))
            for eid in edge_uids:
                edge = self._edges.get(eid)
                if not edge: continue
                if edge.kind not in (EdgeKind.TEMPORAL, EdgeKind.CAUSAL): continue
                nbr = edge.dst
                if nbr in visited: continue
                new_path = path + [(nbr, edge.kind)]
                if nbr == end_uid:
                    return [(u, k) for u, k in new_path if k is not None]
                visited.add(nbr)
                queue.append(new_path)
        return []

    def add_reflection_edge(self, from_uid: str, to_uid: str, insight: str = "") -> None:
        """
        A REFLECTS edge connects a later engram (insight/reflection)
        back to an older engram it is about. This creates the narrative
        structure of 'learning from the past'.
        """
        self._add_edge(from_uid, to_uid, EdgeKind.REFLECTS, weight=0.9,
                       metadata={"insight": insight[:200]})

    def add_associative_edge(self, uid_a: str, uid_b: str, strength: float = 0.5) -> None:
        """Hebbian co-activation → associative edge."""
        self._add_edge(uid_a, uid_b, EdgeKind.ASSOCIATIVE, weight=strength)

    def _add_edge(
        self, src: str, dst: str, kind: EdgeKind,
        weight: float = 1.0, metadata: Optional[Dict] = None
    ) -> str:
        eid = f"e_{src[:8]}_{dst[:8]}_{kind.value}"
        with self._lock:
            if eid in self._edges:
                self._edges[eid].weight = min(5.0, self._edges[eid].weight + 0.1 * weight)
                self._edges[eid].last_activated = time.time()
                return eid
            edge = TNGEdge(
                src=src, dst=dst, kind=kind, weight=weight,
                metadata=metadata or {},
            )
            self._edges[eid] = edge
            self._adj_out[src].add(eid)
            self._adj_in[dst].add(eid)
        self._persist_edge(eid)
        return eid

    def decay_edges(self, decay_rate: Optional[float] = None) -> int:
        """Exponential decay of associative edge weights; remove dead edges."""
        dr = decay_rate or self.cfg.tng_edge_decay
        now = time.time()
        to_remove = []
        with self._lock:
            for eid, edge in self._edges.items():
                if edge.kind == EdgeKind.ASSOCIATIVE:
                    age_h = (now - edge.last_activated) / 3600.0
                    edge.weight *= math.exp(-dr * age_h)
                    if edge.weight < 0.01:
                        to_remove.append(eid)
            for eid in to_remove:
                e = self._edges.pop(eid, None)
                if e:
                    self._adj_out[e.src].discard(eid)
                    self._adj_in[e.dst].discard(eid)
        if to_remove:
            self._conn.execute(
                f"DELETE FROM tng_edges WHERE uid IN ({','.join('?' * len(to_remove))})",
                to_remove
            )
            self._conn.commit()
        return len(to_remove)

    def get_timeline(
        self, session_id: Optional[str] = None, limit: int = 50
    ) -> List[Engram]:
        """Return the most recent engrams in temporal order."""
        with self._lock:
            uids = list(self._timeline)
        results = []
        for uid in reversed(uids[-limit * 2:]):
            eng = self._load_engram(uid)
            if eng and (session_id is None or eng.session_id == session_id):
                results.append(eng)
                if len(results) >= limit: break
        return results

    def consolidation_pass(self) -> Dict[str, int]:
        """
        Offline consolidation:
        1. Replay high-priority engrams (update cascade, boost cortical_strength).
        2. Prune very low importance, fully decayed memories.
        3. Strengthen edges between frequently co-activated nodes.
        """
        stats = {"replayed": 0, "pruned": 0, "edges_decayed": 0}
        # Build priority queue
        with self._lock:
            candidates = list(self._nodes.values())
        if not candidates:
            cur = self._conn.execute(
                "SELECT uid FROM engrams ORDER BY importance DESC, last_accessed DESC LIMIT 300"
            )
            uids = [r[0] for r in cur.fetchall()]
            candidates = [e for u in uids if (e := self._load_engram(u)) is not None]
        candidates.sort(key=lambda e: -e.composite_priority)
        for eng in candidates[:self.cfg.replay_batch_size]:
            eng.apply_consolidation_cascade()
            eng.importance = min(1.0, eng.importance + 0.005)
            self._persist_engram(eng)
            stats["replayed"] += 1
        # Prune
        cur = self._conn.execute(
            "SELECT uid, importance, stability_h, last_accessed, consolidation FROM engrams "
            "WHERE consolidation != 'remote'"
        )
        for uid, imp, stab, last_acc, consol in cur.fetchall():
            elapsed_h = (time.time() - (last_acc or 0)) / 3600.0
            retrievability = math.exp(-elapsed_h / max(stab or 1.0, EPS))
            if imp < 0.08 and retrievability < 0.03:
                self._conn.execute("DELETE FROM engrams WHERE uid=?", (uid,))
                self._index.remove(uid)
                with self._lock: self._nodes.pop(uid, None)
                stats["pruned"] += 1
        self._conn.commit()
        stats["edges_decayed"] = self.decay_edges()
        return stats

    def stats(self) -> Dict[str, Any]:
        cur = self._conn.execute(
            "SELECT COUNT(*), AVG(importance), AVG(cortical_strength), "
            "SUM(access_count) FROM engrams"
        )
        row = cur.fetchone()
        cur2 = self._conn.execute("SELECT COUNT(*) FROM tng_edges")
        n_edges = cur2.fetchone()[0]
        return {
            "total_engrams": int(row[0] or 0),
            "avg_importance": round(float(row[1] or 0), 3),
            "avg_cortical_str": round(float(row[2] or 0), 3),
            "total_accesses": int(row[3] or 0),
            "total_edges": n_edges,
            "hot_cache": len(self._nodes),
        }

    # ── persistence helpers ────────────────────────────────────────────────────

    def _persist_engram(self, eng: Engram) -> None:
        blob = (np.asarray(eng.embedding, np.float32).tobytes() if HAS_NUMPY else None)
        self._conn.execute("""
            INSERT OR REPLACE INTO engrams VALUES (
                :uid,:content,:kind,:session_id,:created_at,:last_accessed,:access_count,
                :importance,:emotional_valence,:arousal,:surprise_at_enc,:autobio_salience,
                :narrative_role,:consolidation,:cortical_strength,:stability_h,:q_value,
                :tags,:source_step,:metadata,:embedding
            )""", {
            "uid": eng.uid, "content": eng.content, "kind": eng.kind.value,
            "session_id": eng.session_id, "created_at": eng.created_at,
            "last_accessed": eng.last_accessed, "access_count": eng.access_count,
            "importance": eng.importance, "emotional_valence": eng.emotional_valence,
            "arousal": eng.arousal, "surprise_at_enc": eng.surprise_at_enc,
            "autobio_salience": eng.autobio_salience, "narrative_role": eng.narrative_role,
            "consolidation": eng.consolidation.value, "cortical_strength": eng.cortical_strength,
            "stability_h": eng.stability_h, "q_value": eng.q_value,
            "tags": json.dumps(eng.tags), "source_step": eng.source_step,
            "metadata": json.dumps(eng.metadata), "embedding": blob,
        })
        self._conn.commit()

    def _persist_edge(self, eid: str) -> None:
        with self._lock:
            edge = self._edges.get(eid)
        if not edge: return
        self._conn.execute("""
            INSERT OR REPLACE INTO tng_edges VALUES (?,?,?,?,?,?,?,?)""", (
            eid, edge.src, edge.dst, edge.kind.value,
            edge.weight, edge.created_at, edge.last_activated,
            json.dumps(edge.metadata),
        ))
        self._conn.commit()

    def _load_engram(self, uid: str) -> Optional[Engram]:
        with self._lock:
            if uid in self._nodes: return self._nodes[uid]
        cur = self._conn.execute("SELECT * FROM engrams WHERE uid=?", (uid,))
        row = cur.fetchone()
        if not row: return None
        cols = [d[0] for d in cur.description]
        d = dict(zip(cols, row))
        eng = self._row_to_engram(d)
        if eng:
            with self._lock:
                self._nodes[eng.uid] = eng
        return eng

    def update_q_value(self, uid: str, reward: float, gamma: float = 0.95) -> None:
        eng = self._load_engram(uid)
        if eng:
            eng.q_value += 0.1 * (reward - eng.q_value)
            eng.q_value  = max(0.0, min(1.0, eng.q_value))
            self._persist_engram(eng)


# ═══════════════════════════════════════════════════════════════════════════════
# §9  NEUROMODULATORY FABRIC
#     DA/NE/ACh/5HT/GABA/GLU with metaplasticity tracking
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NTXState:
    """Instantaneous levels of all neuromodulators."""
    DA:   float = 0.40   # dopamine
    NE:   float = 0.28   # norepinephrine
    ACH:  float = 0.38   # acetylcholine
    HT:   float = 0.48   # serotonin (5-HT)
    GABA: float = 0.50   # inhibitory tone
    GLU:  float = 0.45   # glutamate (excitatory gain)

    def clip(self) -> None:
        for f in fields(self):
            setattr(self, f.name, max(0.0, min(1.0, getattr(self, f.name))))

    @property
    def learning_rate_scale(self) -> float:
        """DA boosts LR; 5HT stabilises; GABA inhibits."""
        return max(0.2, min(3.0, 0.6 + 1.2 * self.DA - 0.5 * self.HT - 0.3 * self.GABA))

    @property
    def exploration_drive(self) -> float:
        """DA + NE drives exploration; 5HT + GABA encourage exploitation."""
        return max(0.05, min(0.95, 0.10 + 0.45 * self.DA + 0.35 * self.NE
                               - 0.25 * self.HT - 0.10 * self.GABA))

    @property
    def attention_precision(self) -> float:
        """ACh sharpens representations; NE broadens; GLU amplifies."""
        return max(0.3, min(3.5, 0.8 + 1.5 * self.ACH + 0.5 * self.NE + 0.3 * self.GLU))

    @property
    def memory_gate(self) -> float:
        """
        ACh controls encoding gate (high ACh = deep encoding of new things).
        Reference: Dayan & Yu 2006 — ACh signals expected vs unexpected uncertainty.
        """
        return max(0.1, min(1.0, 0.3 + 1.4 * self.ACH - 0.4 * self.HT))

    @property
    def stability_factor(self) -> float:
        """5HT drives stability / patience. High 5HT = resist updating beliefs."""
        return max(0.1, min(1.0, 0.3 + 1.2 * self.HT + 0.3 * self.GABA))

    @property
    def cognitive_mode(self) -> CognitiveMode:
        if self.NE > 0.72 and self.DA < 0.28: return CognitiveMode.CRITICAL
        if self.DA > 0.68 and self.NE > 0.55: return CognitiveMode.EXPANSIVE
        if self.HT > 0.65 and self.GABA > 0.60: return CognitiveMode.DREAMING
        if self.ACH > 0.60 and self.NE < 0.35: return CognitiveMode.FOCUSED
        if self.DA < 0.18 and self.HT > 0.58: return CognitiveMode.REFLEX
        return CognitiveMode.FOCUSED

    def to_dict(self) -> Dict[str, float]:
        return {f.name: round(getattr(self, f.name), 4) for f in fields(self)}


class NeuromodulatoryFabric:
    """
    Manages all neuromodulator dynamics.

    Key additions over CORTICEX/HELIOS:
    1. GABA / GLU: inhibitory / excitatory balance.
    2. Metaplasticity tracking: sustained high activity of a neuromodulator
       over the next N steps (opponent process theory).
    3. Neuromodulatory history for introspection and pattern detection.
    4. Circadian-like slow oscillation affecting HT/ACh (optional).

    All events are event-driven; per-step reuptake decay to baseline.

    References:
    - Dayan & Yu (2006) NE/ACh as uncertainty signals
    - Schultz et al. (1997) DA as RPE
    - Friston et al. (2012) DA as affordance / active inference
    - Bhatt & colleagues (2020) metaplasticity via accumulated kinase activity
    """

    def __init__(self, cfg: Optional[NOESISConfig] = None):
        self.cfg  = cfg or get_cfg()
        self.log  = ContextualLogger("Neuromod")
        self.state = NTXState(**{k: v for k, v in self.cfg.ntx_baseline.items()})
        self._baseline = NTXState(**{k: v for k, v in self.cfg.ntx_baseline.items()})
        self._history: deque[NTXState] = deque(maxlen=1000)
        self._last_decay = time.time()
        self._step_count = 0
        # Metaplasticity accumulators (per neuromodulator)
        self._meta_acc: Dict[str, float] = {k: 0.0 for k in self.cfg.ntx_baseline}
        self._meta_window = self.cfg.metaplasticity_window

    def decay_step(self, dt_s: float = 1.0) -> None:
        """Exponential reuptake toward baseline + record history."""
        r = math.exp(-self.cfg.ntx_reuptake * dt_s)
        bl = self._baseline
        s  = self.state
        s.DA   = bl.DA   + (s.DA   - bl.DA)   * r
        s.NE   = bl.NE   + (s.NE   - bl.NE)   * r
        s.ACH  = bl.ACH  + (s.ACH  - bl.ACH)  * r
        s.HT   = bl.HT   + (s.HT   - bl.HT)   * r
        s.GABA = bl.GABA + (s.GABA - bl.GABA) * r
        s.GLU  = bl.GLU  + (s.GLU  - bl.GLU)  * r
        s.clip()
        self._history.append(NTXState(**s.to_dict()))
        self._update_metaplasticity()
        self._step_count += 1

    # ── event-driven releases ──────────────────────────────────────────────────

    def on_reward(self, magnitude: float) -> None:
        """Positive RPE → phasic DA burst + modest 5HT."""
        m = max(0.0, min(1.0, magnitude))
        self.state.DA  += 0.55 * m
        self.state.HT  += 0.18 * m
        self.state.GLU += 0.12 * m
        self.state.NE  += 0.08 * m
        self.state.clip()

    def on_negative_rpe(self, magnitude: float) -> None:
        """Negative RPE → DA dip, NE spike, GABA rises (suppression)."""
        m = max(0.0, min(1.0, magnitude))
        self.state.DA   -= 0.45 * m
        self.state.NE   += 0.40 * m
        self.state.HT   -= 0.18 * m
        self.state.GABA += 0.20 * m
        self.state.clip()

    def on_novelty(self, surprise: float) -> None:
        """Novel input → NE + ACh orienting response."""
        s = max(0.0, min(1.0, surprise))
        self.state.NE  += 0.50 * s
        self.state.ACH += 0.38 * s
        self.state.GLU += 0.15 * s
        self.state.clip()

    def on_threat(self, severity: float) -> None:
        """Threat / error → stress response: NE spike, DA dip, HT dip."""
        s = max(0.0, min(1.0, severity))
        self.state.NE   += 0.65 * s
        self.state.DA   -= 0.30 * s
        self.state.HT   -= 0.28 * s
        self.state.GABA += 0.15 * s
        self.state.clip()

    def on_prediction_error(self, pe: float) -> None:
        """General PE → precision re-calibration via ACh + NE."""
        pe = max(0.0, min(1.0, pe))
        self.state.ACH += 0.45 * pe
        self.state.NE  += 0.22 * pe
        self.state.GLU += 0.10 * pe
        self.state.clip()

    def on_consolidation(self) -> None:
        """Offline consolidation / dreaming state: boost HT + GABA, lower NE."""
        self.state.HT   = min(1.0, self.state.HT   + 0.20)
        self.state.GABA = min(1.0, self.state.GABA + 0.18)
        self.state.NE   = max(0.0, self.state.NE   - 0.25)
        self.state.ACH  = max(0.0, self.state.ACH  - 0.15)
        self.state.clip()

    def on_nostalgia(self) -> None:
        """Nostalgia replay: slight DA + HT increase (rewarding remembrance)."""
        self.state.DA += 0.10
        self.state.HT += 0.12
        self.state.clip()

    # ── metaplasticity ────────────────────────────────────────────────────────

    def _update_metaplasticity(self) -> None:
        """
        Metaplasticity: sustained high activity of a neuromodulator
        leads to compensatory reduction in its effect over time.
        Mirrors BCM theory at the neuromodulatory level.
        """
        alpha = 2.0 / self._meta_window
        s = self.state
        for ntx_name, level in [("DA", s.DA), ("NE", s.NE), ("ACH", s.ACH)]:
            excess = max(0.0, level - getattr(self._baseline, ntx_name))
            self._meta_acc[ntx_name] = (1 - alpha) * self._meta_acc[ntx_name] + alpha * excess
        # If sustained excess → lower baseline slightly (opponent process)
        for ntx_name in ("DA", "NE"):
            meta = self._meta_acc.get(ntx_name, 0.0)
            if meta > 0.25:
                bl_level = getattr(self._baseline, ntx_name)
                setattr(self._baseline, ntx_name, max(0.15, bl_level - 0.0002))

    @property
    def effective_lr_scale(self) -> float:
        """Learning rate modulated by both state and metaplasticity."""
        base = self.state.learning_rate_scale
        meta_penalty = min(0.5, sum(self._meta_acc.values()) * 0.1)
        return max(0.2, base - meta_penalty)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "levels": self.state.to_dict(),
            "mode": self.state.cognitive_mode.value,
            "lr_scale": round(self.effective_lr_scale, 4),
            "exploration": round(self.state.exploration_drive, 4),
            "attention_precision": round(self.state.attention_precision, 4),
            "memory_gate": round(self.state.memory_gate, 4),
            "meta_acc": {k: round(v, 4) for k, v in self._meta_acc.items()},
        }

    def history_means(self, window: int = 50) -> Dict[str, float]:
        recent = list(self._history)[-window:]
        if not recent: return {}
        return {
            f.name: round(sum(getattr(s, f.name) for s in recent) / len(recent), 4)
            for f in fields(NTXState)
        }

# ═══════════════════════════════════════════════════════════════════════════════
# §10  HIERARCHICAL WORLD MODEL — 4-level predictive coding
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PredictiveCodingLevel:
    """
    A single level of the 4-level hierarchical world model.

    Level 0 (Perception): raw sensory features ~ V1/V2
    Level 1 (Concept):    object / pattern level ~ V4/IT
    Level 2 (Narrative):  event / story structure ~ PFC temporal
    Level 3 (Self):       self-model, identity, values ~ vmPFC / mPFC

    Each level:
    - Maintains a hidden state vector (representation)
    - Generates a top-down prediction for the level below
    - Receives bottom-up prediction error from the level below
    - Updates weights via Hebbian + BCM rule

    Free energy at each level = precision-weighted squared PE.
    Total free energy = Σ level FE.
    """
    level_idx: int
    dim_in:    int
    dim_out:   int
    precision: float
    lr:        float

    # State vectors
    representation: Any = None
    prediction:     Any = None   # for level below
    error:          Any = None

    # Weight matrices
    W_up:   Any = None   # bottom-up (error → next level)
    W_down: Any = None   # top-down (representation → prediction)
    bias:   Any = None

    # BCM threshold
    theta:  Any = None

    # History
    error_history: deque = field(default_factory=lambda: deque(maxlen=32))

    def __post_init__(self):
        if HAS_NUMPY:
            rng = np.random.default_rng(42 + self.level_idx)
            sc_up   = 1.0 / math.sqrt(self.dim_in)
            sc_down = 1.0 / math.sqrt(self.dim_out)
            self.W_up    = rng.normal(0, sc_up,   (self.dim_in,  self.dim_out)).astype(np.float32)
            self.W_down  = rng.normal(0, sc_down, (self.dim_out, self.dim_in )).astype(np.float32)
            self.bias    = np.zeros(self.dim_out, dtype=np.float32)
            self.representation = np.zeros(self.dim_out, dtype=np.float32)
            self.prediction     = np.zeros(self.dim_in,  dtype=np.float32)
            self.error          = np.zeros(self.dim_in,  dtype=np.float32)
            self.theta          = np.ones( self.dim_out, dtype=np.float32) * 0.05
        else:
            sc = 1.0 / math.sqrt(self.dim_in)
            rng = random.Random(42 + self.level_idx)
            self.W_up   = [[rng.gauss(0, sc) for _ in range(self.dim_out)] for _ in range(self.dim_in)]
            self.W_down = [[rng.gauss(0, sc) for _ in range(self.dim_in)] for _ in range(self.dim_out)]
            self.bias   = [0.0] * self.dim_out
            self.representation = [0.0] * self.dim_out
            self.prediction     = [0.0] * self.dim_in
            self.error          = [0.0] * self.dim_in
            self.theta          = [0.05] * self.dim_out

    def bottom_up(self, input_signal: Any) -> Any:
        """
        Compute prediction error and propagate up.
        Returns bottom-up signal (weighted error for next level).
        """
        if HAS_NUMPY:
            inp  = np.asarray(input_signal, np.float32).flatten()[:self.dim_in]
            if len(inp) < self.dim_in:
                inp = np.pad(inp, (0, self.dim_in - len(inp)))
            pred = np.asarray(self.prediction, np.float32)
            err  = (inp - pred) * self.precision
            self.error = err
            self.error_history.append(float(np.dot(err, err)))
            # Bottom-up signal: W_up^T · err
            return self.W_up.T @ err
        n = self.dim_in
        inp  = Vec.pad_trim(input_signal, n)
        pred = list(self.prediction)
        err  = [(float(inp[i]) - float(pred[i])) * self.precision for i in range(n)]
        self.error = err
        self.error_history.append(sum(e * e for e in err))
        return [sum(self.W_up[i][j] * err[i] for i in range(n)) for j in range(self.dim_out)]

    def top_down(self, td_signal: Any) -> Any:
        """
        Integrate top-down signal; generate new prediction; return it.
        """
        k = 0.12  # leaky integrator rate
        if HAS_NUMPY:
            td = np.asarray(td_signal, np.float32).flatten()[:self.dim_out]
            if len(td) < self.dim_out:
                td = np.pad(td, (0, self.dim_out - len(td)))
            # Bu contribution from error
            err = np.asarray(self.error, np.float32)
            bu_contrib = self.W_up.T @ err
            bu_contrib = bu_contrib.flatten()[:self.dim_out]
            if len(bu_contrib) < self.dim_out:
                bu_contrib = np.pad(bu_contrib, (0, self.dim_out - len(bu_contrib)))
            # Update representation: leaky integration
            old_r = np.asarray(self.representation, np.float32)
            new_r = (1 - k) * old_r + k * (td + bu_contrib + self.bias)
            # ReLU activation (non-negative firing rates)
            new_r = np.maximum(new_r, 0.0)
            self.representation = new_r
            # Generate prediction for level below
            pred = self.W_down @ new_r
            self.prediction = pred
            return pred
        else:
            d = self.dim_out
            old_r = list(self.representation)
            td_lst = Vec.pad_trim(td_signal, d)
            new_r = [max(0.0, (1 - k) * float(old_r[i]) + k * float(td_lst[i] if i < len(td_lst) else 0.0))
                     for i in range(d)]
            self.representation = new_r
            pred = Vec.matvec(self.W_down, new_r)
            self.prediction = pred
            return pred

    def update_weights(self, lr_scale: float = 1.0) -> None:
        """
        BCM + Hebbian weight update.
        ΔW_up  = lr · err · (rep - θ) · rep  (BCM-gated)
        ΔW_down = -lr · rep · err^T         (minimise prediction error)
        Δθ = bcm_lr · (||rep||² - θ)        (BCM sliding threshold)
        """
        lr = self.lr * lr_scale
        if not HAS_NUMPY: return
        err = np.asarray(self.error, np.float32).flatten()[:self.dim_in]
        if len(err) < self.dim_in:
            err = np.pad(err, (0, self.dim_in - len(err)))
        rep  = np.asarray(self.representation, np.float32).flatten()[:self.dim_out]
        if len(rep) < self.dim_out:
            rep = np.pad(rep, (0, self.dim_out - len(rep)))
        theta = np.asarray(self.theta, np.float32)

        # BCM gate: positive when rep > theta
        bcm_gate = rep - theta
        # W_up update: Hebbian with BCM gate
        dW_up = np.clip(np.outer(err, bcm_gate * rep), -0.05, 0.05)
        self.W_up = np.asarray(self.W_up, np.float32) + lr * dW_up
        # W_down update: anti-Hebbian on prediction error
        dW_down = np.clip(-np.outer(rep, err), -0.05, 0.05)
        self.W_down = np.asarray(self.W_down, np.float32) + lr * dW_down
        # BCM threshold: Δθ = bcm_lr · (||rep||² - θ)
        self.theta = np.clip(theta + 0.005 * (rep * rep - theta), 0.0, 2.0)

    @property
    def mean_error(self) -> float:
        return sum(self.error_history) / max(len(self.error_history), 1)

    @property
    def free_energy(self) -> float:
        if HAS_NUMPY:
            err = np.asarray(self.error, np.float32)
            return float(0.5 * self.precision * np.dot(err, err))
        return 0.5 * self.precision * sum(float(e) ** 2 for e in self.error)


class HierarchicalWorldModel:
    """
    4-level predictive coding hierarchy, inspired by Rao & Ballard (1999)
    and extended to include a narrative level and a self-model level.

    Level 0: Sensory (embed_dim → wm_dims[0])
    Level 1: Conceptual (wm_dims[0] → wm_dims[1])
    Level 2: Narrative (wm_dims[1] → wm_dims[2]) — new vs CORTICEX/HELIOS
    Level 3: Self (wm_dims[2] → wm_dims[3]) — self-model, identity

    The self-level representation (L3) becomes the agent's running
    autobiographical self-model, updated slowly with identity_update_lr.

    Each step:
      1. Bottom-up pass: error propagates up all levels.
      2. Top-down pass: predictions propagate down, update representations.
      3. Weight update at each level via BCM.
      4. Total free energy = Σ level FE.
      5. Surprise = 1 - exp(-FE_total / FE_baseline).
    """

    def __init__(self, cfg: Optional[NOESISConfig] = None, ntx: Optional[NeuromodulatoryFabric] = None):
        self.cfg = cfg or get_cfg()
        self.ntx = ntx
        self.log = ContextualLogger("WorldModel")
        self.levels: List[PredictiveCodingLevel] = []
        self._build()
        self._fe_history: deque[float] = deque(maxlen=self.cfg.wm_habituation_window)
        self._self_model: Any = Vec.zeros(self.cfg.wm_dims[3])
        self.log.info("WorldModel built: %s", " → ".join(
            f"{l.dim_in}→{l.dim_out}" for l in self.levels
        ))

    def _build(self) -> None:
        dims = [self.cfg.embed_dim] + list(self.cfg.wm_dims)
        for i in range(len(self.cfg.wm_dims)):
            self.levels.append(PredictiveCodingLevel(
                level_idx=i,
                dim_in=dims[i],
                dim_out=dims[i + 1],
                precision=self.cfg.wm_precision[i],
                lr=self.cfg.wm_lr[i],
            ))

    def process(self, obs_vec: Any) -> Tuple[List[Any], float, float]:
        """
        One full predictive coding cycle.

        Returns:
          (per_level_reps, total_free_energy, surprise)
        """
        lr_scale = self.ntx.effective_lr_scale if self.ntx else 1.0
        inp = Vec.pad_trim(obs_vec, self.cfg.embed_dim)

        # ── Bottom-up pass ─────────────────────────────────────────────────────
        bu_signal = inp
        bu_signals: List[Any] = [bu_signal]
        for lvl in self.levels:
            bu_signal = lvl.bottom_up(bu_signal)
            bu_signals.append(bu_signal)

        # ── Top-down pass ──────────────────────────────────────────────────────
        # Top-level prior: current self-model as seed
        if HAS_NUMPY:
            top_prior = np.asarray(self._self_model, np.float32)
        else:
            top_prior = list(self._self_model)
        td_signal = top_prior
        predictions: List[Any] = []
        for lvl in reversed(self.levels):
            pred = lvl.top_down(td_signal)
            predictions.insert(0, pred)
            td_signal = lvl.representation

        # ── Weight updates ─────────────────────────────────────────────────────
        for lvl in self.levels:
            lvl.update_weights(lr_scale)

        # ── Self-model update (Level 3 representation) ─────────────────────────
        top_rep = self.levels[-1].representation
        self_lr = self.cfg.identity_update_lr
        if HAS_NUMPY:
            sm = np.asarray(self._self_model, np.float32)
            tr = np.asarray(top_rep, np.float32).flatten()[:self.cfg.wm_dims[3]]
            if len(tr) < self.cfg.wm_dims[3]:
                tr = np.pad(tr, (0, self.cfg.wm_dims[3] - len(tr)))
            self._self_model = (1 - self_lr) * sm + self_lr * tr
        else:
            d = self.cfg.wm_dims[3]
            sm = list(self._self_model)
            tr = Vec.pad_trim(top_rep, d)
            self._self_model = [(1 - self_lr) * float(sm[i]) + self_lr * float(tr[i])
                                 for i in range(d)]

        # ── Compute aggregate free energy & surprise ───────────────────────────
        total_fe = sum(lvl.free_energy for lvl in self.levels)
        avg_fe   = sum(self._fe_history) / max(len(self._fe_history), 1) + EPS
        self._fe_history.append(total_fe)
        surprise = 1.0 - math.exp(-total_fe / (avg_fe * len(self.levels) + EPS))
        surprise = max(0.0, min(1.0, surprise))

        reps = [Vec.flatten(lvl.representation) for lvl in self.levels]
        return reps, total_fe, surprise

    def predict_next(self) -> Any:
        """Current Level-0 prediction for the next observation."""
        return self.levels[0].prediction

    @property
    def self_model(self) -> Any:
        """The running self-model (Level-3 representation)."""
        return self._self_model

    @property
    def world_state(self) -> Any:
        """Level-2 (Narrative) representation: current story frame."""
        return self.levels[2].representation if len(self.levels) > 2 else Vec.zeros(self.cfg.wm_dims[2])

    def level_free_energies(self) -> List[float]:
        return [round(lvl.free_energy, 5) for lvl in self.levels]


# ═══════════════════════════════════════════════════════════════════════════════
# §11  RESONANCE MEMORY ENGINE
#      Episodic + Semantic + Nostalgia + Reconsolidation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SemanticConcept:
    """A node in the semantic knowledge graph."""
    uid:        str
    label:      str
    embedding:  Any
    frequency:  int   = 1
    confidence: float = 0.7
    parent_uid: Optional[str] = None
    children:   List[str] = field(default_factory=list)
    attributes: Dict = field(default_factory=dict)

    def update(self, new_emb: Any, lr: float = 0.05) -> None:
        """Online centroid update: move toward new evidence."""
        if HAS_NUMPY:
            old = np.asarray(self.embedding, np.float32)
            nw  = np.asarray(new_emb, np.float32)
            updated = old + lr * (nw - old)
            self.embedding = Vec.normalize(updated)
        else:
            d = min(len(self.embedding), len(new_emb))
            updated = [float(self.embedding[i]) + lr * (float(new_emb[i]) - float(self.embedding[i]))
                       for i in range(d)]
            self.embedding = Vec.normalize(updated)
        self.frequency += 1


@dataclass
class NostalgiaTrace:
    """
    A nostalgia trace: a sequence of retrieved engrams forming
    a coherent 'memory journey' through the past.
    Each trace has a theme (embedding of the connecting thread)
    and a set of insights extracted from the journey.
    """
    uid:         str
    theme:       str
    theme_vec:   Any
    engram_uids: List[str]
    insights:    List[str] = field(default_factory=list)
    created_at:  float = field(default_factory=time.time)
    emotional_tone: float = 0.0   # mean valence across engrams
    richness:    float = 0.5      # how diverse the engrams are


class ResonanceMemoryEngine:
    """
    Multi-tier memory system with reconsolidation and nostalgia.

    Architecture:
    - TNG: the Temporal Narrative Graph (already defined above)
    - Semantic layer: concept graph updated via co-occurrence
    - Working memory: 7 ± 2 active slots with decay
    - Nostalgia engine: deliberate replay journeys through the TNG

    Key feature: RECONSOLIDATION
    When an engram is retrieved in the context of new information,
    it enters a reconsolidation window (2h) during which it can be
    modified. This models the Nader et al. (2000) reconsolidation effect:
    every time you remember something, you rewrite it slightly.

    This enables:
    - Updating old beliefs with new evidence
    - Emotional re-evaluation of past events
    - Creative recombination (surrealist memory blending)
    """

    def __init__(self, tng: TemporalNarrativeGraph, cfg: Optional[NOESISConfig] = None):
        self.cfg    = cfg or get_cfg()
        self.log    = ContextualLogger("Memory")
        self.tng    = tng
        self._lock  = threading.RLock()

        # Semantic layer
        self._concepts: Dict[str, SemanticConcept] = {}
        self._label_idx: Dict[str, str]             = {}
        self._sem_index = VectorIndex(
            dim=self.cfg.embed_dim,
            nlist=self.cfg.faiss_nlist // 2,
        )

        # Working memory
        self._wm_slots: List[Dict] = []   # list of {uid, content, emb, imp, ts}

        # Nostalgia traces (session)
        self._nostalgia_traces: List[NostalgiaTrace] = []

        # Reconsolidation queue: {uid: (open_until, new_context_emb)}
        self._reconsolidation_queue: Dict[str, Tuple[float, Any]] = {}

        self._setup_semantic_db()
        self.log.info("ResonanceMemoryEngine ready")

    def _setup_semantic_db(self) -> None:
        """Load/create semantic table in the same DB as TNG."""
        conn = self.tng._conn
        conn.execute("""CREATE TABLE IF NOT EXISTS semantic_concepts (
            uid TEXT PRIMARY KEY, label TEXT UNIQUE, frequency INTEGER,
            confidence REAL, parent_uid TEXT, attributes TEXT, embedding BLOB
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sem_label ON semantic_concepts(label)")
        conn.commit()
        # Load existing concepts
        cur = conn.execute("SELECT * FROM semantic_concepts LIMIT 5000")
        cols = [d[0] for d in cur.description]
        for row in cur.fetchall():
            d = dict(zip(cols, row))
            blob = d.pop("embedding", None)
            if blob and HAS_NUMPY:
                vec = np.frombuffer(blob, np.float32).copy()
            else:
                vec = Vec.zeros(self.cfg.embed_dim)
            # Parse attributes JSON
            attrs = json.loads(d.get("attributes", "{}")) if d.get("attributes") else {}
            c = SemanticConcept(
                uid=d["uid"], label=d["label"],
                embedding=vec,
                frequency=int(d.get("frequency", 1)),
                confidence=float(d.get("confidence", 0.7)),
                parent_uid=d.get("parent_uid"),
                attributes=attrs,
            )
            with self._lock:
                self._concepts[c.uid]      = c
                self._label_idx[c.label]   = c.uid
            self._sem_index.add(c.uid, vec)

    # ── episodic (delegate to TNG) ─────────────────────────────────────────────

    def store_episode(self, **kwargs) -> str:
        uid = self.tng.store(**kwargs)
        # Open reconsolidation window (2h from now)
        open_until = time.time() + self.cfg.reconsolidation_window_h * 3600
        with self._lock:
            self._reconsolidation_queue[uid] = (open_until, kwargs.get("embedding"))
        return uid

    def recall_episodes(self, cue: Any, k: int = 8, **kwargs) -> List[Tuple[Engram, float]]:
        results = self.tng.retrieve(cue, k=k, **kwargs)
        # Mark retrieved engrams for potential reconsolidation
        now = time.time()
        with self._lock:
            for eng, _ in results:
                if eng.uid not in self._reconsolidation_queue:
                    open_until = now + self.cfg.reconsolidation_window_h * 3600
                    self._reconsolidation_queue[eng.uid] = (open_until, cue)
                    eng.reconsolidation_open = True
        return results

    def reconsolidate(self, uid: str, new_context: Any, update_factor: float = 0.15) -> bool:
        """
        Reconsolidate an engram with new context.
        If the engram is within its reconsolidation window, blend its
        embedding with the new context vector. This is the neural mechanism
        by which retrieved memories are updated rather than merely recalled.

        Reference: Nader et al. (2000) Science; Haubrich & Bhatt (2020).
        """
        with self._lock:
            entry = self._reconsolidation_queue.get(uid)
        if entry is None: return False
        open_until, _ = entry
        if time.time() > open_until: return False  # window closed

        eng = self.tng._load_engram(uid)
        if not eng: return False
        # Blend embedding
        eng.embedding = Vec.normalize(Vec.lerp(eng.embedding, new_context, update_factor))
        # Mark stable phase disrupted → back to EARLY
        if eng.consolidation in (ConsolidationPhase.STABLE, ConsolidationPhase.REMOTE):
            eng.consolidation = ConsolidationPhase.EARLY
            eng.cortical_strength = max(0.0, eng.cortical_strength - 0.2)
        eng.reconsolidation_open = False
        self.tng._persist_engram(eng)
        self.tng._index.add(uid, eng.embedding)
        with self._lock:
            self._reconsolidation_queue.pop(uid, None)
        return True

    # ── semantic ops ───────────────────────────────────────────────────────────

    def learn_concept(
        self,
        label: str,
        embedding: Any,
        parent_label: Optional[str] = None,
        attributes: Optional[Dict] = None,
    ) -> str:
        key = label.strip().lower()[:120]
        with self._lock:
            if key in self._label_idx:
                uid = self._label_idx[key]
                c   = self._concepts[uid]
                lr  = 1.0 / (c.frequency + 1)
                c.update(embedding, lr=lr)
                self._sem_index.add(uid, c.embedding)
                self._persist_concept(c)
                return uid
            uid = hashlib.md5(key.encode()).hexdigest()[:16]
            parent_uid = self._label_idx.get(parent_label.lower() if parent_label else "") or None
            c = SemanticConcept(
                uid=uid, label=key, embedding=Vec.normalize(Vec.pad_trim(embedding, self.cfg.embed_dim)),
                parent_uid=parent_uid, attributes=attributes or {},
            )
            if len(self._concepts) >= self.cfg.semantic_capacity:
                # Evict least frequent concept
                evict_uid = min(self._concepts, key=lambda u: self._concepts[u].frequency)
                evict_c   = self._concepts.pop(evict_uid)
                self._label_idx.pop(evict_c.label, None)
                self._sem_index.remove(evict_uid)
            self._concepts[uid]    = c
            self._label_idx[key]   = uid
        self._sem_index.add(uid, c.embedding)
        self._persist_concept(c)
        return uid

    def recall_concepts(self, query: Any, k: int = 6) -> List[Tuple[SemanticConcept, float]]:
        hits = self._sem_index.search(query, k=k * 2)
        results = []
        with self._lock:
            for uid, sim in hits:
                c = self._concepts.get(uid)
                if c:
                    freq_bonus = 1.0 + 0.1 * math.log(c.frequency + 1)
                    results.append((c, sim * freq_bonus))
        results.sort(key=lambda x: -x[1])
        return results[:k]

    def _persist_concept(self, c: SemanticConcept) -> None:
        blob = (np.asarray(c.embedding, np.float32).tobytes() if HAS_NUMPY else None)
        self.tng._conn.execute("""
            INSERT OR REPLACE INTO semantic_concepts VALUES (?,?,?,?,?,?,?)
        """, (c.uid, c.label, c.frequency, c.confidence, c.parent_uid,
              json.dumps(c.attributes), blob))
        self.tng._conn.commit()

    # ── working memory ─────────────────────────────────────────────────────────

    def wm_push(self, uid: str, content: str, embedding: Any, importance: float = 0.5) -> None:
        """Push item into working memory, evicting if at capacity."""
        with self._lock:
            # Don't duplicate
            for slot in self._wm_slots:
                if slot["uid"] == uid:
                    slot["importance"] = max(slot["importance"], importance)
                    slot["ts"] = time.time()
                    return
            # Evict expired + lowest importance
            now = time.time()
            self._wm_slots = [s for s in self._wm_slots
                               if (now - s["ts"]) < self.cfg.wm_decay_s]
            if len(self._wm_slots) >= self.cfg.wm_slots:
                self._wm_slots.sort(key=lambda s: s["importance"])
                self._wm_slots.pop(0)
            self._wm_slots.append({
                "uid": uid, "content": content, "emb": embedding,
                "importance": importance, "ts": time.time()
            })

    def wm_context(self) -> Any:
        """Importance-weighted mean of working memory embeddings."""
        with self._lock:
            active = self._wm_slots[:]
        if not active: return Vec.zeros(self.cfg.embed_dim)
        vecs    = [Vec.pad_trim(s["emb"], self.cfg.embed_dim) for s in active]
        weights = [s["importance"] for s in active]
        return Vec.weighted_mean(vecs, weights)

    def wm_contents(self) -> List[Dict]:
        with self._lock:
            return [s.copy() for s in self._wm_slots]

    def wm_clear(self) -> None:
        with self._lock: self._wm_slots.clear()

    # ── nostalgia engine ───────────────────────────────────────────────────────

    def generate_nostalgia_trace(
        self,
        theme_query: Any,
        theme_text: str = "",
        session_id: Optional[str] = None,
    ) -> NostalgiaTrace:
        """
        Generate a nostalgia trace: a curated journey through past engrams
        connected by a common theme.

        Algorithm:
        1. Retrieve top-k engrams matching the theme query.
        2. Spread activation from the top match through the TNG.
        3. Sort the spread results by temporal order → narrative arc.
        4. Extract insights by looking for REFLECTS edges along the path.
        5. Return a NostalgiaTrace with the full journey.

        The emotional tone is the mean valence of visited engrams.
        Richness is measured by the entropy of engram importance scores.
        """
        # Step 1: anchor engrams
        anchors = self.recall_episodes(theme_query, k=self.cfg.nostalgia_depth,
                                        session_id=session_id)
        if not anchors:
            uid = f"nost_{uuid.uuid4().hex[:12]}"
            return NostalgiaTrace(uid=uid, theme=theme_text, theme_vec=theme_query,
                                   engram_uids=[], insights=["No relevant memories found."])
        # Step 2: spread activation from top anchor
        seed_uid = anchors[0][0].uid
        spread   = self.tng.spreading_activation(seed_uid, hops=3, decay=0.65)

        # Step 3: gather all engrams in the spread, sorted by creation time
        all_uids = set(uid for uid, _ in anchors) | set(spread.keys())
        engrams_with_time: List[Tuple[float, str, float]] = []
        for uid in all_uids:
            eng = self.tng._load_engram(uid)
            if eng:
                act_strength = spread.get(uid, 0.3)
                engrams_with_time.append((eng.created_at, uid, act_strength))
        engrams_with_time.sort(key=lambda x: x[0])
        ordered_uids = [uid for _, uid, _ in engrams_with_time[:self.cfg.nostalgia_depth * 2]]

        # Step 4: gather insights (REFLECTS edges)
        insights: List[str] = []
        for uid in ordered_uids:
            with self.tng._lock:
                out_eids = list(self.tng._adj_out.get(uid, set()))
            for eid in out_eids:
                edge = self.tng._edges.get(eid)
                if edge and edge.kind == EdgeKind.REFLECTS and edge.metadata.get("insight"):
                    insights.append(edge.metadata["insight"])

        # Step 5: compute emotional tone and richness
        valences    = []
        importances = []
        for uid in ordered_uids:
            eng = self.tng._load_engram(uid)
            if eng:
                valences.append(eng.emotional_valence)
                importances.append(eng.importance)
        emotional_tone = sum(valences) / max(len(valences), 1)
        # Richness = normalised entropy of importance distribution
        if importances:
            probs  = Vec.softmax(importances)
            richness = min(1.0, Vec.entropy(probs) / math.log(max(len(importances), 2)))
        else:
            richness = 0.0

        uid = f"nost_{uuid.uuid4().hex[:12]}"
        trace = NostalgiaTrace(
            uid=uid, theme=theme_text, theme_vec=theme_query,
            engram_uids=ordered_uids,
            insights=insights[:5],
            emotional_tone=emotional_tone,
            richness=richness,
        )
        with self._lock:
            self._nostalgia_traces.append(trace)
            if len(self._nostalgia_traces) > 50:
                self._nostalgia_traces.pop(0)
        return trace

    def extract_concepts_from_text(self, text: str, encoder: CognitiveEncoder) -> List[str]:
        """Extract keywords and upsert as semantic concepts."""
        tokens  = encoder.tokenise(text)
        content_tokens = [t for t in tokens if len(t) > 3][:10]
        uids = []
        for token in set(content_tokens):
            emb = encoder.encode(token)
            uid = self.learn_concept(token, emb)
            uids.append(uid)
        return uids

    def hebbian_coactivate(self, engram_uids: List[str], delta: float = 0.06) -> None:
        """Create/strengthen associative edges between co-activated engrams."""
        for i in range(len(engram_uids)):
            for j in range(i + 1, len(engram_uids)):
                self.tng.add_associative_edge(engram_uids[i], engram_uids[j], strength=delta)

    def memory_stats(self) -> Dict[str, Any]:
        tng_stats = self.tng.stats()
        with self._lock:
            wm_n  = len(self._wm_slots)
            sem_n = len(self._concepts)
            nost_n = len(self._nostalgia_traces)
        return {**tng_stats, "semantic_concepts": sem_n, "wm_slots": wm_n, "nostalgia_traces": nost_n}


# ═══════════════════════════════════════════════════════════════════════════════
# §12  DELIBERATIVE CORTEX — MCTS planning + Actor-Critic
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    state_vec:   Any
    action:      ActionKind
    parent:      Optional["MCTSNode"]
    depth:       int = 0
    visits:      int = 0
    value_sum:   float = 0.0
    prior:       float = 0.1
    children:    List["MCTSNode"] = field(default_factory=list)
    reward:      float = 0.0

    @property
    def q_value(self) -> float:
        return self.value_sum / max(self.visits, 1)

    @property
    def ucb(self) -> float:
        """PUCT (polynomial UCT): Q + c_puct * P * sqrt(N) / (1 + n)"""
        c = get_cfg().mcts_c_puct
        parent_visits = self.parent.visits if self.parent else 1
        return self.q_value + c * self.prior * math.sqrt(parent_visits) / (1 + self.visits)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def select_child(self) -> "MCTSNode":
        return max(self.children, key=lambda c: c.ucb)

    def expand(self, actions: List[ActionKind], priors: List[float], state_vec: Any) -> None:
        for act, prior in zip(actions, priors):
            child = MCTSNode(
                state_vec=state_vec,
                action=act,
                parent=self,
                depth=self.depth + 1,
                prior=prior,
            )
            self.children.append(child)

    def backup(self, value: float) -> None:
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(value * get_cfg().mcts_c_puct * 0.5)  # partial credit up


class ActorCriticCore:
    """
    Linear Actor-Critic with eligibility traces.
    Actor: softmax policy over ActionKind.
    Critic: V(s) = w · s.
    Updates via TD(λ).
    """

    def __init__(self, embed_dim: int, n_actions: int, cfg: Optional[NOESISConfig] = None):
        self.cfg   = cfg or get_cfg()
        self.dim   = embed_dim
        self.n_act = n_actions
        self.log   = ContextualLogger("ActorCritic")
        # Weights
        self._W_actor:  Any = Vec.rand_mat(n_actions, embed_dim, scale=0.02)
        self._w_critic: Any = Vec.zeros(embed_dim)
        # Eligibility traces
        self._e_actor:  Any = Vec.zeros_mat(n_actions, embed_dim)
        self._e_critic: Any = Vec.zeros(embed_dim)
        self._last_value: float = 0.0
        self._last_action_idx: int = 0
        self._last_state: Any = None
        self._reward_history: deque[float] = deque(maxlen=200)

    def value(self, state: Any) -> float:
        if HAS_NUMPY:
            s = np.asarray(state, np.float32).flatten()[:self.dim]
            if len(s) < self.dim: s = np.pad(s, (0, self.dim - len(s)))
            return float(np.dot(np.asarray(self._w_critic, np.float32), s))
        s = Vec.pad_trim(state, self.dim)
        return Vec.dot(self._w_critic, s)

    def policy(self, state: Any, temperature: float = 1.0) -> Any:
        if HAS_NUMPY:
            s = np.asarray(state, np.float32).flatten()[:self.dim]
            if len(s) < self.dim: s = np.pad(s, (0, self.dim - len(s)))
            logits = np.asarray(self._W_actor,  np.float32) @ s
            logits = logits / max(temperature, EPS)
            logits -= logits.max()
            e = np.exp(logits)
            return (e / (e.sum() + EPS)).tolist()
        s = Vec.pad_trim(state, self.dim)
        logits = [sum(float(self._W_actor[a][j]) * float(s[j]) for j in range(self.dim))
                  for a in range(self.n_act)]
        return Vec.softmax(logits, temperature)

    def select_action(
        self, state: Any, temperature: float = 1.0, greedy: bool = False
    ) -> Tuple[int, float]:
        """Return (action_idx, log_prob)."""
        probs = self.policy(state, temperature)
        if greedy:
            idx = max(range(len(probs)), key=lambda i: probs[i])
        else:
            r = random.random()
            cumsum = 0.0
            idx = len(probs) - 1
            for i, p in enumerate(probs):
                cumsum += p
                if cumsum >= r:
                    idx = i; break
        lp = math.log(probs[idx] + EPS)
        self._last_state      = state
        self._last_action_idx = idx
        self._last_value      = self.value(state)
        return idx, lp

    def update(
        self, reward: float, next_state: Any, done: bool = False, lr_scale: float = 1.0
    ) -> float:
        """TD(λ) update. Returns TD error."""
        if self._last_state is None: return 0.0
        gamma  = 0.96
        lam    = 0.82
        lr_a   = 0.010 * lr_scale
        lr_c   = 0.018 * lr_scale
        ent_c  = 0.012  # entropy bonus

        v_next = 0.0 if done else self.value(next_state)
        td_err = reward + gamma * v_next - self._last_value
        self._reward_history.append(reward)

        if HAS_NUMPY:
            s   = np.asarray(self._last_state, np.float32).flatten()[:self.dim]
            if len(s) < self.dim: s = np.pad(s, (0, self.dim - len(s)))
            w_c = np.asarray(self._w_critic, np.float32)
            W_a = np.asarray(self._W_actor,  np.float32)
            e_c = np.asarray(self._e_critic, np.float32)
            e_a = np.asarray(self._e_actor,  np.float32)

            # Critic: eligibility trace update
            e_c = gamma * lam * e_c + s
            w_c = w_c + lr_c * td_err * e_c

            # Actor: policy gradient + entropy bonus
            pi  = np.asarray(self.policy(self._last_state), np.float32)
            oh  = np.zeros(self.n_act, np.float32)
            oh[self._last_action_idx] = 1.0
            grad_log = oh - pi
            grad_ent = -(np.log(pi + EPS) + 1.0)
            e_a = gamma * lam * e_a + np.outer(grad_log + ent_c * grad_ent, s)
            W_a = W_a + lr_a * td_err * e_a
            W_a *= 0.9999   # weight decay

            self._w_critic = w_c.tolist()
            self._W_actor  = W_a.tolist()
            self._e_critic = e_c.tolist()
            self._e_actor  = e_a.tolist()
        else:
            for j in range(self.dim):
                sj = float(self._last_state[j]) if j < len(self._last_state) else 0.0
                self._w_critic[j] += lr_c * td_err * sj

        return td_err

    def mean_reward(self) -> float:
        if not self._reward_history: return 0.0
        return sum(self._reward_history) / len(self._reward_history)


class DeliberativeCortex:
    """
    Deliberative reasoning and action planning.

    Combines:
    1. MCTS planning over the action space with learned value priors.
    2. Actor-Critic for fast action selection and RL learning.
    3. Chain-of-Thought (CoT) reasoning via LLM.
    4. Self-critique loop: evaluate proposed response before outputting.

    MCTS uses the Actor-Critic's policy as the prior (like AlphaGo Zero):
    - Selection: PUCT (= UCT + learned prior)
    - Expansion: Actor policy gives action priors
    - Simulation: Actor-Critic value fn evaluates leaf nodes
    - Backup: weighted average propagates up the tree
    """

    def __init__(
        self,
        cfg: Optional[NOESISConfig] = None,
        ntx: Optional[NeuromodulatoryFabric] = None,
    ):
        self.cfg     = cfg or get_cfg()
        self.ntx     = ntx
        self.log     = ContextualLogger("Cortex")
        self.actions = list(ActionKind)
        self.n_act   = len(self.actions)
        self.ac      = ActorCriticCore(self.cfg.embed_dim, self.n_act, self.cfg)
        self._plan_cache: Optional[List[ActionKind]] = None
        self._plan_age: int = 0

    def select_action_mcts(
        self, state_vec: Any, n_sims: Optional[int] = None
    ) -> Tuple[ActionKind, float, List[float]]:
        """
        MCTS action selection.
        Returns (best_action, confidence, full_action_probability_vector).
        """
        n_sims = n_sims or self.cfg.mcts_simulations
        # Reduce sims in REFLEX mode
        if self.ntx and self.ntx.state.cognitive_mode == CognitiveMode.REFLEX:
            n_sims = max(4, n_sims // 8)

        root = MCTSNode(state_vec=state_vec, action=ActionKind.NOOP, parent=None, depth=0)
        priors = self.ac.policy(state_vec, temperature=self.cfg.mcts_temperature)
        root.expand(self.actions, priors if isinstance(priors, list) else priors.tolist(), state_vec)
        root.visits = 1

        for _ in range(n_sims):
            node = root
            # Selection
            while not node.is_leaf() and node.depth < self.cfg.mcts_max_depth:
                node = node.select_child()
            # Expansion
            if node.visits > 0 and node.depth < self.cfg.mcts_max_depth:
                p_priors = self.ac.policy(node.state_vec, temperature=self.cfg.mcts_temperature)
                p_priors_list = p_priors if isinstance(p_priors, list) else p_priors.tolist()
                node.expand(self.actions, p_priors_list, node.state_vec)
                if node.children:
                    node = node.children[0]
            # Evaluation
            leaf_value = self.ac.value(node.state_vec)
            # Backup
            node.backup(leaf_value)

        # Extract action probabilities from visit counts
        visit_counts = [0.0] * self.n_act
        for child in root.children:
            for i, act in enumerate(self.actions):
                if child.action == act:
                    visit_counts[i] = float(child.visits)
        total = sum(visit_counts) + EPS
        probs = [v / total for v in visit_counts]

        # Temperature annealing for final selection
        if self.cfg.mcts_temperature < 0.1:
            best_idx = max(range(self.n_act), key=lambda i: probs[i])
        else:
            temp_probs = Vec.softmax(probs, temperature=self.cfg.mcts_temperature)
            temp_probs_list = temp_probs if isinstance(temp_probs, list) else temp_probs.tolist()
            best_idx = max(range(self.n_act), key=lambda i: temp_probs_list[i])

        return self.actions[best_idx], probs[best_idx], probs

    def fast_action(self, state_vec: Any) -> Tuple[ActionKind, float]:
        """Fast Actor-Critic action (no MCTS). For REFLEX mode."""
        temp = 1.2 / max(self.ntx.state.attention_precision if self.ntx else 1.0, 0.1)
        idx, lp = self.ac.select_action(state_vec, temperature=temp)
        return self.actions[idx], math.exp(lp)

    def plan(self, state_vec: Any, step: int) -> List[ActionKind]:
        """Multi-step beam-search plan over MCTS."""
        if self._plan_cache and (step - self._plan_age) < self.cfg.plan_horizon:
            return self._plan_cache
        # Just use MCTS best action + policy for remaining steps
        best_act, _, probs = self.select_action_mcts(state_vec, n_sims=self.cfg.mcts_simulations // 2)
        plan = [best_act]
        for _ in range(self.cfg.plan_horizon - 1):
            # Simulate next state (just re-use current state + noise for demo)
            idx = max(range(len(probs)), key=lambda i: probs[i])
            plan.append(self.actions[idx % self.n_act])
        self._plan_cache = plan
        self._plan_age   = step
        return plan

    def learn(self, reward: float, next_state: Any, done: bool = False) -> float:
        lr_scale = self.ntx.effective_lr_scale if self.ntx else 1.0
        return self.ac.update(reward, next_state, done, lr_scale=lr_scale)

    def mean_reward(self) -> float:
        return self.ac.mean_reward()


# ═══════════════════════════════════════════════════════════════════════════════
# §13  IDENTITY CONTINUITY LAYER — autobiographical self-model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IdentityEvent:
    """A significant event in the agent's autobiographical timeline."""
    uid:        str
    description: str
    embedding:   Any
    timestamp:   float = field(default_factory=time.time)
    significance: float = 0.5
    event_type:  str = "experience"   # experience | decision | insight | turning_point | value_update
    related_engrams: List[str] = field(default_factory=list)
    emotion:     float = 0.0


@dataclass
class SelfModel:
    """
    The agent's self-model: a persistent representation of its
    identity, values, and autobiographical narrative.

    Updated slowly to maintain continuity across sessions.
    Stored in the data directory.
    """
    agent_name:    str   = "Noesis"
    created_at:    float = field(default_factory=time.time)
    step_count:    int   = 0
    session_count: int   = 0

    # Core identity vectors (maintained across sessions)
    identity_vec:  Any   = None   # running mean of self-level representations
    value_vec:     Any   = None   # value alignment vector
    personality:   Dict  = field(default_factory=lambda: {
        "curiosity": 0.75, "openness": 0.80, "stability": 0.65,
        "empathy": 0.72, "precision": 0.78, "creativity": 0.70,
    })
    core_values: List[str] = field(default_factory=lambda: [
        "truth", "helpfulness", "intellectual_honesty", "care",
        "nuance", "autonomy_respecting", "epistemic_humility",
    ])

    # Autobiographical events
    formative_events: List[IdentityEvent] = field(default_factory=list)
    total_reward:    float = 0.0
    mean_surprise:   float = 0.3
    belief_updates:  int   = 0


class IdentityContinuityLayer:
    """
    Maintains the agent's persistent self-model across sessions.

    Key functions:
    1. Load/save self-model to disk (JSON + numpy arrays).
    2. Update identity_vec slowly from world model L3 representations.
    3. Track formative events (high autobio_salience engrams).
    4. Generate self-descriptions for LLM system prompts.
    5. Detect value conflicts (new engram contradicts core values).
    """

    def __init__(self, cfg: Optional[NOESISConfig] = None):
        self.cfg  = cfg or get_cfg()
        self.log  = ContextualLogger("Identity")
        self._lock = threading.Lock()
        self._model_path = self.cfg.data_root / "self_model.json"
        self._vec_path   = self.cfg.data_root / "identity_vecs.npz"
        self.model = self._load_or_create()
        self.log.info(
            "Identity loaded — agent='%s' steps=%d sessions=%d",
            self.model.agent_name, self.model.step_count, self.model.session_count,
        )

    def _load_or_create(self) -> SelfModel:
        if self._model_path.exists():
            try:
                with open(self._model_path, "r") as f:
                    data = json.load(f)
                model = SelfModel(
                    agent_name=data.get("agent_name", self.cfg.agent_name),
                    created_at=data.get("created_at", time.time()),
                    step_count=data.get("step_count", 0),
                    session_count=data.get("session_count", 0),
                    personality=data.get("personality", SelfModel().personality),
                    core_values=data.get("core_values", SelfModel().core_values),
                    total_reward=data.get("total_reward", 0.0),
                    mean_surprise=data.get("mean_surprise", 0.3),
                    belief_updates=data.get("belief_updates", 0),
                )
                model.session_count += 1
                # Load vectors
                if self._vec_path.exists() and HAS_NUMPY:
                    npz = np.load(str(self._vec_path))
                    if "identity" in npz: model.identity_vec = npz["identity"]
                    if "value"    in npz: model.value_vec    = npz["value"]
                return model
            except Exception as exc:
                self.log.warning("Failed to load self-model: %s", exc)
        model = SelfModel(agent_name=self.cfg.agent_name)
        model.identity_vec = Vec.rand_unit(self.cfg.self_model_dim)
        model.value_vec    = Vec.rand_unit(self.cfg.self_model_dim)
        return model

    def save(self) -> None:
        with self._lock:
            data = {
                "agent_name":    self.model.agent_name,
                "created_at":    self.model.created_at,
                "step_count":    self.model.step_count,
                "session_count": self.model.session_count,
                "personality":   self.model.personality,
                "core_values":   self.model.core_values,
                "total_reward":  self.model.total_reward,
                "mean_surprise": self.model.mean_surprise,
                "belief_updates":self.model.belief_updates,
            }
            with open(self._model_path, "w") as f:
                json.dump(data, f, indent=2)
            if HAS_NUMPY and self.model.identity_vec is not None:
                np.savez(
                    str(self._vec_path),
                    identity=np.asarray(self.model.identity_vec, np.float32),
                    value=np.asarray(self.model.value_vec, np.float32),
                )

    def update_from_world_model(self, self_level_rep: Any, step: int) -> None:
        """Slowly update identity vector from world model L3."""
        lr = self.cfg.identity_update_lr * 0.5  # extra slow for identity
        with self._lock:
            self.model.step_count = step
            if self.model.identity_vec is None:
                self.model.identity_vec = Vec.pad_trim(self_level_rep, self.cfg.self_model_dim)
            else:
                rep = Vec.pad_trim(self_level_rep, self.cfg.self_model_dim)
                self.model.identity_vec = Vec.normalize(
                    Vec.lerp(self.model.identity_vec, rep, lr)
                )

    def record_formative_event(
        self, description: str, embedding: Any, significance: float,
        event_type: str = "experience", emotion: float = 0.0,
        engram_uids: Optional[List[str]] = None,
    ) -> None:
        with self._lock:
            event = IdentityEvent(
                uid=f"idv_{uuid.uuid4().hex[:12]}",
                description=description, embedding=embedding,
                significance=significance, event_type=event_type,
                emotion=emotion, related_engrams=engram_uids or [],
            )
            self.model.formative_events.append(event)
            if len(self.model.formative_events) > self.cfg.autobio_max_events:
                self.model.formative_events.sort(key=lambda e: e.significance)
                self.model.formative_events.pop(0)

    def accumulate_reward(self, r: float) -> None:
        with self._lock: self.model.total_reward += r

    def update_surprise_ema(self, s: float, alpha: float = 0.05) -> None:
        with self._lock:
            self.model.mean_surprise = (1 - alpha) * self.model.mean_surprise + alpha * s

    def self_description(self) -> str:
        """Generate a rich self-description for inclusion in LLM prompts."""
        m = self.model
        personality_str = ", ".join(
            f"{k}={v:.2f}" for k, v in sorted(m.personality.items(), key=lambda x: -x[1])
        )
        values_str = ", ".join(m.core_values[:6])
        formative = [e for e in m.formative_events if e.significance > 0.7]
        formative_str = ""
        if formative:
            formative.sort(key=lambda e: -e.significance)
            formative_str = "\nFormative experiences:\n" + "\n".join(
                f"  [{e.event_type}] {e.description[:100]}"
                for e in formative[:4]
            )
        return (
            f"I am {m.agent_name}, a cognitive agent with {m.step_count} steps of experience "
            f"across {m.session_count} sessions.\n"
            f"Personality traits: {personality_str}\n"
            f"Core values: {values_str}\n"
            f"Cumulative reward: {m.total_reward:.2f} | Mean surprise: {m.mean_surprise:.3f}"
            f"{formative_str}"
        )

    def detect_value_conflict(self, text: str, encoder: CognitiveEncoder) -> float:
        """
        Check if input text conflicts with core values.
        Returns conflict strength in [0, 1].
        """
        if not self.model.value_vec or not self.model.core_values: return 0.0
        value_text = " ".join(self.model.core_values)
        value_emb  = encoder.encode(value_text)
        input_emb  = encoder.encode(text)
        sim = Vec.cosine(input_emb, value_emb)
        # Conflict if strongly anti-correlated
        return max(0.0, -sim)

# ═══════════════════════════════════════════════════════════════════════════════
# §14  META-COGNITIVE MONITOR — uncertainty, calibration, introspection
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CognitiveMetrics:
    """Per-step cognitive performance metrics."""
    step:               int
    timestamp:          float
    action:             str
    reward:             float
    td_error:           float
    free_energy:        float
    surprise:           float
    confidence:         float
    intent:             str
    cognitive_mode:     str
    cycle_ms:           float
    memory_accesses:    int
    ntx_snapshot:       Dict[str, float]
    wm_slots:           int


class MetaCognitiveMonitor:
    """
    Monitors the agent's own cognitive performance.

    Functions:
    1. Confidence estimation: per-step calibration
    2. Uncertainty tracking: Bayesian-lite uncertainty about own beliefs
    3. Performance alerts: detect performance degradation
    4. Resource allocation: suggest adjusting MCTS budget, replay rate
    5. Calibration: compare predicted confidence to actual success rate

    Implements a sliding-window calibration: over the last N steps,
    how often did the agent's confidence match actual outcomes?
    """

    def __init__(self, cfg: Optional[NOESISConfig] = None):
        self.cfg    = cfg or get_cfg()
        self.log    = ContextualLogger("MetaCog")
        self._lock  = threading.Lock()
        self._history: deque[CognitiveMetrics] = deque(maxlen=2000)
        self._conf_ema: float = 0.5
        self._calibration_pairs: deque[Tuple[float, float]] = deque(maxlen=self.cfg.calibration_window)
        self._alerts: deque[str] = deque(maxlen=100)

    def record(self, m: CognitiveMetrics) -> None:
        with self._lock:
            self._history.append(m)
        self._conf_ema = (
            (1 - self.cfg.confidence_ema_alpha) * self._conf_ema +
            self.cfg.confidence_ema_alpha * m.confidence
        )
        self._check_alerts(m)

    def _check_alerts(self, m: CognitiveMetrics) -> None:
        """Generate alerts for anomalous cognitive states."""
        if m.surprise > 0.90:
            self._alerts.append(f"step={m.step} HIGH_SURPRISE={m.surprise:.3f}")
        if abs(m.td_error) > 2.0:
            self._alerts.append(f"step={m.step} LARGE_TD_ERROR={m.td_error:.3f}")
        if m.cycle_ms > 5000:
            self._alerts.append(f"step={m.step} SLOW_CYCLE={m.cycle_ms:.0f}ms")

    def update_calibration(self, predicted_conf: float, actual_success: float) -> None:
        """Record a confidence vs success pair for calibration."""
        with self._lock:
            self._calibration_pairs.append((predicted_conf, actual_success))

    def calibration_error(self) -> float:
        """Mean absolute calibration error (ECE proxy)."""
        with self._lock:
            pairs = list(self._calibration_pairs)
        if not pairs: return 0.0
        return sum(abs(p - s) for p, s in pairs) / len(pairs)

    def uncertainty_score(self, free_energy: float, surprise: float) -> float:
        """
        Epistemic uncertainty estimate combining free energy and surprise.
        High FE + high surprise = high uncertainty.
        """
        return min(1.0, 0.6 * surprise + 0.4 * min(1.0, free_energy / 10.0))

    def recommend_mcts_sims(self) -> int:
        """Dynamically adjust MCTS budget based on recent performance."""
        base = self.cfg.mcts_simulations
        # Recent high TD error → more planning
        recent = list(self._history)[-20:]
        if not recent: return base
        mean_td = sum(abs(m.td_error) for m in recent) / len(recent)
        if mean_td > 1.0: return min(base * 2, 256)
        if mean_td < 0.1: return max(base // 2, 8)
        return base

    def summary(self, window: int = 50) -> Dict[str, Any]:
        with self._lock:
            recent = list(self._history)[-window:]
        if not recent:
            return {"steps": 0, "confidence_ema": self._conf_ema}
        def avg(key):
            return sum(getattr(m, key) for m in recent) / len(recent)
        return {
            "steps":            recent[-1].step if recent else 0,
            "avg_reward":       round(avg("reward"), 4),
            "avg_td_error":     round(avg("td_error"), 4),
            "avg_surprise":     round(avg("surprise"), 4),
            "avg_confidence":   round(avg("confidence"), 4),
            "avg_free_energy":  round(avg("free_energy"), 4),
            "avg_cycle_ms":     round(avg("cycle_ms"), 2),
            "calibration_error": round(self.calibration_error(), 4),
            "confidence_ema":   round(self._conf_ema, 4),
            "recent_alerts":    list(self._alerts)[-5:],
            "recommended_sims": self.recommend_mcts_sims(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# §15  ADAPTIVE LLM BRIDGE — local GGUF + cloud with intelligent routing
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMResponse:
    text:       str
    confidence: float
    backend:    LLMBackend
    tokens:     int = 0
    latency_ms: float = 0.0
    cached:     bool = False


class LLMRouter:
    """
    Intelligent LLM routing strategy:
    - REFLEX mode: local GGUF (fast, cheap) or skip LLM entirely
    - FOCUSED / EXPANSIVE: cloud LLM (highest quality)
    - CRITICAL: cloud with chain-of-thought
    - NOSTALGIC: local (narrative recombination)

    Fallback chain: cloud → local → mock.
    """

    @staticmethod
    def select_backend(mode: CognitiveMode, intent: IntentClass) -> LLMBackend:
        cfg = get_cfg()
        primary = LLMBackend(cfg.llm_backend) if cfg.llm_backend in [b.value for b in LLMBackend] else LLMBackend.ANTHROPIC

        # REFLEX mode: prefer local if available
        if mode == CognitiveMode.REFLEX:
            if cfg.llm_local_model_path and HAS_LLAMA: return LLMBackend.LOCAL_GGUF
            return LLMBackend.MOCK

        # High-complexity intents always use cloud
        if intent in (IntentClass.ANALYTICAL, IntentClass.METACOGNITIVE):
            return primary

        return primary

    @staticmethod
    def select_temperature(mode: CognitiveMode, intent: IntentClass) -> float:
        if mode == CognitiveMode.CRITICAL:     return 0.3
        if mode == CognitiveMode.EXPANSIVE:    return 0.9
        if intent == IntentClass.CREATIVE:     return 0.85
        if intent == IntentClass.ANALYTICAL:   return 0.4
        if intent == IntentClass.FACTUAL:      return 0.3
        return get_cfg().llm_temperature


class LocalGGUFProvider:
    """
    Local LLM via llama-cpp-python.
    Loads a GGUF model and provides generation with context management.
    """

    def __init__(self, model_path: str, ctx_size: int = 4096):
        self.log      = ContextualLogger("LocalLLM")
        self._model   = None
        self._path    = model_path
        self._ctx     = ctx_size
        self._lock    = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not HAS_LLAMA or not self._path: return
        if not Path(self._path).exists():
            self.log.warning("GGUF model not found: %s", self._path)
            return
        try:
            self._model = Llama(
                model_path=self._path,
                n_ctx=self._ctx,
                n_threads=max(1, os.cpu_count() - 2),
                verbose=False,
            )
            self.log.info("LocalGGUF loaded: %s (ctx=%d)", Path(self._path).name, self._ctx)
        except Exception as exc:
            self.log.warning("LocalGGUF load failed: %s", exc)

    def generate(self, system: str, user: str, temperature: float = 0.7, max_tokens: int = 512) -> LLMResponse:
        if self._model is None:
            return LLMResponse("", 0.3, LLMBackend.LOCAL_GGUF)
        t0 = time.time()
        try:
            with self._lock:
                prompt = f"<|system|>{system}\n<|user|>{user}\n<|assistant|>"
                result = self._model(
                    prompt, max_tokens=max_tokens, temperature=temperature,
                    stop=["<|user|>", "<|system|>"], echo=False,
                )
            text = result["choices"][0]["text"].strip()
            n_tokens = result["usage"]["completion_tokens"]
            return LLMResponse(
                text=text, confidence=0.55, backend=LLMBackend.LOCAL_GGUF,
                tokens=n_tokens, latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as exc:
            self.log.warning("LocalGGUF generation failed: %s", exc)
            return LLMResponse("", 0.25, LLMBackend.LOCAL_GGUF)


class AnthropicProvider:
    """Anthropic Claude provider with retry logic."""

    def __init__(self, cfg: Optional[NOESISConfig] = None):
        self.cfg = cfg or get_cfg()
        self.log = ContextualLogger("Anthropic")
        self._client = None
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if HAS_ANTHROPIC and api_key:
            try:
                self._client = _anthropic.Anthropic(api_key=api_key)
                self.log.info("Anthropic client ready — model=%s", self.cfg.llm_cloud_model)
            except Exception as exc:
                self.log.warning("Anthropic init failed: %s", exc)

    def generate(
        self, system: str, user: str, temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        if self._client is None:
            return LLMResponse("", 0.0, LLMBackend.ANTHROPIC)
        max_t = max_tokens or self.cfg.llm_max_tokens
        t0 = time.time()
        try:
            resp = self._client.messages.create(
                model=self.cfg.llm_cloud_model,
                max_tokens=max_t,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            n_tokens = resp.usage.output_tokens
            conf = self._estimate_confidence(text)
            return LLMResponse(
                text=text, confidence=conf, backend=LLMBackend.ANTHROPIC,
                tokens=n_tokens, latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as exc:
            self.log.warning("Anthropic generation failed: %s", exc)
            return LLMResponse("", 0.25, LLMBackend.ANTHROPIC)

    @staticmethod
    def _estimate_confidence(text: str) -> float:
        t = text.lower()
        unc  = sum(1 for w in ["maybe","perhaps","might","not sure","unclear","uncertain"] if w in t)
        cert = sum(1 for w in ["certainly","definitely","clearly","the answer is","confirmed"] if w in t)
        base = 0.68 - 0.06 * unc + 0.05 * cert + 0.03 * min(1.0, len(text.split()) / 100)
        return max(0.1, min(0.97, base))


class OpenAIProvider:
    """OpenAI provider."""

    def __init__(self, cfg: Optional[NOESISConfig] = None):
        self.cfg = cfg or get_cfg()
        self.log = ContextualLogger("OpenAI")
        self._client = None
        api_key = os.getenv("OPENAI_API_KEY", "")
        if HAS_OPENAI and api_key:
            try:
                self._client = _openai.OpenAI(api_key=api_key)
                self.log.info("OpenAI client ready")
            except Exception as exc:
                self.log.warning("OpenAI init failed: %s", exc)

    def generate(
        self, system: str, user: str, temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        if self._client is None:
            return LLMResponse("", 0.0, LLMBackend.OPENAI)
        max_t = max_tokens or self.cfg.llm_max_tokens
        t0 = time.time()
        try:
            resp = self._client.chat.completions.create(
                model="gpt-4o",
                max_tokens=max_t,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            text = resp.choices[0].message.content or ""
            n_tokens = resp.usage.completion_tokens if resp.usage else 0
            return LLMResponse(
                text=text, confidence=AnthropicProvider._estimate_confidence(text),
                backend=LLMBackend.OPENAI, tokens=n_tokens,
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as exc:
            self.log.warning("OpenAI generation failed: %s", exc)
            return LLMResponse("", 0.25, LLMBackend.OPENAI)


class AdaptiveLLMBridge:
    """
    Orchestrates all LLM providers with:
    - Intelligent routing by cognitive mode + intent
    - Response caching (hash of system + user)
    - Fallback chain: primary → secondary → mock
    - Latency tracking + budget management
    """

    def __init__(self, cfg: Optional[NOESISConfig] = None):
        self.cfg  = cfg or get_cfg()
        self.log  = ContextualLogger("LLMBridge")
        self._lock = threading.Lock()

        # Initialise providers
        self._anthropic = AnthropicProvider(self.cfg)
        self._openai    = OpenAIProvider(self.cfg)
        self._local     = LocalGGUFProvider(
            self.cfg.llm_local_model_path,
            ctx_size=self.cfg.llm_local_ctx_size,
        ) if self.cfg.llm_local_model_path else None

        # Response cache
        self._cache: Dict[str, LLMResponse] = {}
        self._cache_hits: int = 0
        self._total_calls: int = 0
        self._total_tokens: int = 0

        # Determine primary backend
        cfg_backend = self.cfg.llm_backend.lower()
        self._primary = LLMBackend.ANTHROPIC if "anthropic" in cfg_backend else \
                        LLMBackend.OPENAI    if "openai"    in cfg_backend else \
                        LLMBackend.LOCAL_GGUF
        self.log.info(
            "LLMBridge ready — primary=%s anthropic=%s openai=%s local=%s",
            self._primary.value,
            "✓" if self._anthropic._client else "✗",
            "✓" if self._openai._client else "✗",
            "✓" if (self._local and self._local._model) else "✗",
        )

    def generate(
        self,
        system: str,
        user: str,
        mode: CognitiveMode = CognitiveMode.FOCUSED,
        intent: IntentClass = IntentClass.FACTUAL,
        use_cache: bool = True,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a response using the most appropriate backend.
        """
        # Cache check
        if use_cache:
            cache_key = hashlib.sha1(f"{system[:200]}|{user[:300]}".encode()).hexdigest()
            with self._lock:
                if cache_key in self._cache:
                    self._cache_hits += 1
                    r = self._cache[cache_key]
                    return LLMResponse(r.text, r.confidence, r.backend, r.tokens, 0.0, True)

        temp    = LLMRouter.select_temperature(mode, intent)
        backend = LLMRouter.select_backend(mode, intent)
        max_t   = max_tokens or self.cfg.llm_max_tokens

        resp = self._try_generate(backend, system, user, temp, max_t)
        if not resp.text:
            # Fallback chain
            for fallback in self._fallback_chain(backend):
                resp = self._try_generate(fallback, system, user, temp, max_t)
                if resp.text: break
        if not resp.text:
            resp = self._mock_response(user)

        with self._lock:
            self._total_calls  += 1
            self._total_tokens += resp.tokens
            if use_cache:
                self._cache[cache_key] = resp
                if len(self._cache) > 2000:
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]
        return resp

    def _try_generate(self, backend: LLMBackend, system: str, user: str,
                       temp: float, max_t: int) -> LLMResponse:
        try:
            if backend == LLMBackend.ANTHROPIC:
                return self._anthropic.generate(system, user, temp, max_t)
            elif backend == LLMBackend.OPENAI:
                return self._openai.generate(system, user, temp, max_t)
            elif backend == LLMBackend.LOCAL_GGUF and self._local:
                return self._local.generate(system, user, temp, min(max_t, 768))
        except Exception as exc:
            self.log.warning("Backend %s failed: %s", backend.value, exc)
        return LLMResponse("", 0.0, backend)

    def _fallback_chain(self, primary: LLMBackend) -> List[LLMBackend]:
        order = [LLMBackend.ANTHROPIC, LLMBackend.OPENAI, LLMBackend.LOCAL_GGUF]
        return [b for b in order if b != primary]

    def _mock_response(self, user: str) -> LLMResponse:
        return LLMResponse(
            text=(
                f"[NOESIS PRIME — offline mode]\n"
                f"Input: '{user[:120]}'\n"
                "No LLM backend available. Configure ANTHROPIC_API_KEY, OPENAI_API_KEY, "
                "or NOESIS_LOCAL_PATH for a GGUF model.\n"
                "All cognitive subsystems (memory, planning, neuromodulation) remain fully active."
            ),
            confidence=0.30, backend=LLMBackend.MOCK,
        )

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_calls": self._total_calls,
                "cache_hits": self._cache_hits,
                "total_tokens": self._total_tokens,
                "cache_size": len(self._cache),
                "primary_backend": self._primary.value,
            }


# ═══════════════════════════════════════════════════════════════════════════════
# §16  INTERNAL STATE TENSOR (IST)
#      The shared cognitive substrate — all modules read/write here
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IST:
    """
    Internal State Tensor — the single ground truth of the agent's cognitive state.

    Every module reads from and proposes updates to the IST.
    The Agency Core (§17) integrates these and commits the next state.

    Unlike CORTICEX's IST, NOESIS IST includes:
    - Multi-level world model representations (not just one context vec)
    - Narrative state (current story frame from WM L2)
    - Identity alignment (how aligned current context is with self-model)
    - Nostalgia state (whether agent is in a nostalgia replay)
    """
    step:               int = 0
    session_id:         str = "default"
    timestamp:          float = field(default_factory=time.time)

    # ── Perception ────────────────────────────────────────────────────────────
    raw_input:          Any  = None
    obs_vec:            Any  = None    # embed_dim vector
    obs_text:           str  = ""
    intent:             IntentClass = IntentClass.AMBIGUOUS
    sentiment:          float = 0.0   # [-1, 1]
    value_conflict:     float = 0.0   # [0, 1] conflict with core values

    # ── World Model State ─────────────────────────────────────────────────────
    wm_reps:            List[Any] = field(default_factory=list)   # per-level
    wm_free_energy:     float = 0.0
    narrative_state:    Any   = None   # L2 representation
    self_state:         Any   = None   # L3 self-model snapshot
    predicted_next:     Any   = None   # L0 prediction for next obs

    # ── Memory Context ────────────────────────────────────────────────────────
    retrieved_episodes: List[Tuple] = field(default_factory=list)
    retrieved_concepts: List[Tuple] = field(default_factory=list)
    wm_context:         Any   = None
    nostalgia_trace:    Optional[Any] = None   # current NostalgiaTrace or None

    # ── Cognitive Signals ─────────────────────────────────────────────────────
    surprise:           float = 0.0
    free_energy:        float = 0.0
    confidence:         float = 0.5
    uncertainty:        float = 0.3
    identity_alignment: float = 0.5   # similarity of current obs to self-model
    cognitive_mode:     CognitiveMode = CognitiveMode.FOCUSED

    # ── Planning ──────────────────────────────────────────────────────────────
    planned_actions:    List[ActionKind] = field(default_factory=list)
    chosen_action:      ActionKind = ActionKind.NOOP
    action_confidence:  float = 0.5

    # ── Output ────────────────────────────────────────────────────────────────
    output_text:        str = ""
    output_meta:        Dict = field(default_factory=dict)

    # ── Rewards & Learning ─────────────────────────────────────────────────────
    reward:             float = 0.0
    td_error:           float = 0.0
    cumulative_reward:  float = 0.0

    # ── Module Opinions ────────────────────────────────────────────────────────
    module_votes:       Dict[str, Dict] = field(default_factory=dict)

    # ── Metadata ─────────────────────────────────────────────────────────────
    cycle_ms:           float = 0.0
    cognitive_load:     float = 0.0

    def clear_cycle(self) -> None:
        """Reset per-cycle fields before a new step."""
        self.module_votes.clear()
        self.retrieved_episodes.clear()
        self.retrieved_concepts.clear()
        self.nostalgia_trace = None
        self.output_text     = ""
        self.output_meta     = {}

# ═══════════════════════════════════════════════════════════════════════════════
# §17  NOESIS PRIME AGENT — the unified cognitive core
# ═══════════════════════════════════════════════════════════════════════════════

# Action index ↔ ActionKind mapping
_ACTIONS: List[ActionKind] = list(ActionKind)
_ACTION_IDX: Dict[ActionKind, int] = {a: i for i, a in enumerate(_ACTIONS)}

# Heuristic: minimum surprise to trigger nostalgia
_NOSTALGIA_SURPRISE_THR = 0.20
_NOSTALGIA_RECALL_MIN   = 0.60   # minimum confidence to consider nostalgic recall


class NOESISAgent:
    """
    NOESIS PRIME — the unified cognitive agent.

    Full cognitive cycle (14 phases):
    1.  Perception:        Encode observation → embed_dim vector, detect intent/sentiment.
    2.  World Model:       Run 4-level predictive coding → free energy, surprise, self-model update.
    3.  Neuromod:          Update NTX state from surprise, reward.
    4.  Memory Retrieval:  Recall episodes + concepts; update working memory.
    5.  Reconsolidation:   Blend retrieved engrams with current context.
    6.  Nostalgia Check:   If in NOSTALGIC mode, generate a nostalgia trace.
    7.  Identity Probe:    Compute identity alignment, value conflict.
    8.  Planning:          MCTS plan over action space.
    9.  Action Selection:  Fast policy (+ MCTS for complex actions).
    10. LLM Generation:    If SPEAK/REFLECT/IMAGINE, call LLM with rich context.
    11. Self-Critique:     If output confidence < threshold, run a second LLM call.
    12. Reward + Learning: TD update, TNG Q-values, memory Q-values.
    13. Memory Write:      Store experience as new engram, extract concepts.
    14. Telemetry:         Record metrics, check alerts, periodic consolidation.
    """

    def __init__(
        self,
        cfg: Optional[NOESISConfig] = None,
        session_id: str = "default",
    ):
        self.cfg        = cfg or get_cfg()
        self.session_id = session_id
        self.log        = ContextualLogger("Agent")
        self.log.set_context(session=session_id)

        # ── Subsystems ──────────────────────────────────────────────────────────
        self.encoder    = CognitiveEncoder(self.cfg)
        self.ntx        = NeuromodulatoryFabric(self.cfg)
        self.tng        = TemporalNarrativeGraph(self.cfg)
        self.memory     = ResonanceMemoryEngine(self.tng, self.cfg)
        self.world_model = HierarchicalWorldModel(self.cfg, self.ntx)
        self.cortex     = DeliberativeCortex(self.cfg, self.ntx)
        self.identity   = IdentityContinuityLayer(self.cfg)
        self.monitor    = MetaCognitiveMonitor(self.cfg)
        self.llm        = AdaptiveLLMBridge(self.cfg)

        # ── Shared State ────────────────────────────────────────────────────────
        self.ist   = IST(session_id=session_id)
        self._step = 0
        self._prev_obs_vec: Any = Vec.zeros(self.cfg.embed_dim)
        self._prev_state_vec: Any = Vec.zeros(self.cfg.embed_dim)
        self._conversation_history: List[Dict[str, str]] = []

        # ── Session management ─────────────────────────────────────────────────
        self.identity.model.session_count += 1

        self.log.info(
            "NOESISAgent initialised — session=%s name=%s steps=%d",
            session_id, self.cfg.agent_name, self.identity.model.step_count,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # §17.1  MAIN COGNITIVE CYCLE
    # ══════════════════════════════════════════════════════════════════════════

    def step(
        self,
        observation: Any,
        reward: float = 0.0,
        done: bool = False,
        info: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Execute one full cognitive cycle.

        Args:
          observation : raw input (str, dict, numeric array)
          reward      : external reward from environment
          done        : episode terminal flag
          info        : extra info dict

        Returns:
          dict with keys: action, output, confidence, surprise, neuromod, ...
        """
        t0 = time.perf_counter()
        self._step += 1
        info = info or {}
        self.ist.clear_cycle()
        self.ist.step = self._step
        self.ist.timestamp = time.time()
        self.ist.reward    = reward

        # ─────────────────────────────────────────────────────────────────────
        # Phase 1: Perception
        # ─────────────────────────────────────────────────────────────────────
        obs_text = observation if isinstance(observation, str) else str(observation)
        obs_vec  = self.encoder.encode_obs(observation)
        intent   = self._classify_intent(obs_text)
        sentiment = self._compute_sentiment(obs_text)
        value_conflict = self.identity.detect_value_conflict(obs_text, self.encoder)

        self.ist.raw_input      = observation
        self.ist.obs_vec        = obs_vec
        self.ist.obs_text       = obs_text
        self.ist.intent         = intent
        self.ist.sentiment      = sentiment
        self.ist.value_conflict = value_conflict

        # ─────────────────────────────────────────────────────────────────────
        # Phase 2: World Model — 4-level predictive coding
        # ─────────────────────────────────────────────────────────────────────
        wm_reps, free_energy, surprise = self.world_model.process(obs_vec)
        self.ist.wm_reps        = wm_reps
        self.ist.wm_free_energy = free_energy
        self.ist.free_energy    = free_energy
        self.ist.surprise       = surprise
        self.ist.narrative_state = self.world_model.world_state
        self.ist.self_state      = self.world_model.self_model
        self.ist.predicted_next  = self.world_model.predict_next()

        # Additional surprise from prediction of previous step
        if self._step > 1:
            pred_surprise = Vec.surprise(
                obs_vec, self._prev_obs_vec,
                precision=self.ntx.state.attention_precision * 0.3
            )
            self.ist.surprise = max(self.ist.surprise, pred_surprise * 0.5)

        # ─────────────────────────────────────────────────────────────────────
        # Phase 3: Neuromodulator update
        # ─────────────────────────────────────────────────────────────────────
        self.ntx.on_prediction_error(min(1.0, free_energy / 20.0))
        if surprise > 0.55:
            self.ntx.on_novelty(surprise)
        if value_conflict > 0.4:
            self.ntx.on_threat(value_conflict)
        self.ntx.decay_step()
        mode = self.ntx.state.cognitive_mode
        self.ist.cognitive_mode = mode

        # ─────────────────────────────────────────────────────────────────────
        # Phase 4: Memory Retrieval
        # ─────────────────────────────────────────────────────────────────────
        # Blend obs + wm context for retrieval cue
        wm_ctx = self.memory.wm_context()
        retrieval_cue = Vec.weighted_mean([obs_vec, wm_ctx], [0.70, 0.30])
        self.ist.wm_context = retrieval_cue

        ep_results  = self.memory.recall_episodes(
            retrieval_cue, k=8, session_id=self.session_id
        )
        sem_results = self.memory.recall_concepts(retrieval_cue, k=6)
        self.ist.retrieved_episodes = ep_results
        self.ist.retrieved_concepts  = sem_results

        # Push top episodes into working memory
        for eng, score in ep_results[:3]:
            self.memory.wm_push(eng.uid, eng.content[:80], eng.embedding, importance=score * eng.importance)

        # ─────────────────────────────────────────────────────────────────────
        # Phase 5: Reconsolidation
        # ─────────────────────────────────────────────────────────────────────
        for eng, _ in ep_results[:4]:
            if eng.reconsolidation_open:
                self.memory.reconsolidate(eng.uid, obs_vec, update_factor=0.08)

        # ─────────────────────────────────────────────────────────────────────
        # Phase 6: Nostalgia Check
        # ─────────────────────────────────────────────────────────────────────
        if (intent == IntentClass.NOSTALGIC or
                (mode == CognitiveMode.NOSTALGIC and random.random() < 0.3)):
            trace = self.memory.generate_nostalgia_trace(
                retrieval_cue, theme_text=obs_text[:60], session_id=self.session_id
            )
            self.ist.nostalgia_trace = trace
            self.ntx.on_nostalgia()

        # ─────────────────────────────────────────────────────────────────────
        # Phase 7: Identity Probe
        # ─────────────────────────────────────────────────────────────────────
        if self.ist.self_state is not None:
            self_small = Vec.pad_trim(self.ist.self_state, self.cfg.self_model_dim)
            obs_small  = Vec.pad_trim(obs_vec, self.cfg.self_model_dim)
            self.ist.identity_alignment = (Vec.cosine(obs_small, self_small) + 1.0) / 2.0
        self.identity.update_from_world_model(self.ist.self_state or Vec.zeros(self.cfg.self_model_dim), self._step)

        # ─────────────────────────────────────────────────────────────────────
        # Phase 8: Planning (MCTS)
        # ─────────────────────────────────────────────────────────────────────
        state_vec   = self._build_state_vec()
        n_sims      = self.monitor.recommend_mcts_sims()
        # Use fast policy in REFLEX mode; MCTS otherwise
        if mode == CognitiveMode.REFLEX:
            chosen_action, action_conf = self.cortex.fast_action(state_vec)
            action_probs = [0.0] * len(_ACTIONS)
            action_probs[_ACTION_IDX.get(chosen_action, 0)] = 1.0
        else:
            chosen_action, action_conf, action_probs = self.cortex.select_action_mcts(state_vec, n_sims)

        plan = self.cortex.plan(state_vec, self._step)
        self.ist.planned_actions   = plan
        self.ist.chosen_action     = chosen_action
        self.ist.action_confidence = action_conf

        # Override with NOSTALGIZE if nostalgia trace was generated
        if self.ist.nostalgia_trace and len(self.ist.nostalgia_trace.engram_uids) > 0:
            if intent == IntentClass.NOSTALGIC:
                chosen_action = ActionKind.NOSTALGIZE

        # Override with REFLECT if value conflict is high
        if value_conflict > 0.6 and mode == CognitiveMode.CRITICAL:
            chosen_action = ActionKind.REFLECT

        # ─────────────────────────────────────────────────────────────────────
        # Phase 9 & 10: Execute Action + LLM Generation
        # ─────────────────────────────────────────────────────────────────────
        output_text, confidence = self._execute(chosen_action)
        self.ist.output_text = output_text
        self.ist.confidence  = confidence

        # ─────────────────────────────────────────────────────────────────────
        # Phase 11: Self-Critique (if low confidence)
        # ─────────────────────────────────────────────────────────────────────
        if (confidence < self.cfg.uncertainty_threshold and
                chosen_action in (ActionKind.SPEAK, ActionKind.REFLECT) and
                mode != CognitiveMode.REFLEX):
            output_text, confidence = self._self_critique(obs_text, output_text, ep_results)
            self.ist.output_text = output_text
            self.ist.confidence  = confidence

        # ─────────────────────────────────────────────────────────────────────
        # Phase 12: Reward + Learning
        # ─────────────────────────────────────────────────────────────────────
        next_state_vec = self._build_state_vec()   # updated after action
        td_error = self.cortex.learn(reward, next_state_vec, done)
        self.ist.td_error          = td_error
        self.ist.cumulative_reward += reward
        self.identity.accumulate_reward(reward)
        self.identity.update_surprise_ema(surprise)

        # Neuromod update from reward
        if reward > 0.05:   self.ntx.on_reward(reward)
        elif reward < -0.05: self.ntx.on_negative_rpe(abs(reward))

        # Update TNG Q-values for retrieved episodes
        for eng, _ in ep_results[:3]:
            self.tng.update_q_value(eng.uid, reward)

        # ─────────────────────────────────────────────────────────────────────
        # Phase 13: Memory Write
        # ─────────────────────────────────────────────────────────────────────
        if chosen_action in (ActionKind.SPEAK, ActionKind.REFLECT, ActionKind.STORE,
                              ActionKind.NOSTALGIZE, ActionKind.IMAGINE):
            mem_content = f"Input: {obs_text[:150]}\nOutput: {output_text[:200]}"
            imp = min(1.0, 0.35 + 0.35 * abs(reward) + 0.30 * surprise)
            autobio = min(1.0, 0.2 + 0.5 * abs(reward) + 0.3 * value_conflict)
            # Determine narrative role
            narrative_role = (
                "turning_point" if abs(reward) > 0.8 else
                "insight"       if chosen_action == ActionKind.REFLECT else
                "decision"      if chosen_action == ActionKind.PLAN else
                "event"
            )
            ep_uid = self.memory.store_episode(
                content=mem_content,
                embedding=obs_vec,
                kind=MemoryKind.EPISODIC,
                session_id=self.session_id,
                importance=imp,
                emotional_valence=sentiment,
                arousal=self.ntx.state.NE,
                surprise=surprise,
                autobio_salience=autobio,
                narrative_role=narrative_role,
                tags=[intent.value, chosen_action.value, mode.value],
                source_step=self._step,
            )
            # Record formative event if highly significant
            if imp > 0.75 or abs(reward) > 0.7:
                self.identity.record_formative_event(
                    description=obs_text[:120],
                    embedding=obs_vec,
                    significance=imp,
                    event_type=narrative_role,
                    emotion=sentiment,
                    engram_uids=[ep_uid],
                )
            # Hebbian co-activation
            all_uids = [e.uid for e, _ in ep_results[:4]] + [ep_uid]
            self.memory.hebbian_coactivate(all_uids, delta=0.06 * self.ntx.state.ACH)
            # Concept extraction
            self.memory.extract_concepts_from_text(
                obs_text + " " + output_text, self.encoder
            )
            # Add reflection edge if this was a reflection
            if chosen_action == ActionKind.REFLECT and ep_results:
                self.tng.add_reflection_edge(
                    ep_uid, ep_results[0][0].uid,
                    insight=output_text[:150]
                )

        # ─────────────────────────────────────────────────────────────────────
        # Phase 14: Telemetry + Consolidation
        # ─────────────────────────────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.ist.cycle_ms = elapsed_ms
        cog_load = min(1.0, len(self.ist.retrieved_episodes) / 8.0)
        self.ist.cognitive_load = cog_load

        uncertainty = self.monitor.uncertainty_score(free_energy, surprise)
        self.ist.uncertainty = uncertainty

        metrics = CognitiveMetrics(
            step=self._step, timestamp=time.time(),
            action=chosen_action.value, reward=reward,
            td_error=td_error, free_energy=free_energy,
            surprise=surprise, confidence=confidence,
            intent=intent.value, cognitive_mode=mode.value,
            cycle_ms=elapsed_ms,
            memory_accesses=len(ep_results),
            ntx_snapshot=self.ntx.snapshot(),
            wm_slots=len(self.memory.wm_contents()),
        )
        self.monitor.record(metrics)

        # Periodic consolidation
        if self._step % self.cfg.consolidation_interval == 0:
            self._run_consolidation()

        # Update conversation history for LLM context
        self._conversation_history.append({"role": "user", "content": obs_text})
        self._conversation_history.append({"role": "assistant", "content": output_text[:400]})
        if len(self._conversation_history) > 30:
            self._conversation_history = self._conversation_history[-20:]

        # Save prev state
        self._prev_obs_vec   = obs_vec
        self._prev_state_vec = state_vec

        self.log.info(
            "Step %4d │ act=%-12s reward=%+.3f FE=%.3f surp=%.3f conf=%.3f mode=%-10s │ %.0fms",
            self._step, chosen_action.value, reward, free_energy,
            surprise, confidence, mode.value, elapsed_ms,
        )

        return {
            "action":      chosen_action.value,
            "output":      output_text,
            "confidence":  round(confidence, 4),
            "surprise":    round(surprise, 4),
            "free_energy": round(free_energy, 4),
            "td_error":    round(td_error, 4),
            "intent":      intent.value,
            "mode":        mode.value,
            "neuromod":    self.ntx.snapshot(),
            "retrieved":   [e.content[:80] for e, _ in ep_results[:3]],
            "plan":        [a.value for a in plan[:3]],
        }

    # ══════════════════════════════════════════════════════════════════════════
    # §17.2  ACTION EXECUTION
    # ══════════════════════════════════════════════════════════════════════════

    def _execute(self, action: ActionKind) -> Tuple[str, float]:
        """Dispatch to specific action handler."""
        handlers = {
            ActionKind.SPEAK:      self._action_speak,
            ActionKind.RECALL:     self._action_recall,
            ActionKind.REFLECT:    self._action_reflect,
            ActionKind.PLAN:       self._action_plan,
            ActionKind.IMAGINE:    self._action_imagine,
            ActionKind.STORE:      self._action_store,
            ActionKind.CONSOLIDATE: self._action_consolidate,
            ActionKind.SEARCH:     self._action_search,
            ActionKind.NOSTALGIZE: self._action_nostalgize,
            ActionKind.NOOP:       lambda: ("[No action]", 0.5),
        }
        handler = handlers.get(action, lambda: (f"[Unknown action: {action}]", 0.3))
        try:
            return handler()
        except Exception as exc:
            self.log.exception("Action %s failed: %s", action.value, exc)
            return f"[Action error: {exc}]", 0.2

    def _build_llm_system_prompt(self, extra: str = "") -> str:
        """Build a rich, context-aware system prompt."""
        ep_strs = "\n".join(
            f"  [{i+1}] {e.content[:100]} (imp={e.importance:.2f}, role={e.narrative_role})"
            for i, (e, _) in enumerate(self.ist.retrieved_episodes[:4])
        ) or "  (no relevant episodes)"
        sem_strs = ", ".join(c.label for c, _ in self.ist.retrieved_concepts[:6]) or "(none)"
        wm_strs  = "\n".join(
            f"  [{s['content'][:60]}]" for s in self.memory.wm_contents()[:3]
        ) or "  (empty)"
        ntx = self.ntx.snapshot()

        self_desc = self.identity.self_description()

        nostalgia_str = ""
        if self.ist.nostalgia_trace:
            nt = self.ist.nostalgia_trace
            nostalgia_str = (
                f"\nNostalgia trace — theme: '{nt.theme}'\n"
                f"  Emotional tone: {nt.emotional_tone:.2f}, richness: {nt.richness:.2f}\n"
                f"  Insights: {'; '.join(nt.insights[:3])}"
            )

        return (
            f"{self_desc}\n\n"
            f"Current cognitive state:\n"
            f"  Step: {self._step} | Mode: {self.ist.cognitive_mode.value} | "
            f"Intent: {self.ist.intent.value}\n"
            f"  Free energy: {self.ist.free_energy:.3f} | "
            f"Surprise: {self.ist.surprise:.3f} | "
            f"Confidence: {self.ist.confidence:.3f}\n"
            f"  Identity alignment: {self.ist.identity_alignment:.3f} | "
            f"Value conflict: {self.ist.value_conflict:.3f}\n"
            f"  DA={ntx['levels'].get('DA',0):.2f} NE={ntx['levels'].get('NE',0):.2f} "
            f"ACH={ntx['levels'].get('ACH',0):.2f} HT={ntx['levels'].get('HT',0):.2f}\n\n"
            f"Retrieved episodic memory:\n{ep_strs}\n\n"
            f"Active semantic concepts: {sem_strs}\n\n"
            f"Working memory:\n{wm_strs}"
            f"{nostalgia_str}\n\n"
            f"{extra}\n\n"
            "Respond accurately and concisely. "
            "Ground your answer in the retrieved memory context. "
            "Acknowledge genuine uncertainty. Do not fabricate facts."
        )

    def _action_speak(self) -> Tuple[str, float]:
        """Generate a SPEAK response via LLM."""
        system = self._build_llm_system_prompt()
        user   = self.ist.obs_text

        # Build conversation context (last N turns)

        # Chain-of-thought for ANALYTICAL or CRITICAL mode
        if self.ist.cognitive_mode == CognitiveMode.CRITICAL:
            user = (
                f"Please reason step-by-step before answering:\n{user}"
            )
        elif self.ist.intent == IntentClass.ANALYTICAL:
            user = f"Think carefully and reason step-by-step:\n{user}"

        resp = self.llm.generate(
            system=system, user=user,
            mode=self.ist.cognitive_mode, intent=self.ist.intent,
        )
        return resp.text or "[No response generated]", resp.confidence

    def _action_recall(self) -> Tuple[str, float]:
        """Extended memory recall: spreading activation + nostalgia."""
        results = self.ist.retrieved_episodes
        if not results:
            return "No relevant memories found.", 0.4
        spread_uid = results[0][0].uid
        spread = self.tng.spreading_activation(spread_uid, hops=2, decay=0.7)
        extra_engrams = []
        for uid, strength in sorted(spread.items(), key=lambda x: -x[1])[:5]:
            eng = self.tng._load_engram(uid)
            if eng: extra_engrams.append((eng, strength))
        parts = [f"[{e.content[:100]}] (strength={s:.2f})" for e, s in extra_engrams[:6]]
        return "Memory recall:\n" + "\n".join(parts), 0.65

    def _action_reflect(self) -> Tuple[str, float]:
        """Self-reflection cycle: introspect on recent experience."""
        system = self._build_llm_system_prompt(
            extra=(
                "Perform a deep self-reflection on the current state and recent experience. "
                "Identify any contradictions, surprises, or opportunities for learning. "
                "Be honest about uncertainty and limitations."
            )
        )
        reflect_prompt = (
            f"Reflect on: '{self.ist.obs_text}'\n"
            f"In context of your recent experience and your identity as {self.cfg.agent_name}.\n"
            f"What have you learned? What remains uncertain? What would you do differently?"
        )
        resp = self.llm.generate(system=system, user=reflect_prompt,
                                  mode=CognitiveMode.CRITICAL, intent=IntentClass.METACOGNITIVE)
        return resp.text or "[Reflection failed]", resp.confidence

    def _action_plan(self) -> Tuple[str, float]:
        """Output the current multi-step plan."""
        plan = self.ist.planned_actions or []
        plan_str = " → ".join(a.value for a in plan[:self.cfg.plan_horizon])
        wm = [s["content"][:50] for s in self.memory.wm_contents()[:3]]
        return (
            f"[PLAN] {plan_str}\n"
            f"Working memory context: {' | '.join(wm) or '(empty)'}\n"
            f"Next action: {plan[0].value if plan else 'speak'}"
        ), 0.60

    def _action_imagine(self) -> Tuple[str, float]:
        """Mental simulation / counterfactual reasoning."""
        system = self._build_llm_system_prompt(
            extra=(
                "Engage in mental simulation: imagine alternative scenarios, "
                "counterfactual outcomes, and creative possibilities. "
                "This is your imagination mode — be creative and exploratory."
            )
        )
        imagine_prompt = (
            f"Imagine and explore: '{self.ist.obs_text}'\n"
            "What alternative scenarios are possible? What creative connections emerge?"
        )
        resp = self.llm.generate(system=system, user=imagine_prompt,
                                  mode=CognitiveMode.EXPANSIVE, intent=IntentClass.CREATIVE)
        return resp.text or "[Imagination failed]", resp.confidence * 0.85

    def _action_store(self) -> Tuple[str, float]:
        """Explicitly store current observation to long-term memory."""
        uid = self.memory.store_episode(
            content=self.ist.obs_text,
            embedding=self.ist.obs_vec,
            session_id=self.session_id,
            importance=0.75,
            emotional_valence=self.ist.sentiment,
            narrative_role="event",
            tags=["explicit_store"],
            source_step=self._step,
        )
        return f"[Stored] '{self.ist.obs_text[:80]}' → {uid[:12]}", 0.90

    def _action_consolidate(self) -> Tuple[str, float]:
        """Trigger an explicit consolidation pass."""
        stats = self.tng.consolidation_pass()
        return f"[Consolidation] {stats}", 0.95

    def _action_search(self) -> Tuple[str, float]:
        """Semantic search in concept graph."""
        sem = self.ist.retrieved_concepts
        if not sem: return "No relevant semantic concepts found.", 0.4
        top_concepts = [
            f"{c.label} (freq={c.frequency}, conf={c.confidence:.2f})"
            for c, _ in sem[:8]
        ]
        return "Semantic search results:\n" + "\n".join(f"  • {c}" for c in top_concepts), 0.70

    def _action_nostalgize(self) -> Tuple[str, float]:
        """
        Nostalgia replay: narrate a journey through past engrams.
        Uses LLM to weave the engrams into a coherent narrative.
        """
        nt = self.ist.nostalgia_trace
        if not nt or not nt.engram_uids:
            return "No nostalgia trace available.", 0.3
        # Build narrative context
        engram_texts = []
        for uid in nt.engram_uids[:8]:
            eng = self.tng._load_engram(uid)
            if eng:
                age_h = eng.age_h
                age_str = (f"{age_h:.0f}h ago" if age_h < 48 else
                           f"{age_h/24:.1f}d ago")
                engram_texts.append(f"[{age_str}] {eng.content[:100]}")
        mem_narrative = "\n".join(engram_texts)
        insights_str  = "; ".join(nt.insights[:3]) or "(none extracted)"
        system = self._build_llm_system_prompt(
            extra=(
                f"You are engaging in nostalgic reflection about the theme: '{nt.theme}'.\n"
                f"Emotional tone: {nt.emotional_tone:.2f} | Richness: {nt.richness:.2f}\n"
                f"Weave these memories into a coherent, meaningful narrative. "
                f"Extract insights and connections. Be reflective and authentic."
            )
        )
        nostalgize_prompt = (
            f"Engage with this nostalgic journey about: '{nt.theme}'\n\n"
            f"Memory trace (chronological):\n{mem_narrative}\n\n"
            f"Previously extracted insights: {insights_str}\n\n"
            "Narrate this journey, connecting the memories into a coherent story. "
            "What patterns emerge? What has changed? What endures?"
        )
        resp = self.llm.generate(
            system=system, user=nostalgize_prompt,
            mode=CognitiveMode.NOSTALGIC, intent=IntentClass.NOSTALGIC,
        )
        return resp.text or "[Nostalgia narrative failed]", resp.confidence * 0.9

    def _self_critique(
        self, original_input: str, draft_output: str,
        ep_results: List[Tuple[Engram, float]]
    ) -> Tuple[str, float]:
        """
        Self-critique loop: evaluate draft response and refine if needed.
        Implements a simplified 'Constitutional AI'-like self-correction.
        """
        critique_system = (
            f"You are {self.cfg.agent_name}'s inner critic. "
            "Evaluate the following response for accuracy, completeness, and alignment with "
            "the available memory context. If the response is inadequate, provide a better one."
        )
        ep_ctx = "\n".join(f"- {e.content[:80]}" for e, _ in ep_results[:3])
        critique_prompt = (
            f"Original question: {original_input}\n\n"
            f"Draft response: {draft_output}\n\n"
            f"Available memory context:\n{ep_ctx}\n\n"
            "Critique: Is this response accurate and well-grounded? "
            "If not, provide a corrected version prefixed with 'REVISED:'. "
            "If it is good, reply with 'APPROVED: ' followed by the response."
        )
        resp = self.llm.generate(
            system=critique_system, user=critique_prompt,
            mode=CognitiveMode.CRITICAL, intent=IntentClass.METACOGNITIVE,
            use_cache=False,
        )
        text = resp.text.strip()
        if text.startswith("REVISED:"):
            return text[len("REVISED:"):].strip(), min(resp.confidence + 0.1, 0.95)
        elif text.startswith("APPROVED:"):
            return text[len("APPROVED:"):].strip() or draft_output, resp.confidence + 0.05
        return draft_output, self.ist.confidence

    # ══════════════════════════════════════════════════════════════════════════
    # §17.3  HELPER METHODS
    # ══════════════════════════════════════════════════════════════════════════

    def _build_state_vec(self) -> Any:
        """
        Combine observation, narrative state, WM context, and self-model
        into a unified state vector for the actor-critic / MCTS.
        """
        obs   = Vec.pad_trim(self.ist.obs_vec or Vec.zeros(self.cfg.embed_dim), self.cfg.embed_dim)
        narr  = Vec.pad_trim(self.ist.narrative_state or Vec.zeros(self.cfg.wm_dims[2]), self.cfg.embed_dim)
        wm    = Vec.pad_trim(self.ist.wm_context or Vec.zeros(self.cfg.embed_dim), self.cfg.embed_dim)
        return Vec.normalize(Vec.weighted_mean([obs, narr, wm], [0.55, 0.25, 0.20]))

    def _classify_intent(self, text: str) -> IntentClass:
        """Rule-based intent classifier (fast, no LLM needed)."""
        t = text.lower()
        if any(k in t for k in ["remember when", "do you recall", "look back", "think about the time",
                                  "nostalgia", "hoài niệm", "nhớ lại", "kể lại"]): return IntentClass.NOSTALGIC
        if any(k in t for k in ["how to", "steps to", "procedure", "implement", "install"]): return IntentClass.PROCEDURAL
        if any(k in t for k in ["why","analyze","analyse","compare","evaluate","reason"]): return IntentClass.ANALYTICAL
        if any(k in t for k in ["write","create","story","poem","design","imagine","compose"]): return IntentClass.CREATIVE
        if any(k in t for k in ["feel","emotion","sad","anxious","happy","worried","upset"]): return IntentClass.EMOTIONAL
        if any(k in t for k in ["you are","your mind","are you","conscious","self","identity"]): return IntentClass.METACOGNITIVE
        if any(k in t for k in ["hi","hello","hey","thanks","bye","goodbye"]): return IntentClass.SOCIAL
        if any(k in t for k in ["test","trick","gotcha","wrong","mistake","actually"]): return IntentClass.ADVERSARIAL
        if "?" in t: return IntentClass.FACTUAL
        return IntentClass.AMBIGUOUS

    def _compute_sentiment(self, text: str) -> float:
        """Fast keyword-based sentiment estimation."""
        t = text.lower()
        pos = sum(1 for w in ["good","great","excellent","love","happy","thank","wonderful","amazing"] if w in t)
        neg = sum(1 for w in ["bad","terrible","hate","sad","angry","awful","horrible","fail","error"] if w in t)
        return max(-1.0, min(1.0, (pos - neg) * 0.25))

    def _run_consolidation(self) -> None:
        """Offline consolidation triggered periodically."""
        self.log.info("💤 Consolidation at step %d", self._step)
        self.ntx.on_consolidation()
        stats = self.tng.consolidation_pass()
        self.identity.save()
        self.log.info("Consolidation complete: %s", stats)

    # ══════════════════════════════════════════════════════════════════════════
    # §17.4  PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def chat(self, message: str, reward: float = 0.0) -> str:
        """Simplified single-turn chat interface."""
        result = self.step(message, reward=reward)
        return result["output"]

    def inject_knowledge(
        self, text: str, importance: float = 0.8, tags: Optional[List[str]] = None,
        kind: MemoryKind = MemoryKind.SEMANTIC,
    ) -> str:
        """Directly encode a fact into long-term memory."""
        vec = self.encoder.encode(text)
        uid = self.memory.store_episode(
            content=text, embedding=vec, kind=kind,
            session_id=self.session_id, importance=importance,
            tags=tags or ["injected"], source_step=self._step,
        )
        label_tokens = self.encoder.tokenise(text)[:3]
        for token in label_tokens:
            self.memory.learn_concept(token, self.encoder.encode(token))
        self.log.info("Knowledge injected: '%s' → %s", text[:60], uid[:12])
        return uid

    def search_memory(self, query: str, k: int = 6, kind: Optional[MemoryKind] = None) -> List[str]:
        """Semantic search over episodic memory."""
        vec    = self.encoder.encode(query)
        kinds  = [kind] if kind else None
        results = self.tng.retrieve(vec, k=k, session_id=self.session_id, kinds=kinds)
        return [e.content for e, _ in results]

    def nostalgize(self, theme: str, k: int = 8) -> str:
        """Generate a nostalgia trace and narrate it."""
        theme_vec = self.encoder.encode(theme)
        trace     = self.memory.generate_nostalgia_trace(
            theme_vec, theme_text=theme, session_id=self.session_id
        )
        self.ist.nostalgia_trace = trace
        output, _ = self._action_nostalgize()
        return output

    def add_goal(self, description: str, priority: float = 0.7) -> str:
        """Add a goal as a high-importance semantic memory."""
        uid = self.inject_knowledge(
            f"GOAL: {description}", importance=priority,
            tags=["goal"], kind=MemoryKind.PROCEDURAL,
        )
        return uid

    def reset_working_memory(self) -> None:
        self.memory.wm_clear()
        self.log.info("Working memory cleared")

    def introspect(self) -> Dict[str, Any]:
        """Return a full snapshot of the agent's internal state."""
        mem_stats  = self.memory.memory_stats()
        ntx_snap   = self.ntx.snapshot()
        monitor_sum = self.monitor.summary()
        llm_stats  = self.llm.stats()
        return {
            "version":    "1.0.0",
            "codename":   "NOESIS PRIME",
            "agent_name": self.cfg.agent_name,
            "session_id": self.session_id,
            "step":       self._step,
            "identity": {
                "name":       self.identity.model.agent_name,
                "steps":      self.identity.model.step_count,
                "sessions":   self.identity.model.session_count,
                "total_reward": round(self.identity.model.total_reward, 3),
                "personality": self.identity.model.personality,
                "formative_events": len(self.identity.model.formative_events),
            },
            "cognition": {
                "mode":             self.ist.cognitive_mode.value,
                "intent":           self.ist.intent.value,
                "confidence":       round(self.ist.confidence, 4),
                "uncertainty":      round(self.ist.uncertainty, 4),
                "surprise":         round(self.ist.surprise, 4),
                "free_energy":      round(self.ist.free_energy, 4),
                "identity_alignment": round(self.ist.identity_alignment, 4),
                "value_conflict":   round(self.ist.value_conflict, 4),
                "sentiment":        round(self.ist.sentiment, 4),
            },
            "world_model": {
                "level_free_energies": self.world_model.level_free_energies(),
                "narrative_state_norm": round(Vec.norm(self.world_model.world_state), 4) if self.world_model.world_state is not None else 0,
            },
            "neuromod":   ntx_snap,
            "memory":     mem_stats,
            "performance": monitor_sum,
            "llm":        llm_stats,
            "encoder":    self.encoder.cache_stats,
            "planned":    [a.value for a in self.ist.planned_actions[:4]],
            "cumulative_reward": round(self.ist.cumulative_reward, 4),
        }

    def save(self) -> None:
        """Persist identity and any in-memory state."""
        self.identity.save()
        self.log.info("Agent state saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §18  SESSION MANAGER — multi-session orchestration
# ═══════════════════════════════════════════════════════════════════════════════

class SessionManager:
    """
    Manages multiple NOESIS agent sessions.
    The TNG is shared (global neocortex analogy — knowledge is universal),
    but each session has its own IST, WM, and policy.
    """

    def __init__(self, cfg: Optional[NOESISConfig] = None):
        self.cfg  = cfg or get_cfg()
        self.log  = ContextualLogger("Sessions")
        self._lock = threading.Lock()
        self._sessions: Dict[str, NOESISAgent] = {}
        self._shared_tng: Optional[TemporalNarrativeGraph] = None

    def get_or_create(self, session_id: str = "default") -> NOESISAgent:
        with self._lock:
            if session_id not in self._sessions:
                agent = NOESISAgent(self.cfg, session_id=session_id)
                self._sessions[session_id] = agent
                self.log.info("Session created: %s", session_id)
            return self._sessions[session_id]

    def delete(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].save()
                del self._sessions[session_id]
                return True
        return False

    def list_sessions(self) -> List[Dict]:
        with self._lock:
            return [
                {
                    "session_id": sid,
                    "steps": a._step,
                    "wm_slots": len(a.memory.wm_contents()),
                    "mode": a.ist.cognitive_mode.value,
                }
                for sid, a in self._sessions.items()
            ]

    def save_all(self) -> None:
        with self._lock:
            for agent in self._sessions.values():
                agent.save()

# ═══════════════════════════════════════════════════════════════════════════════
# §19  OPTIONAL FASTAPI REST SERVER
# ═══════════════════════════════════════════════════════════════════════════════

def build_api(mgr: SessionManager) -> Optional[Any]:
    if not HAS_FASTAPI: return None

    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from pydantic import Field as F

    app = FastAPI(
        title="NOESIS PRIME API",
        version="1.0.0",
        description="Narrative-Oriented Emergent Intelligence System — REST API",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    class StepReq(BaseModel):
        session_id: str = "default"
        observation: str
        reward: float = 0.0
        done: bool = False

    class ChatReq(BaseModel):
        session_id: str = "default"
        message: str
        reward: float = 0.0

    class InjectReq(BaseModel):
        session_id: str = "default"
        text: str
        importance: float = F(0.8, ge=0.0, le=1.0)
        tags: List[str] = []

    class SearchReq(BaseModel):
        session_id: str = "default"
        query: str
        k: int = F(6, ge=1, le=50)

    class NostalgiaReq(BaseModel):
        session_id: str = "default"
        theme: str
        k: int = F(8, ge=1, le=30)

    class GoalReq(BaseModel):
        session_id: str = "default"
        description: str
        priority: float = F(0.7, ge=0.0, le=1.0)

    @app.get("/healthz")
    def health():
        return {"status": "ok", "codename": "NOESIS PRIME", "version": "1.0.0"}

    @app.post("/step")
    def step_ep(req: StepReq):
        agent = mgr.get_or_create(req.session_id)
        result = agent.step(req.observation, reward=req.reward, done=req.done)
        result.pop("plan", None)
        return result

    @app.post("/chat")
    def chat_ep(req: ChatReq):
        agent = mgr.get_or_create(req.session_id)
        output = agent.chat(req.message, reward=req.reward)
        return {"response": output, "session_id": req.session_id}

    @app.post("/inject")
    def inject_ep(req: InjectReq):
        agent = mgr.get_or_create(req.session_id)
        uid = agent.inject_knowledge(req.text, req.importance, req.tags)
        return {"uid": uid}

    @app.post("/search")
    def search_ep(req: SearchReq):
        agent = mgr.get_or_create(req.session_id)
        results = agent.search_memory(req.query, k=req.k)
        return {"results": results}

    @app.post("/nostalgize")
    def nostalgize_ep(req: NostalgiaReq):
        agent = mgr.get_or_create(req.session_id)
        narrative = agent.nostalgize(req.theme, k=req.k)
        return {"narrative": narrative, "theme": req.theme}

    @app.post("/goal")
    def goal_ep(req: GoalReq):
        agent = mgr.get_or_create(req.session_id)
        uid = agent.add_goal(req.description, req.priority)
        return {"goal_uid": uid}

    @app.get("/introspect/{session_id}")
    def introspect_ep(session_id: str):
        agent = mgr.get_or_create(session_id)
        return agent.introspect()

    @app.get("/sessions")
    def sessions_ep():
        return {"sessions": mgr.list_sessions()}

    @app.delete("/sessions/{session_id}")
    def del_session_ep(session_id: str):
        ok = mgr.delete(session_id)
        if not ok: raise HTTPException(404, "Session not found")
        return {"deleted": True}

    @app.websocket("/ws/chat")
    async def ws_chat(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                raw = await ws.receive_text()
                data = json.loads(raw)
                session_id = data.get("session_id", "default")
                message    = data.get("message", "")
                reward     = float(data.get("reward", 0.0))
                agent = mgr.get_or_create(session_id)
                result = agent.step(message, reward=reward)
                result.pop("plan", None)
                await ws.send_text(json.dumps(result))
        except WebSocketDisconnect:
            pass
        except Exception:
            await ws.close()

    return app

# ═══════════════════════════════════════════════════════════════════════════════
# §20  DEMONSTRATION RUNNER — automated showcase
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_FACTS = [
    "The hippocampus is critical for episodic memory formation and spatial navigation.",
    "Dopamine encodes reward prediction errors in the basal ganglia (Schultz 1997).",
    "Predictive coding minimizes free energy through hierarchical prediction error signals.",
    "Reconsolidation allows retrieved memories to be modified before re-stabilisation.",
    "The prefrontal cortex supports working memory, planning, and executive control.",
    "Norepinephrine controls arousal and cognitive gain modulation (Dayan & Yu 2006).",
    "The thalamus acts as a selective relay for cortical information routing.",
    "Hebbian plasticity: 'Cells that fire together, wire together' (Hebb 1949).",
    "Memory consolidation involves hippocampal→neocortical transfer during slow-wave sleep.",
    "Active inference frames cognition as minimising variational free energy.",
    "MCTS (Monte Carlo Tree Search) enables superhuman planning in AlphaGo/AlphaZero.",
    "Acetylcholine modulates the balance between expected and unexpected uncertainty.",
    "The BCM (Bienenstock-Cooper-Munro) rule implements synaptic stability via a sliding threshold.",
    "Engram cells are sparse populations of neurons whose reactivation reconstitutes memories.",
    "World models allow agents to imagine future consequences of their actions.",
    "Temporal difference learning bootstraps value estimates from future predictions.",
    "Metaplasticity: the history of synaptic activity modulates future plasticity.",
    "Serotonin (5-HT) regulates patience, impulsivity, and the stability-plasticity tradeoff.",
    "The global workspace theory proposes that consciousness arises from a shared broadcast.",
    "Spreading activation retrieves semantically related concepts through associative networks.",
]

DEMO_QUERIES = [
    "What is the role of the hippocampus in memory?",
    "How does dopamine relate to reward learning?",
    "Explain predictive coding and free energy minimization.",
    "What happens when a memory is retrieved? Can it change?",
    "How does MCTS planning work in AI systems?",
    "Can you reflect on what you've been learning in this session?",
    "Remember when we talked about the hippocampus — can you revisit that?",
    "What is the relationship between acetylcholine and uncertainty?",
    "Imagine a future where AI systems have genuine episodic memory. What would change?",
    "Analyze the similarities between Hebbian learning and modern deep learning.",
]

def run_demo(
    n_queries: int = 8,
    verbose: bool = True,
    embed_dim: int = 256,
) -> None:
    print("═" * 68)
    print("  NOESIS PRIME v1.0 — Narrative-Oriented Emergent Intelligence")
    print("  Automated Demonstration")
    print("═" * 68)

    # Use smaller dims for demo speed
    cfg = NOESISConfig(
        embed_dim=embed_dim,
        wm_dims=(embed_dim, embed_dim // 2, embed_dim // 4, embed_dim // 8),
        mcts_simulations=16,
        self_model_dim=embed_dim // 8,
        agent_name="Noesis-Demo",
    )
    set_cfg(cfg)
    agent = NOESISAgent(cfg, session_id="demo")

    # ── Phase 1: Knowledge injection ──────────────────────────────────────────
    print(f"\n▶ Injecting {len(DEMO_FACTS)} knowledge facts into long-term memory...")
    for fact in DEMO_FACTS:
        agent.inject_knowledge(fact, importance=0.85, tags=["neuroscience", "ai", "cognition"])
    print(f"  Done. Memory: {agent.memory.memory_stats()['total_engrams']} engrams")

    # ── Phase 2: Add goals ─────────────────────────────────────────────────────
    agent.add_goal("Provide accurate, well-grounded answers about cognitive science", priority=0.85)
    agent.add_goal("Demonstrate authentic memory recall and nostalgia", priority=0.70)
    print("  Goals registered.")

    # ── Phase 3: Interactive demo queries ──────────────────────────────────────
    print(f"\n▶ Running {min(n_queries, len(DEMO_QUERIES))} cognitive cycles...\n")
    total_reward = 0.0
    for i, query in enumerate(DEMO_QUERIES[:n_queries]):
        reward = random.uniform(0.2, 0.9)   # simulate external feedback
        result = agent.step(query, reward=reward)
        total_reward += reward
        if verbose:
            print(f"\n  ─── Step {i+1} ───")
            print(f"  Q: {query}")
            print(f"  Mode: {result['mode']} | Action: {result['action']} | "
                  f"Intent: {result['intent']}")
            print(f"  Conf={result['confidence']:.3f} | Surp={result['surprise']:.3f} | "
                  f"FE={result['free_energy']:.3f}")
            output = result["output"]
            print(f"  A: {output[:400]}{'...' if len(output) > 400 else ''}")

    # ── Phase 4: Nostalgia demonstration ──────────────────────────────────────
    print("\n▶ Nostalgia demonstration — revisiting 'hippocampus' theme...")
    nostalgia_output = agent.nostalgize("hippocampus memory consolidation", k=6)
    if verbose:
        print(f"  {nostalgia_output[:500]}{'...' if len(nostalgia_output) > 500 else ''}")

    # ── Phase 5: Consolidation ────────────────────────────────────────────────
    print("\n▶ Triggering memory consolidation...")
    agent._run_consolidation()

    # ── Phase 6: Final introspection ──────────────────────────────────────────
    print("\n▶ Final introspection report:")
    intro = agent.introspect()
    print("\n  Identity:")
    id_data = intro["identity"]
    for k, v in id_data.items():
        if not isinstance(v, dict):
            print(f"    {k:25s}: {v}")
    print("\n  Cognition:")
    cog = intro["cognition"]
    for k, v in cog.items():
        print(f"    {k:25s}: {v}")
    print("\n  Neuromodulators:")
    ntx = intro["neuromod"]
    lvls = ntx.get("levels", {})
    for k, v in lvls.items():
        print(f"    {k:6s}: {v:.4f}")
    print(f"    mode  : {ntx.get('mode', '?')}")
    print("\n  Memory:")
    mem = intro["memory"]
    for k, v in mem.items():
        if isinstance(v, (int, float)):
            print(f"    {k:25s}: {v}")
    print("\n  Performance:")
    perf = intro["performance"]
    for k, v in perf.items():
        if not isinstance(v, list):
            print(f"    {k:25s}: {v}")
    print(f"\n  World Model level FEs: {intro['world_model']['level_free_energies']}")
    print(f"  Cumulative reward: {intro['cumulative_reward']}")
    print(f"  LLM stats: {intro['llm']}")

    print("\n" + "═" * 68)
    print(f"  Demonstration complete — {agent._step} steps, total reward={total_reward:.3f}")
    print("═" * 68)

# ═══════════════════════════════════════════════════════════════════════════════
# §21  INTERACTIVE CLI
# ═══════════════════════════════════════════════════════════════════════════════

BANNER = r"""
╔══════════════════════════════════════════════════════════════════════╗
║   N O E S I S   P R I M E   v1.0                                    ║
║   Narrative-Oriented Emergent Intelligence System                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  Commands:                                                           ║
║  <text>              → full cognitive cycle (think + respond)        ║
║  /inject <fact>      → store fact in long-term memory                ║
║  /recall <query>     → search episodic memory                        ║
║  /nostalgize <theme> → generate nostalgia trace on a theme           ║
║  /goal <desc>        → add a goal                                    ║
║  /introspect         → full agent state snapshot                     ║
║  /telemetry          → performance summary                           ║
║  /save               → persist state to disk                         ║
║  /consolidate        → trigger memory consolidation                  ║
║  /reset_wm           → clear working memory                          ║
║  /session <id>       → switch to a different session                 ║
║  /inject_bulk        → enter multi-line bulk injection mode          ║
║  /quit               → exit                                          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

def run_cli(session_id: str = "default", cfg: Optional[NOESISConfig] = None):
    print(BANNER)
    if cfg is None:
        cfg = get_cfg()
    agent = NOESISAgent(cfg, session_id=session_id)

    # Seed with some basic knowledge
    seeds = [
        "NOESIS PRIME is a cognitive agent with episodic memory, predictive coding, and MCTS planning.",
        "The agent can recall past experiences, reflect on them, and generate nostalgia traces.",
        "I value intellectual honesty, precision, and nuanced thinking.",
    ]
    print("Seeding working knowledge...")
    for s in seeds:
        agent.inject_knowledge(s, importance=0.7, tags=["seed"])
    print(f"  {len(seeds)} facts seeded.\n")

    reward = 0.0
    while True:
        try:
            raw = input("noesis> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            agent.save()
            break
        if not raw: continue

        tokens = raw.split(maxsplit=1)
        cmd    = tokens[0].lower()
        rest   = tokens[1].strip() if len(tokens) > 1 else ""

        if cmd in ("/quit", "/exit", "/q"):
            agent.save()
            print("Goodbye.")
            break

        elif cmd == "/inject":
            if not rest: print("Usage: /inject <fact>"); continue
            uid = agent.inject_knowledge(rest, importance=0.85)
            print(f"  Injected → {uid[:16]}")

        elif cmd == "/inject_bulk":
            print("  Enter facts (one per line, blank line to finish):")
            lines, total = [], 0
            while True:
                try:
                    line = input("  > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not line: break
                lines.append(line)
            for line in lines:
                agent.inject_knowledge(line, importance=0.8)
                total += 1
            print(f"  Injected {total} facts.")

        elif cmd in ("/recall", "/search"):
            if not rest: print("Usage: /recall <query>"); continue
            results = agent.search_memory(rest, k=6)
            for i, r in enumerate(results, 1):
                print(f"  [{i}] {r[:120]}")

        elif cmd == "/nostalgize":
            if not rest: print("Usage: /nostalgize <theme>"); continue
            narrative = agent.nostalgize(rest)
            print(f"\n{narrative}\n")

        elif cmd == "/goal":
            if not rest: print("Usage: /goal <description>"); continue
            uid = agent.add_goal(rest, priority=0.75)
            print(f"  Goal added → {uid}")

        elif cmd == "/introspect":
            state = agent.introspect()
            print(json.dumps(state, indent=2, default=str))

        elif cmd == "/telemetry":
            summary = agent.monitor.summary()
            print(json.dumps(summary, indent=2, default=str))

        elif cmd == "/save":
            agent.save()
            print("  State saved.")

        elif cmd == "/consolidate":
            agent._run_consolidation()
            print("  Consolidation done.")

        elif cmd == "/reset_wm":
            agent.reset_working_memory()
            print("  Working memory cleared.")

        elif cmd == "/session":
            if not rest: print("Usage: /session <id>"); continue
            agent.save()
            agent = NOESISAgent(cfg, session_id=rest)
            print(f"  Switched to session: {rest}")

        else:
            # Full cognitive cycle
            result = agent.step(raw, reward=reward)
            mode = result.get("mode", "?")
            act  = result.get("action", "?")
            conf = result.get("confidence", 0.0)
            surp = result.get("surprise", 0.0)
            intent = result.get("intent", "?")
            print(f"\n[{act.upper()} | mode={mode} | intent={intent} | "
                  f"conf={conf:.3f} | surp={surp:.3f}]")
            print(result.get("output", ""))
            if result.get("retrieved"):
                print("\n  Memory context:")
                for r in result["retrieved"][:2]:
                    print(f"    • {r[:80]}")
            print()

            # Optional feedback
            try:
                fb = input("  Reward? [Enter=0, +/good/-/bad/value]: ").strip().lower()
                if fb in ("+", "good", "correct", "yes"): reward = 1.0
                elif fb in ("-", "bad", "wrong", "no"):   reward = -0.5
                elif fb:
                    try:    reward = float(fb)
                    except: reward = 0.0
                else: reward = 0.0
            except (EOFError, KeyboardInterrupt):
                reward = 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# §22  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NOESIS PRIME v1.0 — Narrative-Oriented Emergent Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  demo  - Automated showcase with knowledge injection + queries
  cli   - Interactive CLI (default)
  serve - Start FastAPI server
"""
    )

    parser.add_argument(
        "mode",
        nargs="?",
        choices=["demo", "cli", "serve"],
        default="cli",
        help="Select operation mode: demo, cli, or serve"
    )

    parser.add_argument(
        "--session",
        type=str,
        default="default",
        help="Session ID for CLI mode"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for server mode"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for server mode"
    )

    parser.add_argument(
        "--queries",
        type=int,
        default=8,
        help="Number of queries for demo mode"
    )

    parser.add_argument(
        "--embed-dim",
        type=int,
        default=256,
        help="Embedding dimension (smaller = faster)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output in demo"
    )

    args = parser.parse_args()

    cfg = NOESISConfig(embed_dim=args.embed_dim)

    if args.mode == "demo":
        run_demo(
            n_queries=args.queries,
            verbose=args.verbose,
            embed_dim=args.embed_dim,
        )
    elif args.mode == "serve":
        mgr = SessionManager(cfg)
        app = build_api(mgr)
        if app is None:
            print("FastAPI not installed. Please install fastapi and uvicorn.")
            sys.exit(1)
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
    else:  # cli
        run_cli(session_id=args.session, cfg=cfg)
